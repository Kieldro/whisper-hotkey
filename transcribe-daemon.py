#!/usr/bin/env python3
"""
Background daemon for push-to-talk voice transcription.
More Superwhisper-like UX: hold key to record, release to transcribe.
"""

import os
import sys
import tempfile
import subprocess
import threading
import signal
import logging
import wave
import pyaudio
import fcntl
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Setup logging
LOG_FILE = "/tmp/whisper-hotkey.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
ENABLE_POLISHING = os.getenv("ENABLE_POLISHING", "false").lower() == "true"
AUTO_PASTE = os.getenv("AUTO_PASTE", "true").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "600"))  # seconds
PRE_RECORDING_BUFFER = int(os.getenv("PRE_RECORDING_BUFFER", "2"))  # seconds
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

# Sound file paths (optional - will skip if not found)
SOUND_START = os.getenv("SOUND_START", "/usr/share/sounds/freedesktop/stereo/message-new-instant.oga")
SOUND_STOP = os.getenv("SOUND_STOP", "/usr/share/sounds/freedesktop/stereo/complete.oga")

# Shared notification ID for replacing notifications
NOTIFICATION_ID = 999999


def send_notification(message: str) -> None:
    """Send desktop notification that replaces previous ones."""
    try:
        subprocess.run(
            ['notify-send', '-t', '1500', '-r', str(NOTIFICATION_ID), 'Voice Transcription', message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        pass  # notify-send not installed


def play_sound(sound_file: str) -> None:
    """Play a sound file using paplay (non-blocking, fails silently)."""
    if not sound_file or not os.path.exists(sound_file):
        return
    try:
        subprocess.Popen(
            ['paplay', sound_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        pass  # paplay not installed


# Detect session type for clipboard/typing
def detect_session_type():
    """Detect the actual running display server session."""

    # First, detect the actual running session using environment variables
    session_type = os.getenv("XDG_SESSION_TYPE", "").lower()
    wayland_display = os.getenv("WAYLAND_DISPLAY")
    x11_display = os.getenv("DISPLAY")

    # Determine session from environment
    if session_type == "wayland" or wayland_display:
        detected_session = "wayland"
    elif session_type == "x11" or x11_display:
        detected_session = "x11"
    else:
        # Fallback: couldn't determine, default to X11
        logger.warning("Could not detect session type from environment, defaulting to X11")
        detected_session = "x11"

    # Verify required tools are available for the detected session
    if detected_session == "wayland":
        has_wl_copy = shutil.which("wl-copy") is not None
        has_ydotool = shutil.which("ydotool") is not None
        if not (has_wl_copy and has_ydotool):
            logger.warning("Wayland session detected but missing tools. Install: wl-clipboard ydotool")
    else:  # X11
        has_xclip = shutil.which("xclip") is not None
        has_xdotool = shutil.which("xdotool") is not None
        if not (has_xclip and has_xdotool):
            logger.warning("X11 session detected but missing tools. Install: xclip xdotool")

    return detected_session

SESSION_TYPE = detect_session_type()
logger.info(f"Using clipboard/typing mode: {SESSION_TYPE}")


class PreRecordingBuffer:
    """Continuously buffers audio in memory to capture pre-hotkey speech."""

    def __init__(self, duration: int = PRE_RECORDING_BUFFER, sample_rate: int = SAMPLE_RATE, chunk_size: int = CHUNK_SIZE):
        self.duration = duration
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.enabled = duration > 0

        if not self.enabled:
            logger.info("Pre-recording buffer disabled (duration = 0)")
            return

        # Calculate buffer size (number of chunks to keep)
        chunks_per_second = sample_rate / chunk_size
        self.max_chunks = int(chunks_per_second * duration)

        # Circular buffer for audio chunks
        self.buffer = deque(maxlen=self.max_chunks)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.audio = None
        self.stream = None

        logger.info(f"Pre-recording buffer initialized: {duration}s ({self.max_chunks} chunks)")

    def start(self) -> None:
        """Start continuous background recording."""
        if not self.enabled or self.thread is not None:
            return

        try:
            self.audio = pyaudio.PyAudio()
            # Use smaller buffer and shorter timeout for responsive shutdown
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self._stop_event.clear()
            self.thread = threading.Thread(target=self._record_loop, daemon=True)
            self.thread.start()
            logger.info("Pre-recording buffer started")
        except Exception as e:
            logger.error(f"Failed to start pre-recording buffer: {e}")
            self.enabled = False

    def _record_loop(self) -> None:
        """Continuously record audio into circular buffer."""
        while not self._stop_event.is_set():
            try:
                # Check if stream is still valid
                if self.stream is None or not self.stream.is_active():
                    break
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                with self._lock:
                    self.buffer.append(data)
            except OSError:
                # Stream was closed
                break
            except Exception as e:
                logger.error(f"Error in pre-recording buffer: {e}")
                break

    def get_buffered_audio(self) -> bytes:
        """Get all buffered audio as bytes."""
        if not self.enabled:
            return b''
        with self._lock:
            if not self.buffer:
                return b''
            return b''.join(self.buffer)

    def stop(self) -> None:
        """Stop the buffer and cleanup resources."""
        if not self.enabled:
            return

        # Signal thread to stop
        self._stop_event.set()

        # Close stream first to unblock the read() call
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        # Now join the thread (should exit quickly since stream is closed)
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

        if self.audio:
            try:
                self.audio.terminate()
            except Exception:
                pass
            self.audio = None

        logger.info("Pre-recording buffer stopped")


class AudioRecorderDaemon:
    """Non-blocking audio recorder using parecord."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, pre_buffer: Optional['PreRecordingBuffer'] = None):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.process: Optional[subprocess.Popen] = None
        self.output_file: Optional[str] = None
        self.pre_buffer = pre_buffer
        self.pre_buffer_file: Optional[str] = None

    def start_recording(self) -> bool:
        """Start recording with parecord. Returns True on success."""
        if self.is_recording:
            logger.warning("Already recording, ignoring start request")
            return False

        # Save pre-buffered audio if available
        self.pre_buffer_file = None
        if self.pre_buffer and self.pre_buffer.enabled:
            buffered_audio = self.pre_buffer.get_buffered_audio()
            if buffered_audio:
                # Write buffered audio to temp WAV file
                temp_buffer = tempfile.NamedTemporaryFile(suffix='_buffer.wav', delete=False)
                self.pre_buffer_file = temp_buffer.name
                temp_buffer.close()

                with wave.open(self.pre_buffer_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit = 2 bytes
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(buffered_audio)

                logger.info(f"Saved {len(buffered_audio)} bytes of pre-buffered audio to {self.pre_buffer_file}")

        # Create temp file for new recording
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.output_file = temp_file.name
        temp_file.close()
        logger.info(f"Created temp audio file: {self.output_file}")

        # Start parecord
        logger.info(f"Starting parecord (rate={self.sample_rate})")
        try:
            self.process = subprocess.Popen(
                [
                    'parecord',
                    '--format=s16le',
                    f'--rate={self.sample_rate}',
                    '--channels=1',
                    self.output_file
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Verify process actually started
            time.sleep(0.1)
            if self.process.poll() is not None:
                logger.error(f"parecord exited immediately with code {self.process.returncode}")
                send_notification("❌ Recording failed (parecord error)")
                self._cleanup_temp_files()
                return False
        except FileNotFoundError:
            logger.error("parecord not found. Install pulseaudio-utils.")
            send_notification("❌ parecord not found")
            self._cleanup_temp_files()
            return False
        except Exception as e:
            logger.error(f"Failed to start parecord: {e}")
            send_notification(f"❌ Recording failed: {str(e)[:30]}")
            self._cleanup_temp_files()
            return False

        self.is_recording = True
        logger.info(f"Recording started, parecord PID: {self.process.pid}")
        play_sound(SOUND_START)
        send_notification("🎤 Recording...")
        return True

    def _cleanup_temp_files(self) -> None:
        """Clean up any temp files created during recording."""
        if self.pre_buffer_file and os.path.exists(self.pre_buffer_file):
            try:
                os.unlink(self.pre_buffer_file)
            except OSError:
                pass
            self.pre_buffer_file = None
        if self.output_file and os.path.exists(self.output_file):
            try:
                os.unlink(self.output_file)
            except OSError:
                pass
            self.output_file = None

    def stop_recording(self) -> Optional[str]:
        """Stop recording and return file path."""
        if not self.is_recording:
            logger.warning("Not recording, ignoring stop request")
            return None

        logger.info("Stopping recording...")
        self.is_recording = False

        if self.process:
            logger.info(f"Terminating parecord process {self.process.pid}")
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.warning("parecord didn't terminate, killing...")
                self.process.kill()
                self.process.wait()
            logger.info("parecord terminated")
            self.process = None

        # Merge pre-buffered audio with new recording if needed
        if self.pre_buffer_file and os.path.exists(self.pre_buffer_file):
            logger.info("Merging pre-buffered audio with recording...")
            merged_file = self._merge_audio_files(self.pre_buffer_file, self.output_file)
            if merged_file:
                # Clean up temp files
                try:
                    os.unlink(self.pre_buffer_file)
                except OSError:
                    pass
                try:
                    os.unlink(self.output_file)
                except OSError:
                    pass
                self.output_file = merged_file
                logger.info(f"Merged audio file: {merged_file}")
            self.pre_buffer_file = None

        # Check file size
        if self.output_file and os.path.exists(self.output_file):
            file_size = os.path.getsize(self.output_file)
            logger.info(f"Final audio file: {self.output_file} ({file_size} bytes)")
        else:
            logger.error(f"Audio file not found: {self.output_file}")

        # Play stop sound
        play_sound(SOUND_STOP)
        # Removed "Recording stopped" notification - goes straight to transcribing
        return self.output_file

    def _merge_audio_files(self, file1: str, file2: str) -> Optional[str]:
        """Merge two WAV files into one."""
        merged_file = None
        try:
            # Read first file
            with wave.open(file1, 'rb') as wf1:
                params1 = wf1.getparams()
                frames1 = wf1.readframes(wf1.getnframes())

            # Read second file
            with wave.open(file2, 'rb') as wf2:
                params2 = wf2.getparams()
                frames2 = wf2.readframes(wf2.getnframes())

            # Verify compatibility
            if params1[:3] != params2[:3]:  # channels, sampwidth, framerate
                logger.error("Audio files have incompatible parameters")
                return None

            # Write merged file
            merged_file = tempfile.NamedTemporaryFile(suffix='_merged.wav', delete=False).name
            with wave.open(merged_file, 'wb') as wf_out:
                wf_out.setparams(params1)
                wf_out.writeframes(frames1 + frames2)

            logger.info(f"Successfully merged audio files: {len(frames1)} + {len(frames2)} bytes")
            return merged_file

        except Exception as e:
            logger.error(f"Failed to merge audio files: {e}")
            # Clean up partial merged file on failure
            if merged_file and os.path.exists(merged_file):
                try:
                    os.unlink(merged_file)
                except OSError:
                    pass
            return None

    def cleanup(self) -> None:
        """Cleanup audio resources."""
        if self.process and self.process.poll() is None:
            self.process.terminate()


class TranscriptionPipeline:
    """Handles transcription and polishing pipeline."""

    def __init__(self):
        logger.info(f"Initializing TranscriptionPipeline (model={WHISPER_MODEL}, device={DEVICE}, polishing={ENABLE_POLISHING})")
        # Removed loading notification - loads in background silently

        logger.info("Loading Whisper model...")
        self.whisper = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
        logger.info("Whisper model loaded successfully")

        if ENABLE_POLISHING:
            logger.info("Initializing OpenAI client...")
            self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.openai = None

        logger.info("TranscriptionPipeline initialized")

    def process(self, audio_path: str) -> str:
        """Full pipeline: transcribe + polish."""
        logger.info(f"Processing audio file: {audio_path}")

        # Transcribe (no notification - happens fast)
        logger.info("Starting Whisper transcription...")

        try:
            segments, _ = self.whisper.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Require 500ms silence to split segments
                    speech_pad_ms=2000,  # Pad speech segments by 2s on each side to capture trailing words
                )
            )
            raw_text = " ".join([seg.text.strip() for seg in segments])
            logger.info(f"Transcription complete. Raw text: '{raw_text}'")
        except ValueError as e:
            # Empty audio or no detectable language
            logger.warning(f"ValueError during transcription: {e}")
            send_notification("⚠️  No speech detected (empty audio)")
            return ""
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            send_notification(f"❌ Transcription error: {str(e)[:50]}")
            return ""

        if not raw_text.strip():
            logger.warning("Transcription returned empty text")
            send_notification("⚠️  No speech detected")
            return ""

        # Polish if enabled
        if ENABLE_POLISHING and self.openai:
            logger.info("Polishing with GPT...")
            # No notification for polishing - happens fast
            try:
                response = self.openai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "Fix grammar, punctuation, formatting. Return only corrected text."
                        },
                        {"role": "user", "content": raw_text}
                    ],
                    temperature=0.3
                )
                final_text = response.choices[0].message.content.strip()
                logger.info(f"Polished text: '{final_text}'")
            except Exception as e:
                logger.error(f"GPT polishing error: {e}", exc_info=True)
                final_text = raw_text
        else:
            final_text = raw_text
            logger.info("Skipping polishing (disabled)")

        # Copy to clipboard
        logger.info("Copying to clipboard...")
        if SESSION_TYPE == "wayland":
            subprocess.run(
                ['wl-copy'],
                input=final_text.encode(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:  # X11
            subprocess.run(
                ['xclip', '-selection', 'clipboard'],
                input=final_text.encode(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        logger.info("Copied to clipboard")

        # Auto-paste if enabled
        if AUTO_PASTE:
            time.sleep(0.2)  # Brief delay
            if SESSION_TYPE == "wayland":
                logger.info("Auto-pasting with ydotool type...")
                result = subprocess.run(
                    ['ydotool', 'type', final_text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.info(f"ydotool type exit code: {result.returncode}")
            else:  # X11
                logger.info("Auto-pasting with xdotool type...")
                result = subprocess.run(
                    ['xdotool', 'type', '--clearmodifiers', final_text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.info(f"xdotool type exit code: {result.returncode}")
            send_notification(f"✅ Pasted: {final_text[:50]}...")
        else:
            logger.info("Auto-paste disabled")
            send_notification(f"✅ Copied: {final_text[:50]}...")

        logger.info("Processing complete")
        return final_text


class TranscriptionDaemon:
    """Background daemon handling recording and transcription."""

    def __init__(self):
        # Initialize pre-recording buffer
        self.pre_buffer = PreRecordingBuffer()
        self.pre_buffer.start()

        self.recorder = AudioRecorderDaemon(pre_buffer=self.pre_buffer)
        self.pipeline = TranscriptionPipeline()
        self._audio_lock = threading.Lock()
        self._current_audio_file: Optional[str] = None
        self.last_activity_time = time.time()
        self.idle_timeout = IDLE_TIMEOUT

    def toggle_recording(self) -> None:
        """Toggle recording state (for push-to-talk)."""
        if not self.recorder.is_recording:
            # Start recording
            self.recorder.start_recording()
        else:
            # Stop recording and process
            audio_path = self.recorder.stop_recording()
            if audio_path:
                with self._audio_lock:
                    self._current_audio_file = audio_path
                # Process in background thread
                threading.Thread(target=self._process_audio, daemon=True).start()

    def _process_audio(self) -> None:
        """Process recorded audio in background."""
        # Grab the file path under lock
        with self._audio_lock:
            audio_file = self._current_audio_file
            self._current_audio_file = None

        if not audio_file:
            return

        try:
            result = self.pipeline.process(audio_file)
            if result:
                print(result, flush=True)
        finally:
            # Cleanup temp file
            if os.path.exists(audio_file):
                try:
                    os.unlink(audio_file)
                except OSError:
                    pass

    def start_recording(self) -> bool:
        """Start recording. Returns True on success."""
        return self.recorder.start_recording()

    def stop_and_process(self) -> None:
        """Stop recording and process."""
        logger.info("stop_and_process called")
        self.last_activity_time = time.time()  # Update activity time
        audio_path = self.recorder.stop_recording()
        if audio_path:
            with self._audio_lock:
                self._current_audio_file = audio_path
            logger.info("Processing audio synchronously...")
            # Process synchronously, not in background thread
            self._process_audio()
            logger.info("Processing complete")

    def is_idle_timeout_exceeded(self) -> bool:
        """Check if daemon has been idle too long."""
        if self.idle_timeout == 0:
            return False  # Never timeout if set to 0
        idle_time = time.time() - self.last_activity_time
        return idle_time > self.idle_timeout

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.recorder.cleanup()
        if self.pre_buffer:
            self.pre_buffer.stop()


def acquire_daemon_lock(lock_file: str) -> Optional[int]:
    """
    Try to acquire exclusive lock on daemon file.
    Returns file descriptor on success, None if another daemon holds the lock.
    """
    try:
        fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # We got the lock - write our PID
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode())
            return fd
        except BlockingIOError:
            # Another process holds the lock
            os.close(fd)
            return None
    except OSError:
        return None


def read_daemon_pid(lock_file: str) -> Optional[int]:
    """Read PID from daemon lock file."""
    try:
        with open(lock_file, 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def main():
    """Run daemon in toggle mode (for hotkey integration)."""
    logger.info("=== Daemon started ===")
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Config: ENABLE_POLISHING={ENABLE_POLISHING}, AUTO_PASTE={AUTO_PASTE}, MODEL={WHISPER_MODEL}")

    if ENABLE_POLISHING and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set but polishing is enabled")
        print("❌ Error: OPENAI_API_KEY not set (required when ENABLE_POLISHING=true)", file=sys.stderr)
        sys.exit(1)

    state_file = "/tmp/whisper-hotkey-state"
    daemon_lock_file = "/tmp/whisper-hotkey-daemon.lock"

    # Try to acquire the daemon lock
    lock_fd = acquire_daemon_lock(daemon_lock_file)

    if lock_fd is None:
        # Another daemon is running, send it a signal
        daemon_pid = read_daemon_pid(daemon_lock_file)
        if daemon_pid:
            try:
                # Verify it's still alive
                os.kill(daemon_pid, 0)

                # Daemon is alive, check if we're starting or stopping recording
                if os.path.exists(state_file):
                    # Currently recording, send stop signal
                    logger.info(f"Sending stop signal to daemon PID {daemon_pid}")
                    os.kill(daemon_pid, signal.SIGUSR1)
                    try:
                        os.unlink(state_file)
                    except OSError:
                        pass
                else:
                    # Start new recording
                    logger.info(f"Sending start signal to daemon PID {daemon_pid}")
                    Path(state_file).touch()
                    os.kill(daemon_pid, signal.SIGUSR2)
                sys.exit(0)

            except ProcessLookupError:
                # Daemon died but still holds lock (shouldn't happen with flock)
                logger.warning("Daemon PID exists but process is dead")
                # Try again to acquire lock
                lock_fd = acquire_daemon_lock(daemon_lock_file)
                if lock_fd is None:
                    logger.error("Could not acquire daemon lock")
                    sys.exit(1)
            except OSError as e:
                logger.error(f"Failed to signal daemon: {e}")
                sys.exit(1)
        else:
            logger.error("Could not read daemon PID")
            sys.exit(1)

    # We now hold the lock
    logger.info(f"Acquired daemon lock, PID written to {daemon_lock_file}")

    # Signal handlers - use threading.Event for thread-safe signaling
    # NOTE: Signal handlers should only set flags, not call complex functions like logging
    start_recording_event = threading.Event()
    stop_recording_event = threading.Event()
    shutdown_event = threading.Event()

    def handle_start(signum, frame):
        # UNSAFE to log here - just set the flag
        start_recording_event.set()

    def handle_stop(signum, frame):
        # UNSAFE to log here - just set the flag
        stop_recording_event.set()

    def handle_shutdown(signum, frame):
        # UNSAFE to log here - just set the flag
        shutdown_event.set()

    signal.signal(signal.SIGUSR2, handle_start)  # Start recording
    signal.signal(signal.SIGUSR1, handle_stop)   # Stop recording
    signal.signal(signal.SIGTERM, handle_shutdown)  # Shutdown daemon
    signal.signal(signal.SIGINT, handle_shutdown)   # Ctrl+C
    logger.info("Signal handlers registered")

    # Start persistent daemon (slow: loads Whisper model)
    logger.info("Starting persistent daemon (will stay alive for 10 minutes after last use)")
    logger.info("Loading Whisper model... (this may take a few seconds on first start)")
    daemon = TranscriptionDaemon()

    # Immediately start recording (since we're starting fresh daemon)
    logger.info("Auto-starting recording on fresh daemon start")
    Path(state_file).touch()  # Mark as recording
    daemon.start_recording()
    daemon.last_activity_time = time.time()

    try:
        logger.info("Daemon loop started, model loaded and ready")

        while not shutdown_event.is_set():
            # Check for start recording signal
            if start_recording_event.is_set():
                start_recording_event.clear()
                logger.info("Received start recording signal")
                daemon.start_recording()
                daemon.last_activity_time = time.time()

            # Check for stop recording signal
            if stop_recording_event.is_set():
                stop_recording_event.clear()
                logger.info("Received stop recording signal")
                daemon.stop_and_process()

            # Check for idle timeout
            if daemon.is_idle_timeout_exceeded():
                logger.info(f"Idle timeout exceeded ({daemon.idle_timeout}s), shutting down daemon")
                break

            # Wait for signals or timeout (efficient polling)
            # Wakes immediately on signals, sleeps otherwise
            shutdown_event.wait(timeout=0.1)

        logger.info("Daemon loop exiting")

    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up...")
        daemon.cleanup()
        try:
            if os.path.exists(state_file):
                os.unlink(state_file)
                logger.info("Removed state file")
        except OSError:
            pass
        # Release the lock by closing the file descriptor
        if lock_fd is not None:
            try:
                os.close(lock_fd)
                logger.info("Released daemon lock")
            except OSError:
                pass
        # Remove lock file
        try:
            if os.path.exists(daemon_lock_file):
                os.unlink(daemon_lock_file)
                logger.info("Removed daemon lock file")
        except OSError:
            pass
        logger.info("=== Daemon exiting ===")


if __name__ == "__main__":
    main()
