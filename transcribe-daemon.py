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
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

# Detect session type for clipboard/typing
def detect_session_type():
    """Detect if running under X11 or Wayland."""
    session_type = os.getenv("XDG_SESSION_TYPE", "").lower()
    wayland_display = os.getenv("WAYLAND_DISPLAY", "")

    if session_type == "wayland" or wayland_display:
        return "wayland"
    return "x11"

SESSION_TYPE = detect_session_type()
logger.info(f"Detected session type: {SESSION_TYPE}")


class AudioRecorderDaemon:
    """Non-blocking audio recorder using parecord."""

    # Shared notification ID for replacing notifications
    NOTIFICATION_ID = 999999

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.process: Optional[subprocess.Popen] = None
        self.output_file: Optional[str] = None

    def start_recording(self) -> None:
        """Start recording with parecord."""
        if self.is_recording:
            logger.warning("Already recording, ignoring start request")
            return

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.output_file = temp_file.name
        temp_file.close()
        logger.info(f"Created temp audio file: {self.output_file}")

        # Start parecord
        logger.info(f"Starting parecord (rate={self.sample_rate})")
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

        self.is_recording = True
        logger.info(f"Recording started, parecord PID: {self.process.pid}")
        self._play_sound('/usr/share/sounds/freedesktop/stereo/message-new-instant.oga')
        self._notify("🎤 Recording...")

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
            self.process.wait(timeout=2)
            logger.info("parecord terminated")

        # Check file size
        if self.output_file and os.path.exists(self.output_file):
            file_size = os.path.getsize(self.output_file)
            logger.info(f"Recorded audio file: {self.output_file} ({file_size} bytes)")
        else:
            logger.error(f"Audio file not found: {self.output_file}")

        # Play stop sound
        self._play_sound('/usr/share/sounds/freedesktop/stereo/complete.oga')
        # Removed "Recording stopped" notification - goes straight to transcribing
        return self.output_file

    def _notify(self, message: str) -> None:
        """Send desktop notification that replaces previous ones."""
        subprocess.run(
            ['notify-send', '-t', '1500', '-r', str(self.NOTIFICATION_ID), 'Voice Transcription', message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def _play_sound(self, sound_file: str) -> None:
        """Play a sound file using paplay."""
        if os.path.exists(sound_file):
            subprocess.Popen(
                ['paplay', sound_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            logger.warning(f"Sound file not found: {sound_file}")

    def cleanup(self) -> None:
        """Cleanup audio resources."""
        if self.process and self.process.poll() is None:
            self.process.terminate()


class TranscriptionPipeline:
    """Handles transcription and polishing pipeline."""

    # Shared notification ID for replacing notifications
    NOTIFICATION_ID = 999999

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
                    speech_pad_ms=400,  # Pad speech segments by 400ms on each side
                )
            )
            raw_text = " ".join([seg.text.strip() for seg in segments])
            logger.info(f"Transcription complete. Raw text: '{raw_text}'")
        except ValueError as e:
            # Empty audio or no detectable language
            logger.warning(f"ValueError during transcription: {e}")
            self._notify("⚠️  No speech detected (empty audio)")
            return ""
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            self._notify(f"❌ Transcription error: {str(e)[:50]}")
            return ""

        if not raw_text.strip():
            logger.warning("Transcription returned empty text")
            self._notify("⚠️  No speech detected")
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
            self._notify(f"✅ Pasted: {final_text[:50]}...")
        else:
            logger.info("Auto-paste disabled")
            self._notify(f"✅ Copied: {final_text[:50]}...")

        logger.info("Processing complete")
        return final_text

    def _notify(self, message: str) -> None:
        """Send desktop notification that replaces previous ones."""
        subprocess.run(
            ['notify-send', '-t', '1500', '-r', str(self.NOTIFICATION_ID), 'Voice Transcription', message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


class TranscriptionDaemon:
    """Background daemon handling recording and transcription."""

    def __init__(self):
        self.recorder = AudioRecorderDaemon()
        self.pipeline = TranscriptionPipeline()
        self.current_audio_file: Optional[str] = None
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
                self.current_audio_file = audio_path
                # Process in background thread
                threading.Thread(target=self._process_audio, daemon=True).start()

    def _process_audio(self) -> None:
        """Process recorded audio in background."""
        if not self.current_audio_file:
            return

        try:
            result = self.pipeline.process(self.current_audio_file)
            if result:
                print(result, flush=True)
        finally:
            # Cleanup temp file
            if os.path.exists(self.current_audio_file):
                os.unlink(self.current_audio_file)
            self.current_audio_file = None

    def start_recording(self) -> None:
        """Start recording."""
        self.recorder.start_recording()

    def stop_and_process(self) -> None:
        """Stop recording and process."""
        logger.info("stop_and_process called")
        self.last_activity_time = time.time()  # Update activity time
        audio_path = self.recorder.stop_recording()
        if audio_path:
            self.current_audio_file = audio_path
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
    daemon_running_file = "/tmp/whisper-hotkey-daemon-running"

    # Check if persistent daemon is already running
    if os.path.exists(daemon_running_file):
        try:
            with open(daemon_running_file, 'r') as f:
                daemon_pid = int(f.read().strip())

            # Check if it's actually alive
            os.kill(daemon_pid, 0)  # Raises error if process doesn't exist

            # Daemon is alive, check if we're starting or stopping recording
            if os.path.exists(state_file):
                # Currently recording, send stop signal
                logger.info(f"Sending stop signal to daemon PID {daemon_pid}")
                os.kill(daemon_pid, signal.SIGUSR1)
                os.unlink(state_file)
            else:
                # Start new recording
                logger.info(f"Sending start signal to daemon PID {daemon_pid}")
                Path(state_file).touch()
                os.kill(daemon_pid, signal.SIGUSR2)
            sys.exit(0)

        except (FileNotFoundError, ProcessLookupError, ValueError, OSError):
            # Daemon died, clean up and start new one
            logger.warning("Daemon PID file exists but process is dead, starting new daemon")
            if os.path.exists(daemon_running_file):
                os.unlink(daemon_running_file)
            if os.path.exists(state_file):
                os.unlink(state_file)

    # Start persistent daemon
    logger.info("Starting persistent daemon (will stay alive for 10 minutes after last use)")
    daemon = TranscriptionDaemon()

    # Write our PID to daemon file
    with open(daemon_running_file, 'w') as f:
        f.write(str(os.getpid()))
    logger.info(f"Wrote daemon PID to {daemon_running_file}")

    # Signal handlers
    start_recording_event = threading.Event()
    stop_recording_event = threading.Event()
    shutdown_event = threading.Event()

    def handle_start(signum, frame):
        logger.info("Received start recording signal")
        start_recording_event.set()

    def handle_stop(signum, frame):
        logger.info("Received stop recording signal")
        stop_recording_event.set()

    def handle_shutdown(signum, frame):
        logger.info("Received shutdown signal")
        shutdown_event.set()

    signal.signal(signal.SIGUSR2, handle_start)  # Start recording
    signal.signal(signal.SIGUSR1, handle_stop)   # Stop recording
    signal.signal(signal.SIGTERM, handle_shutdown)  # Shutdown daemon
    signal.signal(signal.SIGINT, handle_shutdown)   # Ctrl+C
    logger.info("Signal handlers registered")

    try:
        logger.info("Daemon loop started, model loaded and ready")

        while not shutdown_event.is_set():
            # Check for start recording signal
            if start_recording_event.is_set():
                start_recording_event.clear()
                daemon.start_recording()
                daemon.last_activity_time = time.time()

            # Check for stop recording signal
            if stop_recording_event.is_set():
                stop_recording_event.clear()
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
        if os.path.exists(state_file):
            os.unlink(state_file)
            logger.info("Removed state file")
        if os.path.exists(daemon_running_file):
            os.unlink(daemon_running_file)
            logger.info("Removed daemon PID file")
        logger.info("=== Daemon exiting ===")


if __name__ == "__main__":
    main()
