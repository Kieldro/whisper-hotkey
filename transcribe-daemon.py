#!/usr/bin/env python3
"""
Background daemon for push-to-talk voice transcription.
More Superwhisper-like UX: hold key to record, release to transcribe.
"""

import os
import sys
import tempfile
import subprocess
import ctypes
import threading
import signal
import logging
import fcntl
import shutil
from pathlib import Path
from typing import Optional
import platform
IS_MACOS = platform.system() == "Darwin"

from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Setup logging - use user-private log file
RUNTIME_DIR = os.getenv("TMPDIR", "/tmp").rstrip('/') if IS_MACOS else os.getenv("XDG_RUNTIME_DIR", "/tmp")
LOG_FILE = os.path.join(RUNTIME_DIR, "whisper-hotkey.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
ENGINE = os.getenv("ENGINE", "parakeet")  # whisper or parakeet
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
ENABLE_POLISHING = os.getenv("ENABLE_POLISHING", "false").lower() == "true"
AUTO_PASTE = os.getenv("AUTO_PASTE", "true").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
try:
    IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "600"))
except ValueError:
    print(f"Error: IDLE_TIMEOUT='{os.getenv('IDLE_TIMEOUT')}' is not a valid integer", file=sys.stderr)
    sys.exit(1)
SAMPLE_RATE = 16000
try:
    TRAILING_SPEECH_DELAY = float(os.getenv("TRAILING_SPEECH_DELAY", "1.2"))
except ValueError:
    print(f"Error: TRAILING_SPEECH_DELAY='{os.getenv('TRAILING_SPEECH_DELAY')}' is not a valid number", file=sys.stderr)
    sys.exit(1)

# Sound file paths (optional - will skip if not found)
if IS_MACOS:
    SOUND_START = os.getenv("SOUND_START", "/System/Library/Sounds/Ping.aiff")
    SOUND_LOADING = os.getenv("SOUND_LOADING", "/System/Library/Sounds/Sosumi.aiff")
    SOUND_PASTE = os.getenv("SOUND_PASTE", "/System/Library/Sounds/Glass.aiff")
else:
    SOUND_START = os.getenv("SOUND_START", "/usr/share/sounds/freedesktop/stereo/message-new-instant.oga")
    SOUND_LOADING = os.getenv("SOUND_LOADING", "/usr/share/sounds/ubuntu/ringtones/Wind chime.ogg")
    SOUND_PASTE = os.getenv("SOUND_PASTE", "/usr/share/sounds/ubuntu/notifications/Positive.ogg")

# Shared notification ID for replacing notifications
NOTIFICATION_ID = 999999

# Valid config values for validation
VALID_WHISPER_MODELS = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large",
    "distil-large-v2", "distil-large-v3", "distil-medium.en", "distil-small.en",
}
VALID_DEVICES = {"cpu", "cuda", "auto", "mps"}
VALID_COMPUTE_TYPES = {
    "int8", "int8_float16", "int8_float32", "int16",
    "float16", "float32", "bfloat16", "auto",
}


VALID_ENGINES = {"whisper", "parakeet"}


def validate_config() -> None:
    """Validate all config values at startup. Exit with clear error on invalid values."""
    errors = []

    if ENGINE not in VALID_ENGINES:
        errors.append(f"ENGINE='{ENGINE}' invalid. Valid: {sorted(VALID_ENGINES)}")

    if ENGINE == "whisper":
        if WHISPER_MODEL not in VALID_WHISPER_MODELS:
            errors.append(f"WHISPER_MODEL='{WHISPER_MODEL}' not recognized. Valid: {sorted(VALID_WHISPER_MODELS)}")

        if DEVICE not in VALID_DEVICES:
            errors.append(f"DEVICE='{DEVICE}' invalid. Valid: {sorted(VALID_DEVICES)}")

        if COMPUTE_TYPE not in VALID_COMPUTE_TYPES:
            errors.append(f"COMPUTE_TYPE='{COMPUTE_TYPE}' invalid. Valid: {sorted(VALID_COMPUTE_TYPES)}")

    if IDLE_TIMEOUT < 0:
        errors.append(f"IDLE_TIMEOUT={IDLE_TIMEOUT} must be >= 0")

    if errors:
        for err in errors:
            logger.error(f"Config error: {err}")
            print(f"❌ Config error: {err}", file=sys.stderr)
        sys.exit(1)

    logger.info("Config validation passed")

def send_notification(message: str) -> None:
    """Send desktop notification."""
    try:
        if IS_MACOS:
            subprocess.run(
                ['osascript', '-e', f'display notification "{message}" with title "Voice Transcription"'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            subprocess.run(
                ['notify-send', '-t', '1500', '-r', str(NOTIFICATION_ID), 'Voice Transcription', message],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
    except FileNotFoundError:
        pass  # notify tool not installed


def play_sound(sound_file: str) -> None:
    """Play a sound file (non-blocking, fails silently)."""
    if not sound_file or not os.path.exists(sound_file):
        return
    try:
        cmd = ['afplay', sound_file] if IS_MACOS else ['paplay', sound_file]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass


def is_shift_held() -> bool:
    """Check if Shift is currently pressed at paste time.

    User holds Shift while transcription runs; checked right before paste.
    macOS: Quartz CGEventSourceKeyState. Linux: X11 XQueryKeymap.
    """
    if IS_MACOS:
        try:
            from Quartz import CGEventSourceKeyState, kCGEventSourceStateCombinedSessionState
            # kVK_Shift = 0x38, kVK_RightShift = 0x3C
            left = CGEventSourceKeyState(kCGEventSourceStateCombinedSessionState, 0x38)
            right = CGEventSourceKeyState(kCGEventSourceStateCombinedSessionState, 0x3C)
            return bool(left or right)
        except Exception:
            return False
    else:
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so.6')
            display = x11.XOpenDisplay(None)
            if not display:
                return False
            keymap = (ctypes.c_char * 32)()
            x11.XQueryKeymap(display, keymap)
            x11.XCloseDisplay(display)
            # Shift_L = keycode 50, Shift_R = keycode 62
            sl = ord(keymap[50 // 8]) & (1 << (50 % 8))
            sr = ord(keymap[62 // 8]) & (1 << (62 % 8))
            return bool(sl or sr)
        except Exception:
            return False


# Detect session type for clipboard/typing
def detect_session_type():
    """Detect the display server session type."""
    if IS_MACOS:
        return "macos"

    session_type = os.getenv("XDG_SESSION_TYPE", "").lower()
    wayland_display = os.getenv("WAYLAND_DISPLAY")
    x11_display = os.getenv("DISPLAY")

    if session_type == "wayland" or wayland_display:
        detected_session = "wayland"
    elif session_type == "x11" or x11_display:
        detected_session = "x11"
    else:
        logger.warning("Could not detect session type from environment, defaulting to X11")
        detected_session = "x11"

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


class AudioRecorderDaemon:
    """Non-blocking audio recorder. Uses sounddevice on macOS, parecord on Linux."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.process: Optional[subprocess.Popen] = None
        self.output_file: Optional[str] = None
        self._sd_stream = None
        self._audio_queue = None

    def start_recording(self) -> bool:
        """Start recording. Returns True on success."""
        if self.is_recording:
            logger.warning("Already recording, ignoring start request")
            return False
        if IS_MACOS:
            return self._start_recording_macos()
        return self._start_recording_linux()

    def _start_recording_macos(self) -> bool:
        """Record audio using sounddevice (macOS CoreAudio)."""
        import queue as _queue
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed. Run: pip install sounddevice soundfile numpy")
            send_notification("❌ sounddevice not installed")
            return False

        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.output_file = temp_file.name
        temp_file.close()
        logger.info(f"Created temp audio file: {self.output_file}")

        self._audio_queue = _queue.Queue()

        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio status: {status}")
            self._audio_queue.put(indata.copy())

        try:
            self._sd_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                callback=callback
            )
            self._sd_stream.start()
        except Exception as e:
            logger.error(f"Failed to start sounddevice: {e}")
            send_notification(f"❌ Recording failed: {str(e)[:30]}")
            self._cleanup_temp_files()
            return False

        self.is_recording = True
        logger.info("Recording started (sounddevice/CoreAudio)")
        play_sound(SOUND_START)
        send_notification("🎤 Recording...")
        return True

    def _start_recording_linux(self) -> bool:
        """Record audio using parecord (Linux/PulseAudio)."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.output_file = temp_file.name
        temp_file.close()
        logger.info(f"Created temp audio file: {self.output_file}")

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
        if IS_MACOS:
            return self._stop_recording_macos()
        return self._stop_recording_linux()

    def _stop_recording_macos(self) -> Optional[str]:
        """Stop sounddevice recording, close stream, write WAV file."""
        logger.info("Stopping recording (sounddevice)...")
        self.is_recording = False

        time.sleep(TRAILING_SPEECH_DELAY)

        if self._sd_stream:
            self._sd_stream.stop()
            self._sd_stream.close()
            self._sd_stream = None

        # Drain queue and write WAV
        frames = []
        while self._audio_queue and not self._audio_queue.empty():
            frames.append(self._audio_queue.get())

        if not frames:
            logger.error("No audio recorded (empty queue)")
            return None

        try:
            import numpy as np
            import soundfile as sf
            audio_data = np.concatenate(frames, axis=0)
            sf.write(self.output_file, audio_data, self.sample_rate, subtype='PCM_16')
        except ImportError as e:
            logger.error(f"Missing dependency: {e}. Run: pip install sounddevice soundfile numpy")
            return None
        except Exception as e:
            logger.error(f"Failed to write audio file: {e}")
            return None

        file_size = os.path.getsize(self.output_file)
        logger.info(f"Final audio file: {self.output_file} ({file_size} bytes)")
        return self.output_file

    def _stop_recording_linux(self) -> Optional[str]:
        """Stop parecord recording."""
        logger.info("Stopping recording (parecord)...")
        self.is_recording = False

        if self.process:
            time.sleep(TRAILING_SPEECH_DELAY)
            logger.info(f"Stopping parecord process {self.process.pid}")
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                logger.warning("parecord didn't stop on SIGINT, sending SIGTERM...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.warning("parecord didn't terminate, killing...")
                    self.process.kill()
                    self.process.wait()
            logger.info("parecord stopped")
            self.process = None

        if self.output_file and os.path.exists(self.output_file):
            file_size = os.path.getsize(self.output_file)
            logger.info(f"Final audio file: {self.output_file} ({file_size} bytes)")
        else:
            logger.error(f"Audio file not found: {self.output_file}")

        return self.output_file

    def cleanup(self) -> None:
        """Cleanup audio resources."""
        if self._sd_stream:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
        if self.process and self.process.poll() is None:
            self.process.terminate()


class TranscriptionPipeline:
    """Handles transcription and polishing pipeline."""

    def __init__(self):
        logger.info(f"Initializing TranscriptionPipeline (engine={ENGINE}, model={WHISPER_MODEL}, device={DEVICE}, polishing={ENABLE_POLISHING})")
        self.engine = ENGINE

        if self.engine == "parakeet":
            logger.info("Loading Parakeet TDT model via onnx-asr...")
            import onnx_asr
            self.model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
            logger.info("Parakeet model loaded successfully")
        else:
            from faster_whisper import WhisperModel
            logger.info("Loading Whisper model...")
            try:
                self.whisper = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
            except Exception as e:
                if DEVICE != "cpu":
                    logger.warning(f"Failed to load model on {DEVICE}: {e}")
                    logger.warning("Falling back to CPU with int8 compute type")
                    send_notification(f"⚠️ {DEVICE} failed, falling back to CPU")
                    self.whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
                else:
                    raise
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

        send_notification("Transcribing...")
        logger.info(f"Starting transcription (engine={self.engine})...")

        try:
            if self.engine == "parakeet":
                raw_text = self.model.recognize(audio_path)
                logger.info(f"Transcription complete ({len(raw_text)} chars)")
            else:
                segments, _ = self.whisper.transcribe(
                    audio_path,
                    beam_size=5,
                    vad_filter=False,  # Disabled - process entire audio without cutting
                )
                raw_text = " ".join([seg.text.strip() for seg in segments])
                logger.info(f"Transcription complete ({len(raw_text)} chars)")
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
                content = response.choices[0].message.content
                final_text = content.strip() if content else raw_text
                logger.info(f"Polished text ({len(final_text)} chars)")
            except Exception as e:
                logger.error(f"GPT polishing error: {e}", exc_info=True)
                final_text = raw_text
        else:
            final_text = raw_text
            logger.info("Skipping polishing (disabled)")

        # Copy to clipboard
        logger.info("Copying to clipboard...")
        try:
            if IS_MACOS:
                subprocess.run(
                    ['pbcopy'],
                    input=final_text.encode(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            elif SESSION_TYPE == "wayland":
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
        except FileNotFoundError as e:
            logger.error(f"Clipboard tool not found: {e}")
            send_notification("Clipboard tool missing")

        # Auto-paste if enabled
        if AUTO_PASTE:
            # Check shift BEFORE paste — on macOS, holding Shift during
            # osascript Cmd+V sends Cmd+Shift+V (Paste Special), which is wrong.
            # So: detect shift → wait for release → paste cleanly → press Enter.
            submit_after_paste = is_shift_held()
            if submit_after_paste and IS_MACOS:
                logger.info("Shift held — waiting for release before paste...")
                deadline = time.time() + 5.0
                while is_shift_held() and time.time() < deadline:
                    time.sleep(0.05)
                time.sleep(0.1)  # small buffer after release

            time.sleep(0.2)  # Brief delay
            try:
                if IS_MACOS:
                    logger.info("Auto-pasting with osascript Cmd+V...")
                    result = subprocess.run(
                        ['osascript', '-e',
                         'tell application "System Events" to keystroke "v" using {command down}'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    logger.info(f"osascript exit code: {result.returncode}")
                elif SESSION_TYPE == "wayland":
                    logger.info("Auto-pasting with ydotool Ctrl+V...")
                    result = subprocess.run(
                        ['ydotool', 'key', '29:1', '47:1', '47:0', '29:0'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    logger.info(f"ydotool key exit code: {result.returncode}")
                else:  # X11
                    logger.info("Auto-pasting with xdotool type...")
                    result = subprocess.run(
                        ['xdotool', 'type', '--clearmodifiers', '--delay', '0', final_text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    logger.info(f"xdotool type exit code: {result.returncode}")
                # Submit with Enter if Shift was held
                if submit_after_paste or (not IS_MACOS and is_shift_held()):
                    logger.info("Shift-to-submit: pressing Enter")
                    time.sleep(0.1)
                    if IS_MACOS:
                        subprocess.run(
                            ['osascript', '-e',
                             'tell application "System Events" to key code 36'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    elif SESSION_TYPE == "wayland":
                        subprocess.run(['ydotool', 'key', '28:1', '28:0'],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        subprocess.run(['xdotool', 'key', '--clearmodifiers', 'Return'],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError as e:
                logger.error(f"Paste tool not found: {e}")
                send_notification("Paste tool missing (text copied to clipboard)")

            play_sound(SOUND_PASTE)
            send_notification(f"Pasted: {final_text[:50]}...")
        else:
            logger.info("Auto-paste disabled")
            play_sound(SOUND_PASTE)
            send_notification(f"Copied: {final_text[:50]}...")

        logger.info("Processing complete")
        return final_text


class TranscriptionDaemon:
    """Background daemon handling recording and transcription."""

    def __init__(self):
        self.recorder = AudioRecorderDaemon()
        self.pipeline = TranscriptionPipeline()
        self._audio_lock = threading.Lock()
        self._current_audio_file: Optional[str] = None
        self.last_activity_time = time.time()
        self.idle_timeout = IDLE_TIMEOUT

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


def acquire_daemon_lock(lock_file: str) -> Optional[int]:
    """
    Try to acquire exclusive lock on daemon file.
    Returns file descriptor on success, None if another daemon holds the lock.
    """
    try:
        fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o600)
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
    logger.info(f"Config: ENGINE={ENGINE}, ENABLE_POLISHING={ENABLE_POLISHING}, AUTO_PASTE={AUTO_PASTE}, MODEL={WHISPER_MODEL}")

    validate_config()

    if ENABLE_POLISHING and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set but polishing is enabled")
        print("❌ Error: OPENAI_API_KEY not set (required when ENABLE_POLISHING=true)", file=sys.stderr)
        sys.exit(1)

    state_file = os.path.join(RUNTIME_DIR, "whisper-hotkey-state")
    daemon_lock_file = os.path.join(RUNTIME_DIR, "whisper-hotkey-daemon.lock")

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

    # Start persistent daemon (slow: loads model)
    # Create state file immediately so extra presses during load send STOP (not START)
    Path(state_file).touch()

    # Play calming sound while model loads
    loading_sound_proc = None
    if SOUND_LOADING and os.path.exists(SOUND_LOADING):
        try:
            cmd = ['afplay', SOUND_LOADING] if IS_MACOS else ['paplay', SOUND_LOADING]
            loading_sound_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            pass

    logger.info(f"Loading {ENGINE} model... (this may take a few seconds on first start)")
    daemon = TranscriptionDaemon()
    daemon.last_activity_time = time.time()

    # Stop loading sound
    if loading_sound_proc:
        loading_sound_proc.terminate()
        loading_sound_proc = None

    # Clear any signals the user sent while the model was loading (impatient extra presses)
    start_recording_event.clear()
    stop_recording_event.clear()
    # Re-create state file in case a press during loading deleted it
    Path(state_file).touch()

    # Auto-start recording on first launch (the user pressed the hotkey to get here)
    logger.info("Daemon ready, auto-starting first recording...")
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
