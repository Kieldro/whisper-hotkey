#!/usr/bin/env python3
"""
Background daemon for push-to-talk voice transcription.
More Superwhisper-like UX: hold key to record, release to transcribe.
"""

import os
import sys
import re
import json
import wave
import struct
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
    handlers=[logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger(__name__)
# Route uncaught exceptions through the logger so they land in the same file
# (previously they went to stderr and were swallowed by `2>/dev/null`).
sys.excepthook = lambda t, v, tb: logger.error("Uncaught exception", exc_info=(t, v, tb))

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
MAX_RECORDING_SECONDS = int(os.getenv("MAX_RECORDING_SECONDS", "120"))
try:
    TRAILING_SPEECH_DELAY = float(os.getenv("TRAILING_SPEECH_DELAY", "0.5"))
except ValueError:
    print(f"Error: TRAILING_SPEECH_DELAY='{os.getenv('TRAILING_SPEECH_DELAY')}' is not a valid number", file=sys.stderr)
    sys.exit(1)

# Feature configs
ENABLE_VAD = os.getenv("ENABLE_VAD", "true").lower() == "true"
try:
    VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
except ValueError:
    print(f"Error: VAD_THRESHOLD='{os.getenv('VAD_THRESHOLD')}' is not a valid number", file=sys.stderr)
    sys.exit(1)
ENABLE_AUDIO_NORMALIZATION = os.getenv("ENABLE_AUDIO_NORMALIZATION", "true").lower() == "true"
ENABLE_SPOKEN_PUNCTUATION = os.getenv("ENABLE_SPOKEN_PUNCTUATION", "false").lower() == "true"
REPLACEMENTS_FILE = os.getenv("REPLACEMENTS_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "replacements.json"))
PASTE_METHOD = os.getenv("PASTE_METHOD", "auto")
ENABLE_NOTIFICATIONS = os.getenv("ENABLE_NOTIFICATIONS", "true").lower() == "true"
RESTORE_CLIPBOARD = os.getenv("RESTORE_CLIPBOARD", "false").lower() == "true"
ENABLE_OVERLAY = os.getenv("ENABLE_OVERLAY", "true").lower() == "true"

# Sound file paths (optional - will skip if not found)
if IS_MACOS:
    SOUND_START = os.getenv("SOUND_START", "/System/Library/Sounds/Ping.aiff")
    _chime = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds", "loading-chime.wav")
    SOUND_LOADING = os.getenv("SOUND_LOADING", _chime)
    SOUND_PASTE = os.getenv("SOUND_PASTE", "/System/Library/Sounds/Glass.aiff")
else:
    SOUND_START = os.getenv("SOUND_START", "/usr/share/sounds/freedesktop/stereo/message-new-instant.oga")
    SOUND_LOADING = os.getenv("SOUND_LOADING", "/usr/share/sounds/ubuntu/ringtones/Wind chime.ogg")
    SOUND_PASTE = os.getenv("SOUND_PASTE", "/usr/share/sounds/ubuntu/notifications/Positive.ogg")

# Shared notification ID for replacing notifications
NOTIFICATION_ID = 999999

# Status file for waybar/polybar integration
STATUS_FILE = os.path.join(RUNTIME_DIR, "whisper-hotkey-status.json")

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


VALID_ENGINES = {"whisper", "parakeet", "apple-streaming"}


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
            print(f"Config error: {err}", file=sys.stderr)
        sys.exit(1)

    logger.info("Config validation passed")

def send_notification(message: str) -> None:
    """Send desktop notification that replaces previous ones."""
    if not ENABLE_NOTIFICATIONS:
        return
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


# ---------------------------------------------------------------------------
# Feature 7: Waybar/Polybar status integration
# ---------------------------------------------------------------------------

def update_status(state: str, **kwargs) -> None:
    """Write current daemon state to JSON file for waybar/polybar."""
    data = {"state": state, "timestamp": time.time()}
    data.update(kwargs)
    try:
        tmp_path = STATUS_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, STATUS_FILE)
    except OSError as e:
        logger.debug(f"Failed to write status file: {e}")


def cleanup_status() -> None:
    """Remove status file on daemon shutdown."""
    try:
        os.remove(STATUS_FILE)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.debug(f"Failed to remove status file: {e}")


def is_shift_held() -> bool:
    """Check if Shift is currently held.

    macOS: CGEventSourceFlagsState reads physical modifier key state directly.
    Linux: reads X11 keymap directly.
    """
    if IS_MACOS:
        try:
            from Quartz import CGEventSourceFlagsState, kCGEventFlagMaskShift
            # kCGEventSourceStateHIDSystemState (1) = physical key state
            flags = CGEventSourceFlagsState(1)
            return bool(flags & kCGEventFlagMaskShift)
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


# ---------------------------------------------------------------------------
# Feature 1: Silero VAD trimming
# ---------------------------------------------------------------------------

_vad_model = None
_vad_utils = None


def _get_vad_model():
    """Lazy-load Silero VAD model once, reuse on subsequent calls."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _vad_model = model
        _vad_utils = utils
        logger.info("Silero VAD model loaded")
    return _vad_model, _vad_utils


def vad_trim_audio(audio_path: str) -> Optional[str]:
    """Trim leading/trailing silence using Silero VAD. Returns None if no speech."""
    if not ENABLE_VAD:
        return audio_path

    try:
        import torch
    except ImportError:
        logger.warning("torch not installed, skipping VAD")
        return audio_path

    try:
        model, utils = _get_vad_model()
        get_speech_timestamps = utils[0]

        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        samples = struct.unpack(f"<{n_frames * n_channels}h", raw)
        audio_tensor = torch.FloatTensor(samples) / 32768.0

        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            threshold=VAD_THRESHOLD,
            sampling_rate=SAMPLE_RATE,
        )

        if not speech_timestamps:
            logger.info("VAD: no speech detected, skipping transcription")
            send_notification("No speech detected")
            update_status("idle")
            return None

        # Trim with 0.3s padding to preserve trailing consonants/words
        pad = int(0.3 * SAMPLE_RATE)
        start = max(0, speech_timestamps[0]["start"] - pad)
        end = min(len(samples), speech_timestamps[-1]["end"] + pad)

        trimmed = struct.pack(f"<{end - start}h", *samples[start:end])
        with wave.open(audio_path, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(trimmed)

        duration_before = len(samples) / SAMPLE_RATE
        duration_after = (end - start) / SAMPLE_RATE
        logger.info(
            "VAD: trimmed %.2fs -> %.2fs (%.0f%% removed)",
            duration_before,
            duration_after,
            (1 - duration_after / duration_before) * 100 if duration_before > 0 else 0,
        )
        return audio_path

    except Exception as e:
        logger.error("VAD trimming failed, using original audio: %s", e)
        return audio_path


# ---------------------------------------------------------------------------
# Feature 5: Audio normalization
# ---------------------------------------------------------------------------

def normalize_audio(audio_path: str) -> str:
    """Normalize audio to 80% of max amplitude. Skips if near target."""
    if not ENABLE_AUDIO_NORMALIZATION:
        return audio_path

    try:
        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        n_samples = n_frames * n_channels
        target = int(32767 * 0.8)

        try:
            import numpy as np

            samples = np.frombuffer(raw, dtype=np.int16).copy()
            peak = int(np.max(np.abs(samples)))

            if peak == 0:
                logger.info("Normalization: silent audio, skipping")
                return audio_path

            scale = target / peak

            if 0.9 <= scale <= 1.1:
                logger.debug("Normalization: already near target (scale=%.3f), skipping", scale)
                return audio_path

            normalized = np.clip(samples * scale, -32768, 32767).astype(np.int16)
            raw_out = normalized.tobytes()

        except ImportError:
            samples = list(struct.unpack(f"<{n_samples}h", raw))
            peak = max(abs(s) for s in samples)

            if peak == 0:
                logger.info("Normalization: silent audio, skipping")
                return audio_path

            scale = target / peak

            if 0.9 <= scale <= 1.1:
                logger.debug("Normalization: already near target (scale=%.3f), skipping", scale)
                return audio_path

            normalized = [max(-32768, min(32767, int(s * scale))) for s in samples]
            raw_out = struct.pack(f"<{n_samples}h", *normalized)

        with wave.open(audio_path, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(raw_out)

        logger.info("Normalization: scale=%.3f, peak %d -> %d", scale, peak, target)
        return audio_path

    except Exception as e:
        logger.error("Audio normalization failed, using original: %s", e)
        return audio_path


# ---------------------------------------------------------------------------
# Feature 3: Spoken punctuation conversion
# ---------------------------------------------------------------------------

# Multi-word phrases must come before single-word components
SPOKEN_PUNCTUATION = [
    ("new paragraph", "\n\n"),
    ("new line", "\n"),
    ("exclamation mark", "!"),
    ("exclamation point", "!"),
    ("question mark", "?"),
    ("open paren", "("),
    ("close paren", ")"),
    ("semicolon", ";"),
    ("colon", ":"),
    ("comma", ","),
    ("period", "."),
    ("en dash", "\u2013"),
    ("em dash", "\u2014"),
    ("dash", "\u2014"),
    ("hyphen", "-"),
    ("open quote", '"'),
    ("close quote", '"'),
    ("end quote", '"'),
    ("quote", '"'),
]

_SPOKEN_PUNCT_COMPILED = [
    (re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE), replacement)
    for phrase, replacement in SPOKEN_PUNCTUATION
]


def apply_spoken_punctuation(text: str) -> str:
    """Convert spoken punctuation words to their symbol equivalents."""
    if not ENABLE_SPOKEN_PUNCTUATION:
        return text

    for pattern, replacement in _SPOKEN_PUNCT_COMPILED:
        text = pattern.sub(replacement, text)

    # Clean up spacing around punctuation
    text = re.sub(r'\s+([.,!?;:)])', r'\1', text)
    text = re.sub(r'([(])\s+', r'\1', text)
    text = re.sub(r'(^|\s)"(\s+)', r'\1"', text)
    text = re.sub(r'\s+"(\s|$|[.,!?;:)])', r'"\1', text)
    text = re.sub(r' *\n *', '\n', text)

    return text


# ---------------------------------------------------------------------------
# Feature 6: Word replacement dictionary
# ---------------------------------------------------------------------------

def load_replacements() -> list:
    """Load word replacement dictionary from JSON file."""
    try:
        with open(REPLACEMENTS_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.debug("Replacements file not found: %s", REPLACEMENTS_FILE)
        return []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load replacements file %s: %s", REPLACEMENTS_FILE, e)
        return []

    if not isinstance(data, dict):
        logger.warning("Replacements file must contain a JSON object, got %s", type(data).__name__)
        return []

    result = []
    for word, replacement in data.items():
        try:
            compiled = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
            result.append((compiled, replacement))
        except re.error as e:
            logger.warning("Invalid replacement pattern %r: %s", word, e)

    if result:
        logger.info(f"Loaded {len(result)} word replacements from {REPLACEMENTS_FILE}")
    return result


_WORD_REPLACEMENTS = load_replacements()


def apply_word_replacements(text: str) -> str:
    """Apply user-defined word replacements from replacements.json."""
    for pattern, replacement in _WORD_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Feature 2: Clipboard save/restore
# ---------------------------------------------------------------------------

def save_clipboard() -> Optional[bytes]:
    """Save current clipboard contents before overwriting."""
    try:
        if IS_MACOS:
            cmd = ["pbpaste"]
        elif SESSION_TYPE == "wayland":
            cmd = ["wl-paste", "--no-newline"]
        else:
            cmd = ["xclip", "-selection", "clipboard", "-o"]

        result = subprocess.run(cmd, capture_output=True, timeout=2)
        if result.returncode != 0 or not result.stdout:
            return None
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"Could not save clipboard: {e}")
        return None


def restore_clipboard(content: Optional[bytes]) -> None:
    """Restore previously saved clipboard contents."""
    if content is None:
        return

    try:
        if IS_MACOS:
            cmd = ["pbcopy"]
        elif SESSION_TYPE == "wayland":
            cmd = ["wl-copy"]
        else:
            cmd = ["xclip", "-selection", "clipboard"]

        subprocess.run(cmd, input=content, timeout=2, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"Could not restore clipboard: {e}")


# ---------------------------------------------------------------------------
# Feature 4: Paste fallback chain
# ---------------------------------------------------------------------------

def _build_paste_chain() -> list:
    """Build ordered list of paste methods based on session type and available tools."""
    chain = []

    if IS_MACOS:
        chain.append("macos-cgevent")
    elif SESSION_TYPE == "wayland":
        if shutil.which("wtype"):
            chain.append("wtype")
        if shutil.which("ydotool"):
            chain.append("ydotool-clipboard")
    else:
        if shutil.which("xdotool"):
            chain.append("xdotool-type")
            chain.append("xdotool-clipboard")

    return chain


PASTE_CHAIN = _build_paste_chain()
logger.info(f"Paste chain: {PASTE_CHAIN}")


def _copy_to_clipboard(text: str) -> None:
    """Copy text to clipboard (helper for clipboard-based paste methods)."""
    try:
        if IS_MACOS:
            subprocess.run(["pbcopy"], input=text.encode(), timeout=2, check=False)
        elif SESSION_TYPE == "wayland":
            subprocess.run(["wl-copy"], input=text.encode(), timeout=2, check=False)
        else:
            subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), timeout=2, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"Clipboard copy failed: {e}")


def _execute_paste(method: str, text: str) -> bool:
    """Execute a single paste method. Returns True on success."""
    if method == "macos-cgevent":
        _copy_to_clipboard(text)
        try:
            from Quartz import (CGEventCreateKeyboardEvent, CGEventSetFlags,
                                CGEventPost, kCGEventFlagMaskCommand, kCGHIDEventTap)
            # Cmd+V: keycode 9 = V. CGEvent ignores physical modifiers (e.g. Shift)
            ev = CGEventCreateKeyboardEvent(None, 9, True)
            CGEventSetFlags(ev, kCGEventFlagMaskCommand)
            CGEventPost(kCGHIDEventTap, ev)
            time.sleep(0.05)
            ev = CGEventCreateKeyboardEvent(None, 9, False)
            CGEventSetFlags(ev, kCGEventFlagMaskCommand)
            CGEventPost(kCGHIDEventTap, ev)
            return True
        except Exception as e:
            logger.warning(f"CGEvent Cmd+V failed ({e}), trying osascript fallback")
            result = subprocess.run(
                ['osascript', '-e',
                 'tell application "System Events" to keystroke "v" using {command down}'],
                timeout=5, capture_output=True, check=False)
            if result.returncode == 0:
                logger.info("Pasted via osascript fallback (install pyobjc-framework-Quartz for native path)")
            return result.returncode == 0

    if method == "wtype":
        result = subprocess.run(
            ["wtype", "--", text],
            timeout=5, capture_output=True, check=False,
        )
        return result.returncode == 0

    elif method == "ydotool-clipboard":
        _copy_to_clipboard(text)
        result = subprocess.run(
            ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
            timeout=5, capture_output=True, check=False,
        )
        return result.returncode == 0

    elif method == "xdotool-type":
        result = subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "0", text],
            timeout=5, capture_output=True, check=False,
        )
        return result.returncode == 0

    elif method == "xdotool-clipboard":
        _copy_to_clipboard(text)
        result = subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
            timeout=5, capture_output=True, check=False,
        )
        return result.returncode == 0

    else:
        logger.warning(f"Unknown paste method: {method}")
        return False


def paste_text(text: str) -> tuple:
    """Paste text using fallback chain. Returns (success, method_used)."""
    if PASTE_METHOD != "auto":
        chain = [PASTE_METHOD]
    else:
        chain = PASTE_CHAIN

    if not chain:
        logger.warning(f"No paste methods available for {SESSION_TYPE}")
        return (False, "")

    for method in chain:
        try:
            success = _execute_paste(method, text)
            if success:
                logger.info(f"Pasted with method: {method}")
                return (True, method)
            logger.debug(f"Paste method {method} failed, trying next")
        except Exception as e:
            logger.debug(f"Paste method {method} raised {type(e).__name__}: {e}")

    logger.warning("All paste methods exhausted")
    return (False, "")


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class AudioRecorderDaemon:
    """Non-blocking audio recorder. AVAudioRecorder on macOS, parecord on Linux."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.recording_start_time: Optional[float] = None
        self.process: Optional[subprocess.Popen] = None
        self.output_file: Optional[str] = None
        self._av_recorder = None
        # Apple streaming state
        self._av_engine = None
        self._sfs_request = None
        self._sfs_task = None
        self._sfs_recognizer = None
        self._sfs_stopped = False          # guard against late handler calls
        self._sfs_segments: list = []      # finalized per-segment texts
        self._sfs_current_partial: str = ""
        self.live_text: str = ""
        self._final_text: Optional[str] = None

    def start_recording(self) -> bool:
        """Start recording. Returns True on success."""
        if self.is_recording:
            logger.warning("Already recording, ignoring start request")
            return False
        if IS_MACOS:
            if ENGINE == "apple-streaming":
                return self._start_recording_apple_streaming()
            return self._start_recording_macos()
        return self._start_recording_linux()

    def _start_recording_macos(self) -> bool:
        """Record audio via AVAudioRecorder (native macOS; sidesteps
        PortAudio, whose abort()/stop() path can hang indefinitely on
        CoreAudio HAL state transitions)."""
        try:
            from AVFoundation import AVAudioRecorder
            from Foundation import NSURL
        except ImportError:
            logger.error("AVFoundation not available. Run: pip install pyobjc-framework-AVFoundation")
            send_notification("❌ AVFoundation missing")
            return False

        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.output_file = temp_file.name
        temp_file.close()
        # AVAudioRecorder refuses to overwrite an existing file via this init.
        try:
            os.unlink(self.output_file)
        except OSError:
            pass
        logger.info(f"Created temp audio file: {self.output_file}")

        # kAudioFormatLinearPCM = 'lpcm' FourCC as big-endian uint32
        settings = {
            "AVFormatIDKey": 0x6C70636D,
            "AVSampleRateKey": float(self.sample_rate),
            "AVNumberOfChannelsKey": 1,
            "AVLinearPCMBitDepthKey": 16,
            "AVLinearPCMIsFloatKey": False,
            "AVLinearPCMIsBigEndianKey": False,
        }
        url = NSURL.fileURLWithPath_(self.output_file)
        recorder, err = AVAudioRecorder.alloc().initWithURL_settings_error_(url, settings, None)
        if recorder is None:
            logger.error(f"AVAudioRecorder init failed: {err}")
            send_notification("❌ Recording init failed")
            self._cleanup_temp_files()
            return False
        if not recorder.prepareToRecord():
            logger.error("AVAudioRecorder prepareToRecord returned False (mic permission?)")
            send_notification("❌ Mic permission needed for skhd in System Settings")
            self._cleanup_temp_files()
            return False
        if not recorder.record():
            logger.error("AVAudioRecorder record returned False")
            send_notification("❌ Recording failed to start")
            self._cleanup_temp_files()
            return False

        self._av_recorder = recorder
        self.is_recording = True
        self.recording_start_time = time.time()
        logger.info("Recording started (AVAudioRecorder)")
        play_sound(SOUND_START)
        send_notification("🎤 Recording...")
        update_status("recording")
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
                    '--latency-msec=30',
                    self.output_file
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Verify process actually started (short wait to detect immediate failure)
            time.sleep(0.005)
            if self.process.poll() is not None:
                logger.error(f"parecord exited immediately with code {self.process.returncode}")
                send_notification("Recording failed (parecord error)")
                self._cleanup_temp_files()
                return False
        except FileNotFoundError:
            logger.error("parecord not found. Install pulseaudio-utils.")
            send_notification("parecord not found")
            self._cleanup_temp_files()
            return False
        except Exception as e:
            logger.error(f"Failed to start parecord: {e}")
            send_notification(f"Recording failed: {str(e)[:30]}")
            self._cleanup_temp_files()
            return False

        self.is_recording = True
        self.recording_start_time = time.time()
        logger.info(f"Recording started, parecord PID: {self.process.pid}")
        play_sound(SOUND_START)
        send_notification("Recording...")
        update_status("recording")
        return True

    def _start_recording_apple_streaming(self) -> bool:
        """Start live streaming via SFSpeechRecognizer.

        Pipes the AVAudioEngine input tap directly into
        SFSpeechAudioBufferRecognitionRequest and updates self.live_text
        from the result handler. On stop we return the final text;
        stop_and_process treats this engine as "already transcribed" and
        skips the batch pipeline (VAD → normalize → parakeet/whisper).
        """
        try:
            from Speech import (
                SFSpeechRecognizer,
                SFSpeechAudioBufferRecognitionRequest,
                SFSpeechRecognizerAuthorizationStatusAuthorized,
                SFSpeechRecognizerAuthorizationStatusNotDetermined,
                SFSpeechRecognizerAuthorizationStatusDenied,
                SFSpeechRecognizerAuthorizationStatusRestricted,
                SFSpeechRecognitionTaskHintDictation,
            )
            from AVFoundation import AVAudioEngine
        except ImportError as exc:
            logger.error(f"Apple Speech framework unavailable: {exc}")
            send_notification("❌ pyobjc-framework-Speech not installed")
            return False

        status = SFSpeechRecognizer.authorizationStatus()
        if status == SFSpeechRecognizerAuthorizationStatusNotDetermined:
            # First use: actually ask macOS to prompt the user. Wait a few
            # seconds for them to click Allow/Deny; if we never resolve, we
            # abort this recording but the TCC entry is now registered so
            # they can grant it manually next time.
            logger.info("Speech Recognition permission not yet requested; prompting...")
            send_notification("Grant Speech Recognition permission in the prompt")
            done = threading.Event()
            got = [status]

            def _auth_cb(new_status):
                got[0] = new_status
                done.set()

            SFSpeechRecognizer.requestAuthorization_(_auth_cb)
            try:
                from Foundation import NSRunLoop, NSDate
                deadline = time.monotonic() + 15
                while not done.is_set() and time.monotonic() < deadline:
                    NSRunLoop.currentRunLoop().runUntilDate_(
                        NSDate.dateWithTimeIntervalSinceNow_(0.1))
            except ImportError:
                done.wait(15)
            status = got[0]
            if status != SFSpeechRecognizerAuthorizationStatusAuthorized:
                logger.error(f"Speech Recognition not authorized after prompt (status={status})")
                send_notification("❌ Speech Recognition permission required — try again")
                return False
            logger.info("Speech Recognition granted")
        elif status == SFSpeechRecognizerAuthorizationStatusDenied:
            logger.error("Speech Recognition denied by user")
            send_notification("❌ Enable Speech Recognition in System Settings > Privacy")
            return False
        elif status == SFSpeechRecognizerAuthorizationStatusRestricted:
            logger.error("Speech Recognition restricted (parental controls / MDM)")
            send_notification("❌ Speech Recognition restricted on this Mac")
            return False

        recog = SFSpeechRecognizer.alloc().init()
        if recog is None or not recog.isAvailable():
            logger.error("SFSpeechRecognizer unavailable (Dictation enabled in System Settings?)")
            send_notification("❌ Enable Dictation in System Settings → Keyboard")
            return False

        # Reset streaming accumulators.
        self._sfs_stopped = False
        self._sfs_segments = []
        self._sfs_current_partial = ""
        self.live_text = ""
        self._final_text = None
        self._sfs_recognizer = recog

        # SFSpeechRecognizer finalizes its task on extended silence
        # (~1-2 s pause) and on its internal ~60 s limit. When that
        # happens the task stops delivering partials, which the user
        # sees as "my text disappeared when I paused." We handle this by
        # chaining tasks: on isFinal we append the segment's text to
        # self._sfs_segments and immediately start a fresh request+task.
        # The audio tap continues flowing into whichever request is
        # currently installed.
        def handler(result, err):
            if self._sfs_stopped:
                return
            if err is not None:
                msg = str(err)
                logger.info(f"[sfs] handler err: {msg[:120]} — chaining")
                self._start_next_sfs_segment()
                return
            if result is None:
                return
            text = str(result.bestTranscription().formattedString())
            is_final = bool(result.isFinal())
            prev = self._sfs_current_partial

            # Apple doesn't always fire isFinal on short pauses; sometimes
            # it just silently restarts the utterance and the next partial
            # begins a fresh, shorter transcription. Detect that by
            # watching for the partial to shrink or replace its head, and
            # promote the previous partial into a finalized segment so we
            # don't lose what the user already said.
            if prev and not is_final:
                shrank_a_lot = len(text) < max(3, len(prev) // 2)
                head_changed = len(prev) > 15 and text[:10].lower() != prev[:10].lower()
                if shrank_a_lot or head_changed:
                    logger.info(f"[sfs] utterance restart detected; promoting partial to segment: {prev!r}")
                    self._sfs_segments.append(prev)

            self._sfs_current_partial = text
            full = (" ".join(self._sfs_segments) + " " + text).strip()
            self.live_text = full
            update_status("recording", live_text=full)
            if is_final:
                logger.info(f"[sfs] segment final ({len(text)} chars): {text!r} — chaining")
                if text:
                    self._sfs_segments.append(text)
                self._sfs_current_partial = ""
                self._start_next_sfs_segment()

        self._sfs_handler = handler

        req = SFSpeechAudioBufferRecognitionRequest.alloc().init()
        req.setShouldReportPartialResults_(True)
        # Hint dictation so Apple segments on natural pauses.
        if hasattr(req, "setTaskHint_"):
            req.setTaskHint_(SFSpeechRecognitionTaskHintDictation)
        if hasattr(req, "setRequiresOnDeviceRecognition_"):
            req.setRequiresOnDeviceRecognition_(True)
        if hasattr(req, "setAddsPunctuation_"):
            req.setAddsPunctuation_(True)
        self._sfs_request = req
        self._sfs_task = recog.recognitionTaskWithRequest_resultHandler_(req, handler)

        engine = AVAudioEngine.alloc().init()
        inp = engine.inputNode()
        fmt = inp.outputFormatForBus_(0)

        # Tap routes audio to whichever request is current at fire time,
        # so it keeps working after we swap requests on segment boundaries.
        def tap(buf, when):
            r = self._sfs_request
            if r is not None and not self._sfs_stopped:
                r.appendAudioPCMBuffer_(buf)

        inp.installTapOnBus_bufferSize_format_block_(0, 1024, fmt, tap)
        ok, err = engine.startAndReturnError_(None)
        if not ok:
            logger.error(f"AVAudioEngine start failed: {err}")
            self._sfs_task.cancel()
            self._sfs_task = None
            return False

        self._av_engine = engine
        self.is_recording = True
        self.recording_start_time = time.time()
        logger.info("Recording started (Apple streaming)")
        play_sound(SOUND_START)
        send_notification("🎤 Streaming...")
        update_status("recording", live_text="")
        return True

    def _start_next_sfs_segment(self) -> None:
        """Start a fresh SFSpeech request+task after the previous one
        finalized on silence, so recording continues seamlessly."""
        if self._sfs_stopped or self._sfs_recognizer is None:
            return
        try:
            from Speech import (
                SFSpeechAudioBufferRecognitionRequest,
                SFSpeechRecognitionTaskHintDictation,
            )
        except ImportError:
            return
        req = SFSpeechAudioBufferRecognitionRequest.alloc().init()
        req.setShouldReportPartialResults_(True)
        if hasattr(req, "setTaskHint_"):
            req.setTaskHint_(SFSpeechRecognitionTaskHintDictation)
        if hasattr(req, "setRequiresOnDeviceRecognition_"):
            req.setRequiresOnDeviceRecognition_(True)
        if hasattr(req, "setAddsPunctuation_"):
            req.setAddsPunctuation_(True)
        self._sfs_request = req
        self._sfs_task = self._sfs_recognizer.recognitionTaskWithRequest_resultHandler_(
            req, self._sfs_handler)
        logger.info(f"[sfs] started new segment (segments so far: {len(self._sfs_segments)})")

    def _stop_recording_apple_streaming(self) -> Optional[str]:
        """Finalize the streaming recognition and return the text.

        Unlike the batch recorders this doesn't return a file path — it
        returns the transcript directly and stop_and_process knows to
        skip the pipeline.
        """
        logger.info("Stopping recording (Apple streaming)...")
        self.is_recording = False
        # Flip this before endAudio so any late handler callbacks (incl.
        # isFinal from the last segment) are ignored and don't re-write
        # state=recording onto the status JSON after we've moved on.
        self._sfs_stopped = True

        # Give the user a brief tail so the last word isn't clipped,
        # but much shorter than the batch delay — streaming is live.
        time.sleep(min(TRAILING_SPEECH_DELAY, 0.3))

        if self._av_engine is not None:
            try:
                self._av_engine.stop()
                self._av_engine.inputNode().removeTapOnBus_(0)
            except Exception as exc:
                logger.warning(f"AVAudioEngine stop raised: {exc}")
            self._av_engine = None

        last_req = self._sfs_request
        if last_req is not None:
            try:
                last_req.endAudio()
            except Exception as exc:
                logger.warning(f"endAudio raised: {exc}")

        # Wait briefly for the last segment's isFinal to land so we can
        # include the most recent partial as a finalized segment.
        try:
            from Foundation import NSRunLoop, NSDate
            initial_n = len(self._sfs_segments)
            deadline = time.monotonic() + 0.6
            while len(self._sfs_segments) == initial_n and time.monotonic() < deadline:
                NSRunLoop.currentRunLoop().runUntilDate_(
                    NSDate.dateWithTimeIntervalSinceNow_(0.03))
        except ImportError:
            pass

        # Assemble final text. Prefer finalized segments; fall back to
        # the most recent partial if the last segment didn't finalize
        # in time.
        pieces = list(self._sfs_segments)
        if self._sfs_current_partial and (not pieces or pieces[-1] != self._sfs_current_partial):
            pieces.append(self._sfs_current_partial)
        text = " ".join(pieces).strip()

        # Cleanup
        if self._sfs_task is not None:
            try:
                self._sfs_task.cancel()
            except Exception:
                pass
            self._sfs_task = None
        self._sfs_request = None
        self._sfs_recognizer = None
        self._sfs_handler = None
        self._sfs_segments = []
        self._sfs_current_partial = ""

        logger.info(f"Apple streaming transcript: {len(text)} chars")
        return text or None

    def _cleanup_temp_files(self) -> None:
        """Clean up any temp files created during recording."""
        if self.output_file and os.path.exists(self.output_file):
            try:
                os.unlink(self.output_file)
            except OSError:
                pass
            self.output_file = None

    def stop_recording(self) -> Optional[str]:
        """Stop recording. Returns a WAV path for batch engines, or the
        transcribed text directly for streaming engines."""
        if not self.is_recording:
            logger.warning("Not recording, ignoring stop request")
            return None
        if IS_MACOS:
            if ENGINE == "apple-streaming":
                return self._stop_recording_apple_streaming()
            return self._stop_recording_macos()
        return self._stop_recording_linux()

    def _stop_recording_macos(self) -> Optional[str]:
        """Stop AVAudioRecorder and return the written WAV path.

        AVAudioRecorder.stop() is documented as synchronous and finalizes
        the file on return — typically < 10 ms. No watchdog needed.
        """
        logger.info("Stopping recording (AVAudioRecorder)...")
        self.is_recording = False

        time.sleep(TRAILING_SPEECH_DELAY)

        if self._av_recorder is not None:
            t0 = time.monotonic()
            try:
                self._av_recorder.stop()
            except Exception as exc:
                logger.warning(f"AVAudioRecorder.stop() raised: {exc}")
            logger.info(f"AVAudioRecorder stopped in {time.monotonic()-t0:.3f}s")
            self._av_recorder = None

        if not self.output_file or not os.path.exists(self.output_file):
            logger.error(f"Recording file missing: {self.output_file}")
            return None

        file_size = os.path.getsize(self.output_file)
        if file_size == 0:
            logger.error("Recording file is empty")
            return None

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

        update_status("processing")
        return self.output_file

    def cleanup(self) -> None:
        """Cleanup audio resources."""
        if self._av_recorder is not None:
            try:
                self._av_recorder.stop()
            except Exception:
                pass
            self._av_recorder = None
        if self.process and self.process.poll() is None:
            self.process.terminate()


class TranscriptionPipeline:
    """Handles transcription and polishing pipeline."""

    def __init__(self):
        logger.info(f"Initializing TranscriptionPipeline (engine={ENGINE}, model={WHISPER_MODEL}, device={DEVICE}, polishing={ENABLE_POLISHING})")
        self.engine = ENGINE

        if self.engine == "apple-streaming":
            # No model to load — SFSpeechRecognizer runs inside AudioRecorderDaemon.
            # Pipeline exists only to reuse post_transcribe() for clipboard + paste.
            logger.info("Apple streaming engine — no batch model to load")
        elif self.engine == "parakeet":
            logger.info("Loading Parakeet TDT model via onnx-asr...")
            import onnx_asr
            import onnxruntime as rt
            # CoreMLExecutionProvider crashes on parakeet because the encoder
            # uses external-data (.onnx.data) and CoreML's initializer loses
            # the path: "model_path must not be empty" (reproduced on
            # onnxruntime 1.24.x and 1.25.0). Exclude CoreML but keep every
            # other provider — on Linux/NVIDIA this preserves CUDA/TensorRT
            # acceleration; on macOS it falls back to CPU (~107ms inference
            # on Apple Silicon, so still plenty fast).
            providers = [p for p in rt.get_available_providers() if p != "CoreMLExecutionProvider"]
            logger.info(f"onnxruntime providers: {providers}")
            self.model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", providers=providers)
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
                    send_notification(f"{DEVICE} failed, falling back to CPU")
                    self.whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
                else:
                    raise
            logger.info("Whisper model loaded successfully")

        if ENABLE_POLISHING:
            logger.info("Initializing OpenAI client...")
            from openai import OpenAI
            self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.openai = None

        logger.info("TranscriptionPipeline initialized")

    def process(self, audio_path: str) -> str:
        """Full pipeline: transcribe + post-process + paste."""
        logger.info(f"Processing audio file: {audio_path}")

        send_notification("Transcribing...")
        update_status("transcribing")
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
            send_notification("No speech detected (empty audio)")
            update_status("idle")
            return ""
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            send_notification(f"Transcription error: {str(e)[:50]}")
            update_status("error", error=str(e)[:100])
            return ""

        return self.post_transcribe(raw_text)

    def post_transcribe(self, raw_text: str) -> str:
        """Everything that happens after we have transcribed text:
        empty-check, spoken-punctuation, GPT polish, word replacements,
        clipboard, auto-paste, shift-to-submit. Called by process() after
        batch transcription; called directly by stop_and_process for
        streaming engines that produce text without a WAV file."""
        if not raw_text or not raw_text.strip():
            logger.warning("Transcription returned empty text")
            send_notification("No speech detected")
            update_status("idle")
            return ""

        # Spoken punctuation conversion (before polishing so GPT gets cleaner input)
        raw_text = apply_spoken_punctuation(raw_text)

        # Polish if enabled
        if ENABLE_POLISHING and self.openai:
            logger.info("Polishing with GPT...")
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

        # Word replacements (after polishing — user's final override)
        final_text = apply_word_replacements(final_text)

        # Save clipboard before we overwrite it (only when auto-pasting with restore enabled)
        saved_clipboard = save_clipboard() if (AUTO_PASTE and RESTORE_CLIPBOARD) else None

        # Copy to clipboard
        logger.info("Copying to clipboard...")
        _copy_to_clipboard(final_text)
        logger.info("Copied to clipboard")

        # Auto-paste if enabled
        if AUTO_PASTE:
            # Capture shift state before paste (paste itself clears physical modifiers on some platforms)
            submit_after_paste = is_shift_held()
            success, method_used = paste_text(final_text)

            if not success:
                send_notification("Paste failed (text copied to clipboard)")

            # If Shift is held during paste, press Enter to submit
            if submit_after_paste or (not IS_MACOS and is_shift_held()):
                logger.info("Shift-to-submit: pressing Enter")
                time.sleep(0.2)  # wait for paste to be processed by app
                if IS_MACOS:
                    try:
                        from Quartz import (CGEventCreateKeyboardEvent, CGEventSetFlags,
                                            CGEventPost, kCGHIDEventTap)
                        ev = CGEventCreateKeyboardEvent(None, 36, True)
                        CGEventSetFlags(ev, 0)
                        CGEventPost(kCGHIDEventTap, ev)
                        time.sleep(0.05)
                        ev = CGEventCreateKeyboardEvent(None, 36, False)
                        CGEventSetFlags(ev, 0)
                        CGEventPost(kCGHIDEventTap, ev)
                    except Exception:
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

            play_sound(SOUND_PASTE)
            send_notification(f"Pasted: {final_text[:50]}...")

            # Restore clipboard (only for direct-type methods that don't need clipboard)
            if saved_clipboard is not None and method_used in ("xdotool-type", "wtype"):
                time.sleep(0.1)
                restore_clipboard(saved_clipboard)
        else:
            logger.info("Auto-paste disabled")
            play_sound(SOUND_PASTE)
            send_notification(f"Copied: {final_text[:50]}...")

        logger.info("Processing complete")
        update_status("idle", last_text=final_text[:100])
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
        """Process recorded audio: VAD -> normalize -> transcribe."""
        # Grab the file path under lock
        with self._audio_lock:
            audio_file = self._current_audio_file
            self._current_audio_file = None

        if not audio_file:
            return

        try:
            # VAD trimming — skip transcription if no speech
            audio_file = vad_trim_audio(audio_file)
            if audio_file is None:
                return

            # Audio normalization
            audio_file = normalize_audio(audio_file)

            result = self.pipeline.process(audio_file)
            if result:
                print(result, flush=True)
        finally:
            # Cleanup temp file
            if audio_file and os.path.exists(audio_file):
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
        result = self.recorder.stop_recording()
        if not result:
            return
        if ENGINE == "apple-streaming":
            # Streaming engines return the transcript directly; skip the
            # VAD/normalize/parakeet pipeline entirely.
            logger.info("Streaming result — skipping batch pipeline")
            self.pipeline.post_transcribe(result)
            logger.info("Processing complete")
            return
        else:
            with self._audio_lock:
                self._current_audio_file = result
            logger.info("Processing audio synchronously...")
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

    # SFSpeechRecognizer callbacks only fire when there is a pumped
    # NSRunLoop backed by an NSApplication. Initialize early so the
    # streaming engine works out of the box. Harmless for other engines.
    if IS_MACOS:
        try:
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
            _app = NSApplication.sharedApplication()
            _app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        except ImportError:
            pass

    if ENABLE_POLISHING and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set but polishing is enabled")
        print("Error: OPENAI_API_KEY not set (required when ENABLE_POLISHING=true)", file=sys.stderr)
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

    # Launch status overlay — GTK on Linux, native AppKit on macOS.
    # Use the same Python interpreter as the daemon (needed on macOS so the
    # overlay picks up the venv's pyobjc; python3 on PATH won't have it).
    overlay_script = None
    if ENABLE_OVERLAY:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(
            script_dir,
            "whisper-status-macos.py" if IS_MACOS else "whisper-status.py",
        )
        if os.path.exists(candidate):
            overlay_script = candidate

    def spawn_overlay():
        if overlay_script is None:
            return None
        try:
            p = subprocess.Popen(
                [sys.executable, overlay_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"Started status overlay (PID {p.pid}) from {os.path.basename(overlay_script)}")
            return p
        except Exception as e:
            logger.warning(f"Failed to start status overlay: {e}")
            return None

    overlay_proc = spawn_overlay()

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

    # Play calming sound while model loads — loop until model is ready
    # (`afplay` has no loop flag and the chime is only ~4s, shorter than the
    # model load; a background thread restarts it until we signal stop).
    loading_sound_stop = threading.Event()
    loading_sound_thread = None
    if SOUND_LOADING and os.path.exists(SOUND_LOADING):
        player = 'afplay' if IS_MACOS else 'paplay'

        def _loop_chime():
            while not loading_sound_stop.is_set():
                try:
                    subprocess.run(
                        [player, SOUND_LOADING],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    return

        loading_sound_thread = threading.Thread(target=_loop_chime, daemon=True, name="loading-chime")
        loading_sound_thread.start()

    send_notification(f"Loading {ENGINE} model…")
    logger.info(f"Loading {ENGINE} model... (this may take a few seconds on first start)")
    daemon = TranscriptionDaemon()
    daemon.last_activity_time = time.time()

    # Stop loading sound loop and kill any in-flight player
    loading_sound_stop.set()
    subprocess.run(['pkill', '-x', 'afplay' if IS_MACOS else 'paplay'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

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

            # Check for max recording duration
            if (MAX_RECORDING_SECONDS > 0
                    and daemon.recorder.is_recording
                    and daemon.recorder.recording_start_time
                    and time.time() - daemon.recorder.recording_start_time >= MAX_RECORDING_SECONDS):
                logger.info(f"Max recording duration reached ({MAX_RECORDING_SECONDS}s), auto-stopping")
                send_notification(f"Recording auto-stopped ({MAX_RECORDING_SECONDS}s limit)")
                try:
                    os.unlink(state_file)
                except OSError:
                    pass
                daemon.stop_and_process()

            # Keep the overlay alive: respawn if the user/test killed it.
            if overlay_script and (overlay_proc is None or overlay_proc.poll() is not None):
                if overlay_proc is not None:
                    logger.warning(f"Status overlay exited (code {overlay_proc.returncode}); respawning")
                overlay_proc = spawn_overlay()

            # Check for idle timeout
            if daemon.is_idle_timeout_exceeded():
                logger.info(f"Idle timeout exceeded ({daemon.idle_timeout}s), shutting down daemon")
                break

            # Wait for signals or timeout (efficient polling).
            # On macOS, pump the NSRunLoop so SFSpeechRecognizer callbacks
            # fire during streaming. 5ms keeps signal response near-instant
            # for the non-streaming engines too.
            if IS_MACOS:
                try:
                    from Foundation import NSRunLoop, NSDate
                    NSRunLoop.currentRunLoop().runUntilDate_(
                        NSDate.dateWithTimeIntervalSinceNow_(0.005))
                except ImportError:
                    shutdown_event.wait(timeout=0.005)
            else:
                shutdown_event.wait(timeout=0.005)

        logger.info("Daemon loop exiting")

    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up...")
        if overlay_proc and overlay_proc.poll() is None:
            overlay_proc.terminate()
            logger.info("Terminated status overlay")
        daemon.cleanup()
        cleanup_status()
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
