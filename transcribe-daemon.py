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
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
ENABLE_POLISHING = os.getenv("ENABLE_POLISHING", "false").lower() == "true"
AUTO_PASTE = os.getenv("AUTO_PASTE", "true").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024


class AudioRecorderDaemon:
    """Non-blocking audio recorder using parecord."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.process: Optional[subprocess.Popen] = None
        self.output_file: Optional[str] = None

    def start_recording(self) -> None:
        """Start recording with parecord."""
        if self.is_recording:
            return

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.output_file = temp_file.name
        temp_file.close()

        # Start parecord
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
        self._notify("🎤 Recording...")

    def stop_recording(self) -> Optional[str]:
        """Stop recording and return file path."""
        if not self.is_recording:
            return None

        self.is_recording = False

        if self.process:
            self.process.terminate()
            self.process.wait(timeout=2)

        self._notify("⏹️  Recording stopped")
        return self.output_file

    def _notify(self, message: str) -> None:
        """Send desktop notification."""
        subprocess.run(
            ['notify-send', '-t', '2000', 'Voice Transcription', message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def cleanup(self) -> None:
        """Cleanup audio resources."""
        if self.process and self.process.poll() is None:
            self.process.terminate()


class TranscriptionPipeline:
    """Handles transcription and polishing pipeline."""

    def __init__(self):
        self._notify("📥 Loading models...")
        self.whisper = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if ENABLE_POLISHING else None
        if ENABLE_POLISHING:
            self._notify("✅ Models loaded (with polishing)")
        else:
            self._notify("✅ Whisper loaded (local-only mode)")

    def process(self, audio_path: str) -> str:
        """Full pipeline: transcribe + polish."""
        # Transcribe
        self._notify("🔄 Transcribing...")
        try:
            segments, _ = self.whisper.transcribe(audio_path, beam_size=5)
            raw_text = " ".join([seg.text.strip() for seg in segments])
        except ValueError as e:
            # Empty audio or no detectable language
            self._notify("⚠️  No speech detected (empty audio)")
            return ""
        except Exception as e:
            self._notify(f"❌ Transcription error: {str(e)[:50]}")
            return ""

        if not raw_text.strip():
            self._notify("⚠️  No speech detected")
            return ""

        # Polish if enabled
        if ENABLE_POLISHING and self.openai:
            self._notify("✨ Polishing...")
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
        else:
            final_text = raw_text

        # Copy to clipboard
        subprocess.run(
            ['xclip', '-selection', 'clipboard'],
            input=final_text.encode(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Auto-paste if enabled
        if AUTO_PASTE:
            time.sleep(0.1)  # Brief delay to ensure clipboard is populated
            subprocess.run(
                ['xdotool', 'key', 'ctrl+v'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self._notify(f"✅ Pasted: {final_text[:50]}...")
        else:
            self._notify(f"✅ Copied: {final_text[:50]}...")

        return final_text

    def _notify(self, message: str) -> None:
        """Send desktop notification."""
        subprocess.run(
            ['notify-send', '-t', '3000', 'Voice Transcription', message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


class TranscriptionDaemon:
    """Background daemon handling recording and transcription."""

    def __init__(self):
        self.recorder = AudioRecorderDaemon()
        self.pipeline = TranscriptionPipeline()
        self.current_audio_file: Optional[str] = None

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
        audio_path = self.recorder.stop_recording()
        if audio_path:
            self.current_audio_file = audio_path
            threading.Thread(target=self._process_audio, daemon=True).start()

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.recorder.cleanup()


def main():
    """Run daemon in toggle mode (for hotkey integration)."""
    if ENABLE_POLISHING and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set (required when ENABLE_POLISHING=true)", file=sys.stderr)
        sys.exit(1)

    state_file = "/tmp/whisper-hotkey-state"
    pid_file = "/tmp/whisper-hotkey-pid"

    if os.path.exists(state_file):
        # Already recording, send signal to stop
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGUSR1)  # Signal to stop recording
            os.unlink(state_file)
            os.unlink(pid_file)
        except (FileNotFoundError, ProcessLookupError, ValueError):
            # Process already dead, clean up
            if os.path.exists(state_file):
                os.unlink(state_file)
            if os.path.exists(pid_file):
                os.unlink(pid_file)
        sys.exit(0)

    # Start new recording session
    daemon = TranscriptionDaemon()

    # Write our PID
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

    Path(state_file).touch()

    # Set up signal handler to stop recording
    stop_requested = threading.Event()

    def handle_stop(signum, frame):
        stop_requested.set()

    signal.signal(signal.SIGUSR1, handle_stop)

    try:
        daemon.start_recording()

        # Wait for stop signal
        stop_requested.wait()

        # Process the recording
        daemon.stop_and_process()

    except KeyboardInterrupt:
        pass
    finally:
        daemon.cleanup()
        if os.path.exists(state_file):
            os.unlink(state_file)
        if os.path.exists(pid_file):
            os.unlink(pid_file)


if __name__ == "__main__":
    main()
