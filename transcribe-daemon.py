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
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv
import pyaudio
import wave
import time

# Load environment variables
load_dotenv()

# Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
ENABLE_POLISHING = os.getenv("ENABLE_POLISHING", "false").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024


class AudioRecorderDaemon:
    """Non-blocking audio recorder for daemon mode."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.chunk_size = CHUNK_SIZE
        self.is_recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None

    def start_recording(self) -> None:
        """Start recording in background thread."""
        if self.is_recording:
            return

        self.is_recording = True
        self.frames = []

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )

        self.stream.start_stream()
        self._notify("🎤 Recording...")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream."""
        if self.is_recording:
            self.frames.append(in_data)
            return (in_data, pyaudio.paContinue)
        return (in_data, pyaudio.paComplete)

    def stop_recording(self) -> Optional[str]:
        """Stop recording and save to temporary file."""
        if not self.is_recording:
            return None

        self.is_recording = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if not self.frames:
            self._notify("⚠️  No audio recorded")
            return None

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wav_path = temp_file.name
        temp_file.close()

        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))

        self._notify("⏹️  Recording stopped")
        return wav_path

    def _notify(self, message: str) -> None:
        """Send desktop notification."""
        subprocess.run(
            ['notify-send', '-t', '2000', 'Voice Transcription', message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def cleanup(self) -> None:
        """Cleanup audio resources."""
        if self.stream:
            self.stream.close()
        self.audio.terminate()


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
        segments, _ = self.whisper.transcribe(audio_path, beam_size=5)
        raw_text = " ".join([seg.text.strip() for seg in segments])

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

    daemon = TranscriptionDaemon()

    try:
        # Toggle mode: start recording, wait for second invocation to stop
        state_file = "/tmp/whisper-hotkey-state"

        if os.path.exists(state_file):
            # Already recording, stop and process
            daemon.stop_and_process()
            os.unlink(state_file)
        else:
            # Start recording
            daemon.start_recording()
            Path(state_file).touch()

            # Keep process alive for recording
            while daemon.recorder.is_recording:
                time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        daemon.cleanup()


if __name__ == "__main__":
    main()
