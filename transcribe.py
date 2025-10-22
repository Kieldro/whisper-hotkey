#!/usr/bin/env python3
"""
Voice transcription with faster-whisper and GPT-4o mini post-processing.
Records audio on hotkey trigger, transcribes locally, and polishes with LLM.
"""

import os
import sys
import tempfile
import subprocess
import signal
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # tiny, base, small, medium, large-v3
DEVICE = os.getenv("DEVICE", "cpu")  # cpu or cuda
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # int8, float16, float32
ENABLE_POLISHING = os.getenv("ENABLE_POLISHING", "false").lower() == "true"
AUTO_PASTE = os.getenv("AUTO_PASTE", "true").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SAMPLE_RATE = 16000


class AudioRecorder:
    """Handles audio recording via PulseAudio/PipeWire."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.process: Optional[subprocess.Popen] = None

    def start(self, output_file: str) -> None:
        """Start recording to output file."""
        print("🎤 Recording... Press Enter to stop", file=sys.stderr)
        self.process = subprocess.Popen(
            [
                'parecord',
                '--format=s16le',
                f'--rate={self.sample_rate}',
                '--channels=1',
                output_file
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def stop(self) -> None:
        """Stop recording."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("⏹️  Recording stopped", file=sys.stderr)


class WhisperTranscriber:
    """Local transcription using faster-whisper."""

    def __init__(self, model_name: str = WHISPER_MODEL, device: str = DEVICE, compute_type: str = COMPUTE_TYPE):
        print(f"📥 Loading Whisper model: {model_name}...", file=sys.stderr)
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        print("✅ Model loaded", file=sys.stderr)

    def transcribe(self, audio_file: str) -> str:
        """Transcribe audio file to text."""
        print("🔄 Transcribing...", file=sys.stderr)
        try:
            segments, info = self.model.transcribe(audio_file, beam_size=5)
            text = " ".join([segment.text.strip() for segment in segments])
        except ValueError as e:
            print("⚠️  No speech detected (empty audio)", file=sys.stderr)
            return ""
        except Exception as e:
            print(f"❌ Transcription error: {e}", file=sys.stderr)
            return ""

        print(f"📝 Raw transcription: {text}", file=sys.stderr)
        return text


class GPTPolisher:
    """Post-process transcription with GPT-4o mini."""

    def __init__(self, model: str = OPENAI_MODEL):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def polish(self, raw_text: str) -> str:
        """Correct grammar, punctuation, and formatting."""
        print("✨ Polishing with GPT-4o mini...", file=sys.stderr)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a text correction assistant. Fix grammar, punctuation, and formatting. Preserve the original meaning and tone. Return only the corrected text without explanations."
                },
                {
                    "role": "user",
                    "content": raw_text
                }
            ],
            temperature=0.3
        )

        polished_text = response.choices[0].message.content.strip()
        print(f"✅ Polished: {polished_text}", file=sys.stderr)
        return polished_text


def copy_to_clipboard(text: str) -> None:
    """Copy text to system clipboard using xclip."""
    try:
        subprocess.run(
            ['xclip', '-selection', 'clipboard'],
            input=text.encode(),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("📋 Copied to clipboard", file=sys.stderr)

        # Auto-paste if enabled
        if AUTO_PASTE:
            import time
            time.sleep(0.1)  # Brief delay to ensure clipboard is populated
            try:
                subprocess.run(
                    ['xdotool', 'key', 'ctrl+v'],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print("✅ Auto-pasted", file=sys.stderr)
            except subprocess.CalledProcessError:
                print("⚠️  Failed to auto-paste (xdotool error)", file=sys.stderr)
            except FileNotFoundError:
                print("⚠️  xdotool not found. Install with: sudo apt install xdotool", file=sys.stderr)
    except subprocess.CalledProcessError:
        print("⚠️  Failed to copy to clipboard (xclip not installed?)", file=sys.stderr)
    except FileNotFoundError:
        print("⚠️  xclip not found. Install with: sudo apt install xclip", file=sys.stderr)


def main():
    """Main transcription pipeline."""
    # Validate API key if polishing is enabled
    if ENABLE_POLISHING and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set in .env file (required when ENABLE_POLISHING=true)", file=sys.stderr)
        sys.exit(1)

    # Create temporary audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        audio_path = f.name

    try:
        # Initialize components
        recorder = AudioRecorder()
        transcriber = WhisperTranscriber()
        polisher = GPTPolisher() if ENABLE_POLISHING else None

        # Record audio
        recorder.start(audio_path)
        input()  # Wait for Enter key
        recorder.stop()

        # Transcribe
        raw_text = transcriber.transcribe(audio_path)

        if not raw_text.strip():
            print("⚠️  No speech detected", file=sys.stderr)
            sys.exit(0)

        # Polish with GPT if enabled
        if ENABLE_POLISHING and polisher:
            final_text = polisher.polish(raw_text)
        else:
            final_text = raw_text
            print("ℹ️  Polishing disabled (local-only mode)", file=sys.stderr)

        # Output and copy to clipboard
        print("\n" + "="*50, file=sys.stderr)
        print(final_text)
        print("="*50, file=sys.stderr)

        copy_to_clipboard(final_text)

    finally:
        # Cleanup
        if os.path.exists(audio_path):
            os.unlink(audio_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Cancelled", file=sys.stderr)
        sys.exit(0)
