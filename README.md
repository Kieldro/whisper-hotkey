# Whisper Hotkey

Push-to-talk voice transcription for Linux. Press a hotkey, speak, press again — your words appear instantly as text wherever your cursor is.

Runs 100% locally with [Parakeet TDT](https://github.com/NVIDIA/NeMo) or [faster-whisper](https://github.com/guillaumekln/faster-whisper). No cloud, no subscription, no data leaves your machine.

## How It Works

1. Press your hotkey
2. Speak naturally
3. Press the hotkey again
4. Text is transcribed and pasted into your active window

Transcription takes ~0.3-0.6s after you stop recording. The model stays loaded in memory so subsequent uses are instant.

## Features

- **Fast local transcription** — Parakeet TDT (~0.3s) or faster-whisper (~0.5s) on GPU
- **Auto-paste** — text appears wherever your cursor is, no manual pasting
- **Push-to-talk daemon** — background process with audio feedback (start chime, wind chime during model load, completion sound after paste)
- **Pre-recording buffer** — captures up to 2s of audio before you press the hotkey, so your first words aren't lost
- **Optional GPT polishing** — clean up grammar/formatting via OpenAI API
- **Works offline** — no internet required in default mode
- **X11 and Wayland** — auto-detects your session type

## Quick Start

### 1. Install dependencies

```bash
# Ubuntu/Debian (X11)
sudo apt install python3-venv python3-pip portaudio19-dev pulseaudio-utils \
  xclip xdotool libnotify-bin

# Ubuntu/Debian (Wayland)
sudo apt install python3-venv python3-pip portaudio19-dev pulseaudio-utils \
  wl-clipboard ydotool libnotify-bin
# sudo systemctl enable --now ydotool
```

### 2. Clone and install

```bash
git clone https://github.com/Kieldro/whisper-hotkey.git
cd whisper-hotkey
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up your hotkey

Bind your preferred key combo to run:

```
~/repos/whisper-hotkey/scripts/hotkey-wrapper.sh
```

**i3/sway** — add to config:
```
bindsym $mod+Shift+v exec ~/repos/whisper-hotkey/scripts/hotkey-wrapper.sh
```

**GNOME/KDE** — add a custom keyboard shortcut pointing to the same script.

### 4. Use it

Press your hotkey, talk, press it again. Done.

## Configuration

Copy `.env.example` to `.env` and edit as needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGINE` | `parakeet` | `parakeet` (faster) or `whisper` |
| `WHISPER_MODEL` | `small` | Whisper model size (ignored for parakeet) |
| `DEVICE` | `cpu` | `cpu` or `cuda` for GPU acceleration |
| `COMPUTE_TYPE` | `int8` | `int8` (CPU), `float16` (GPU) |
| `AUTO_PASTE` | `true` | Paste transcription into active window |
| `PRE_RECORDING_BUFFER` | `2` | Seconds of audio to capture before hotkey (0 = off) |
| `ENABLE_POLISHING` | `false` | GPT grammar/formatting cleanup |
| `OPENAI_API_KEY` | — | Required only if polishing is enabled |
| `IDLE_TIMEOUT` | `600` | Seconds before daemon unloads model (0 = never) |

## GPU Acceleration

For 3-5x faster transcription with an NVIDIA GPU:

```bash
# .env
DEVICE=cuda
COMPUTE_TYPE=float16
```

If you get cuDNN errors on Ubuntu 24.04:
```bash
sudo apt install nvidia-cudnn
pip install --force-reinstall ctranslate2==4.4.0
```

## Architecture

```
Hotkey press
  → parecord captures audio
  → Parakeet TDT / faster-whisper transcribes locally
  → (optional) GPT-4o mini polishes text
  → xdotool/ydotool pastes into active window
```

The daemon (`transcribe-daemon.py`) loads the model once and stays resident for 10 minutes (configurable). First launch plays a wind chime while the model loads (~2-3s), then auto-starts recording. Subsequent recordings start instantly.

## Troubleshooting

**No audio detected** — test your mic with `parecord --format=s16le --rate=16000 --channels=1 test.wav`, speak, Ctrl+C, then `paplay test.wav`.

**Not pasting** — make sure `xdotool` (X11) or `ydotool` (Wayland) is installed and working.

**Model download fails** — models download on first run. Run `python3 -c "import onnx_asr; onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v2')"` manually to debug.

**Check logs** — `tail -f /tmp/whisper-hotkey.log`

## License

MIT
