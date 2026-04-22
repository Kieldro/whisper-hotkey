# Whisper Hotkey

Push-to-talk voice transcription for macOS and Linux. Press a hotkey, speak, press again — your words appear instantly as text wherever your cursor is.

Runs 100% locally with [faster-whisper](https://github.com/guillaumekln/faster-whisper) or [Parakeet TDT v3](https://github.com/NVIDIA/NeMo). No cloud, no subscription, no data leaves your machine.

## How It Works

1. Press your hotkey
2. Speak naturally
3. Press the hotkey again
4. Text is transcribed and pasted into your active window

The model stays loaded in memory so subsequent uses are instant.

## Features

- **Fast local transcription** — faster-whisper (~0.5s GPU, ~0.9s CPU) or Parakeet TDT (~0.3s)
- **Auto-paste** — text appears wherever your cursor is
- **Push-to-talk daemon** — background process with audio feedback (start chime, wind chime during model load, completion sound)
- **Shift-to-submit** — hold Shift when pressing stop to press Enter after paste (great for chat apps)
- **Optional GPT polishing** — clean up grammar/formatting via OpenAI API
- **Works offline** — no internet required in default mode
- **Cross-platform** — macOS (Hammerspoon) and Linux (X11/Wayland)

## Quick Start

```bash
git clone https://github.com/Kieldro/whisper-hotkey.git
cd whisper-hotkey
./install.sh
```

The installer auto-detects your OS and handles dependencies, Python environment, engine selection, GPU detection, hotkey setup, and model download.

## Platform Notes

### macOS

**Requirements:** macOS 10.15+, Python 3.11+, [Homebrew](https://brew.sh)

**What the installer does:**
- Installs [Hammerspoon](https://www.hammerspoon.org/) via Homebrew
- Installs Python deps: sounddevice, soundfile, numpy, faster-whisper
- Configures Option+Space hotkey in `~/.hammerspoon/init.lua`
- Offers MPS acceleration on Apple Silicon

**After install — grant Accessibility permission:**
System Settings > Privacy & Security > Accessibility > add Hammerspoon

**Default engine:** faster-whisper with `tiny` model and `int8` compute — ~0.9s transcription for 5s of speech.

**Performance:**
- First press: ~2.5s startup (model loads once, stays resident)
- Subsequent presses: near-instant (audio stream stays open, model in memory)

**GPU (Apple Silicon):** Set `DEVICE=mps` and `COMPUTE_TYPE=float16` in `.env` for faster-whisper acceleration.

**Parakeet engine:** Not recommended on Intel Mac (very slow). Only use on Apple Silicon.

**Paste mechanism:** Uses `pbcopy` + `osascript` Cmd+V (requires Accessibility permission).

**Logs:** `tail -f $TMPDIR/whisper-hotkey.log`

### Linux

**Requirements:** Python 3.8+, PulseAudio

**What the installer does:**
- Installs system deps via apt/pacman/dnf (pulseaudio-utils, xclip/wl-clipboard, xdotool/ydotool, libnotify-bin)
- Sets up hotkey binding for your desktop environment (i3, sway, GNOME, KDE)

**Default engine:** Parakeet TDT (`nemo-parakeet-tdt-0.6b-v3`) — ~0.3s transcription, 25 languages.

**Hotkey:** Mod+Shift+V (i3/sway), or custom shortcut (GNOME/KDE). See `scripts/setup-hotkey.sh`.

**GPU (NVIDIA):** Set `DEVICE=cuda` and `COMPUTE_TYPE=float16` in `.env`.

If you get cuDNN errors on Ubuntu 24.04:
```bash
sudo apt install nvidia-cudnn
pip install --force-reinstall ctranslate2==4.4.0
```

**Paste mechanism:** `xdotool type` (X11) or `ydotool` Ctrl+V (Wayland).

**Shift-to-submit (X11 only):** Detects Shift key via XQueryKeymap. Not available on Wayland.

**Logs:** `tail -f ${XDG_RUNTIME_DIR:-/tmp}/whisper-hotkey.log`

<details>
<summary>Manual installation (Linux)</summary>

### 1. Install dependencies

```bash
# Ubuntu/Debian (X11)
sudo apt install python3-venv python3-pip pulseaudio-utils \
  xclip xdotool libnotify-bin

# Ubuntu/Debian (Wayland)
sudo apt install python3-venv python3-pip pulseaudio-utils \
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

Bind your preferred key combo to run `~/repos/whisper-hotkey/scripts/hotkey-wrapper.sh`.

**i3/sway** — add to config:
```
bindsym $mod+Shift+v exec ~/repos/whisper-hotkey/scripts/hotkey-wrapper.sh
```

**GNOME/KDE** — add a custom keyboard shortcut pointing to the same script.

</details>

## Configuration

Copy `.env.example` to `.env` and edit as needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGINE` | `whisper` | `whisper` (recommended) or `parakeet` |
| `WHISPER_MODEL` | `tiny` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `DEVICE` | `cpu` | `cpu`, `cuda` (NVIDIA), or `mps` (Apple Silicon) |
| `COMPUTE_TYPE` | `int8` | `int8` (CPU), `float16` (GPU/MPS) |
| `AUTO_PASTE` | `true` | Paste transcription into active window |
| `ENABLE_POLISHING` | `false` | GPT grammar/formatting cleanup |
| `OPENAI_API_KEY` | — | Required only if polishing is enabled |
| `IDLE_TIMEOUT` | `600` | Seconds before daemon unloads model (0 = never) |
| `TRAILING_SPEECH_DELAY` | `1.2` | Seconds to keep recording after hotkey release |

## Architecture

```
Hotkey press
  -> records audio (sounddevice on macOS, parecord on Linux)
  -> faster-whisper / Parakeet TDT transcribes locally
  -> (optional) GPT-4o mini polishes text
  -> pastes into active window (pbcopy+osascript on macOS, xdotool/ydotool on Linux)
```

The daemon (`transcribe-daemon.py`) loads the model once and stays resident (configurable via `IDLE_TIMEOUT`). First launch plays a wind chime while the model loads, then auto-starts recording.

## Troubleshooting

**Daemon won't start** — check logs for errors. Kill any existing instance with `pkill -f transcribe-daemon` and retry.

**No audio detected (Linux)** — test your mic: `parecord --format=s16le --rate=16000 --channels=1 test.wav`, speak, Ctrl+C, then `paplay test.wav`.

**Not pasting (macOS)** — ensure Hammerspoon has Accessibility permission in System Settings > Privacy & Security > Accessibility.

**Not pasting (Linux)** — make sure `xdotool` (X11) or `ydotool` (Wayland) is installed and working.

**Model download fails** — models download on first run. Check your internet connection and try running the daemon manually to see errors.

**Hotkey not working (macOS)** — open the Hammerspoon console (click menu bar icon) and reload config with Cmd+Shift+R.

## License

MIT
