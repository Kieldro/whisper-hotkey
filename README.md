# Whisper Hotkey

Push-to-talk voice transcription for macOS and Linux. Press a hotkey, speak, press again — your words appear instantly as text wherever your cursor is.

Runs 100% locally with [faster-whisper](https://github.com/guillaumekln/faster-whisper), [Parakeet TDT v3](https://github.com/NVIDIA/NeMo), or (macOS) Apple's on-device speech recognizer. No cloud, no subscription, no data leaves your machine.

## How It Works

1. Press your hotkey
2. Speak naturally
3. Press the hotkey again
4. Text is transcribed and pasted into your active window

The model stays loaded in memory so subsequent uses are instant.

## Features

- **Fast local transcription** — faster-whisper (~0.5s GPU, ~0.9s CPU) or Parakeet TDT (~0.3s)
- **Live transcript (macOS)** — `ENGINE=apple-streaming` shows words in the overlay as you speak
- **Auto-paste** — text appears wherever your cursor is
- **Push-to-talk daemon** — background process with audio feedback (start chime, wind chime during model load, completion sound)
- **Shift-to-submit** — hold Shift when pressing stop to press Enter after paste (great for chat apps)
- **Optional GPT polishing** — clean up grammar/formatting via OpenAI API
- **Works offline** — no internet required in default mode
- **Cross-platform** — macOS (skhd) and Linux (X11/Wayland)

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
- Installs [skhd](https://github.com/koekeishiya/skhd) via Homebrew
- Installs Python deps: numpy, **pyobjc-framework-Quartz**, pyobjc-framework-AVFoundation, pyobjc-framework-Speech, and your chosen engine (parakeet, faster-whisper, or apple-streaming)
- Configures Option+Space hotkey in `~/.config/skhd/skhdrc`
- Starts the skhd launchd service (`skhd --start-service`)

**After install — grant Accessibility permission:**
System Settings > Privacy & Security > Accessibility > add skhd

**Recommended engine (Apple Silicon): Parakeet TDT 0.6b-v3** — ~0.3s transcription, built-in punctuation, 25 European languages, beats any faster-whisper size on this hardware. This is the installer default and what `ENGINE=parakeet` selects. `ENGINE=apple-streaming` needs no model download and streams a live transcript; grant Speech Recognition on first use.

**Intel Mac:** Parakeet runs but is very slow. Use `ENGINE=whisper` with `WHISPER_MODEL=small`.

**Performance:**
- First press: ~2.5s startup (model loads once, stays resident)
- Subsequent presses: near-instant (model stays in memory)

**GPU (Apple Silicon):** Parakeet runs on CPU via onnxruntime and is already fast enough. faster-whisper (CTranslate2) has no Apple GPU backend, so `ENGINE=whisper` runs on CPU.

**Paste mechanism:** `pyobjc-framework-Quartz` provides the fast CGEvent Cmd+V path (~5ms) and shift-to-submit detection. If Quartz is missing the daemon falls back to `osascript` (~100–300ms) and silently disables shift-to-submit — so install it.

**Logs:** `tail -f $TMPDIR/whisper-hotkey.log`

### Linux

**Requirements:** Python 3.8+, PulseAudio

**What the installer does:**
- Installs system deps via apt/pacman/dnf (pulseaudio-utils, xclip/wl-clipboard, xdotool/ydotool, libnotify-bin, python3-gi for the overlay)
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
  xclip xdotool libnotify-bin python3-gi gir1.2-gtk-3.0

# Ubuntu/Debian (Wayland)
sudo apt install python3-venv python3-pip pulseaudio-utils \
  wl-clipboard ydotool libnotify-bin python3-gi gir1.2-gtk-3.0
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
| `ENGINE` | `parakeet` | `parakeet` (recommended on Apple Silicon / Linux), `whisper`, or `apple-streaming` (macOS, live transcript) |
| `WHISPER_MODEL` | `small` | Only used when `ENGINE=whisper`: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `DEVICE` | `cpu` | `cpu` or `cuda` (NVIDIA; whisper only) |
| `COMPUTE_TYPE` | `int8` | `int8` (CPU), `float16` (GPU) |
| `AUTO_PASTE` | `true` | Paste transcription into active window |
| `ENABLE_POLISHING` | `false` | GPT grammar/formatting cleanup |
| `OPENAI_API_KEY` | — | Required only if polishing is enabled |
| `IDLE_TIMEOUT` | `600` | Seconds idle before the daemon exits and unloads the model (0 = never) |
| `MAX_RECORDING_SECONDS` | `120` | Auto-stop a recording after this long (0 = no limit) |
| `TRAILING_SPEECH_DELAY` | `0.5` | Seconds to keep recording after the stop press (apple-streaming caps it at 0.3) |
| `VOCABULARY_HINTS` | `Claude,Anthropic` | Comma-separated words to bias recognition toward (apple-streaming and whisper; Parakeet has no hint support, use `replacements.json`) |

## Architecture

```
Hotkey press
  -> records audio (AVAudioRecorder on macOS, parecord on Linux)
     or streams it to Apple's recognizer (ENGINE=apple-streaming)
  -> faster-whisper / Parakeet TDT transcribes locally
  -> (optional) GPT-4o mini polishes text
  -> pastes into active window (CGEvent Cmd+V on macOS, xdotool/ydotool on Linux)
```

The daemon (`transcribe-daemon.py`) loads the model once and stays resident (configurable via `IDLE_TIMEOUT`). First launch plays a wind chime while the model loads, then auto-starts recording.

## Troubleshooting

**Daemon won't start** — check logs for errors. Kill any existing instance with `pkill -f transcribe-daemon` and retry.

**No audio detected (Linux)** — test your mic: `parecord --format=s16le --rate=16000 --channels=1 test.wav`, speak, Ctrl+C, then `paplay test.wav`.

**Not pasting (macOS)** — ensure skhd has Accessibility permission in System Settings > Privacy & Security > Accessibility.

**Not pasting (Linux)** — make sure `xdotool` (X11) or `ydotool` (Wayland) is installed and working.

**Model download fails** — models download on first run. Check your internet connection and try running the daemon manually to see errors.

**Hotkey not working (macOS)** — verify skhd is running (`launchctl list | grep skhd`). Restart with `skhd --restart-service` (not `brew services`, which would start a second skhd that runs every hotkey twice). Check the binding exists in `~/.config/skhd/skhdrc`.

## License

MIT
