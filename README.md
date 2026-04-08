# Whisper Hotkey

Push-to-talk voice transcription for Linux. Press a hotkey, speak, press again — text appears where your cursor is.

Runs locally with [Parakeet TDT](https://github.com/NVIDIA/NeMo) or [faster-whisper](https://github.com/guillaumekln/faster-whisper). No cloud, no subscription.

## Quick Start

```bash
git clone https://github.com/Kieldro/whisper-hotkey.git
cd whisper-hotkey
./install.sh
```

Bind your hotkey to `scripts/hotkey-wrapper.sh`. Press it, speak, press again.

## Features

- **~0.3s transcription** with Parakeet TDT (25 languages) or faster-whisper on GPU
- **Auto-paste** into the active window via xdotool/ydotool/wtype
- **Voice Activity Detection** — trims silence, skips transcription when no speech detected
- **Audio normalization** — handles quiet and loud mics automatically
- **Paste fallback chain** — tries multiple paste methods until one works
- **Clipboard preservation** — saves and restores your clipboard after pasting
- **Word replacements** — fix recurring misrecognitions via `replacements.json`
- **Status overlay** — floating widget shows recording/transcribing state
- **Shift-to-submit** — hold Shift during paste to press Enter (for chat apps)
- **Optional GPT polishing** — grammar cleanup via OpenAI API
- **Spoken punctuation** — say "period", "comma", "new line" (opt-in)
- **X11 and Wayland** support with auto-detection

## Configuration

Edit `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `ENGINE` | `parakeet` | `parakeet` or `whisper` |
| `DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` |
| `COMPUTE_TYPE` | `int8` | `int8` (CPU), `float16` (GPU) |
| `AUTO_PASTE` | `true` | Auto-paste into active window |
| `PASTE_METHOD` | `auto` | `auto`, `wtype`, `xdotool-type`, `xdotool-clipboard`, `ydotool-clipboard` |
| `ENABLE_VAD` | `true` | Trim silence, skip empty recordings |
| `VAD_THRESHOLD` | `0.5` | Speech detection sensitivity (0-1) |
| `ENABLE_AUDIO_NORMALIZATION` | `true` | Normalize mic volume levels |
| `ENABLE_SPOKEN_PUNCTUATION` | `false` | Convert "period" to `.`, etc. |
| `ENABLE_NOTIFICATIONS` | `true` | Desktop notification banners |
| `REPLACEMENTS_FILE` | `./replacements.json` | Word replacement dictionary |
| `IDLE_TIMEOUT` | `600` | Seconds before model unloads (0 = never) |
| `TRAILING_SPEECH_DELAY` | `0.2` | Seconds to record after hotkey release |
| `ENABLE_POLISHING` | `false` | GPT grammar cleanup (needs `OPENAI_API_KEY`) |

## Word Replacements

Create `replacements.json` to fix words the model consistently gets wrong:

```json
{
  "gonna": "going to",
  "kubernetes": "Kubernetes",
  "mycorp": "MyCorp"
}
```

## Status Overlay

A floating widget that shows recording/transcribing state:

```bash
python3 whisper-status.py &
```

Auto-hides when idle, reappears on next recording.

## GPU Acceleration

```bash
# In .env
DEVICE=cuda
COMPUTE_TYPE=float16
```

## Architecture

```
Hotkey → Record (parecord) → VAD trim → Normalize
→ Transcribe (Parakeet/Whisper) → Punctuation → Replacements
→ Save clipboard → Paste (fallback chain) → Restore clipboard
```

The daemon loads the model once and stays resident. First press plays a chime while loading (~2-3s), then auto-starts recording.

## Troubleshooting

- **Check logs**: `tail -f $XDG_RUNTIME_DIR/whisper-hotkey.log`
- **Kill daemon**: `pkill -f transcribe-daemon`
- **Test mic**: `parecord --format=s16le --rate=16000 --channels=1 test.wav`
- **Status file**: `cat $XDG_RUNTIME_DIR/whisper-hotkey-status.json`

## License

MIT
