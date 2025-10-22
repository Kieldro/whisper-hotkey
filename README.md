# Voice Transcription with Hotkey

Linux implementation of Superwhisper-style voice transcription using faster-whisper (local) with optional GPT-4o mini polishing (API).

## Architecture

**Local-only mode (default, no API key required):**
```
[Microphone] → [PulseAudio] → [faster-whisper] → [Clipboard]
     ↓              ↓                ↓                  ↓
  Hardware      Recording      Local STT           Output
                 (local)        (~500MB)
```

**With GPT polishing (optional, requires OpenAI API key):**
```
[Microphone] → [PulseAudio] → [faster-whisper] → [GPT-4o mini] → [Clipboard]
     ↓              ↓                ↓                  ↓              ↓
  Hardware      Recording      Local STT          Polishing      Output
                 (local)        (~500MB)         (API call)
```

## Features

- **Local transcription**: Uses faster-whisper (Whisper small model, ~500MB)
- **Optional cloud polishing**: GPT-4o mini corrects grammar and formatting (disabled by default)
- **Works offline**: Local-only mode requires no internet or API keys
- **Two usage modes**: Simple (terminal-based) or Daemon (push-to-talk)
- **Hotkey trigger**: Integrates with i3/sway/KDE/GNOME
- **Desktop notifications**: Visual feedback during processing (daemon mode)
- **Clipboard output**: Automatically copies result
- **Low latency**: ~2s local-only, ~3-5s with polishing

## Usage Modes

### Simple Mode (`transcribe.py`)
- Opens terminal window
- Press Enter to stop recording
- Good for testing and manual use

### Daemon Mode (`transcribe-daemon.py`) **Recommended**
- Background process, no terminal window
- Press hotkey once to start, again to stop
- Desktop notifications for feedback
- More Superwhisper-like experience

**Comparison:**

| Feature | Simple Mode | Daemon Mode |
|---------|-------------|-------------|
| Terminal window | Yes | No |
| Stop recording | Press Enter | Press hotkey again |
| Visual feedback | Terminal output | Desktop notifications |
| Background process | No | Yes |
| UX | Basic | Superwhisper-like |

## Requirements

**Minimum (local-only mode):**
- Python 3.8+
- PulseAudio or PipeWire (for audio recording)
- xclip (for clipboard support)

**Optional (for polishing):**
- OpenAI API key (get one at https://platform.openai.com/api-keys)

## Installation

### 1. Install System Dependencies

```bash
# Debian/Ubuntu
sudo apt install python3-venv python3-pip portaudio19-dev xclip libnotify-bin

# Arch Linux
sudo pacman -S python python-pip portaudio xclip libnotify

# Fedora
sudo dnf install python3-virtualenv portaudio-devel xclip libnotify
```

### 2. Clone and Setup

```bash
cd ~/repos
git clone <your-repo-url> whisper-hotkey
cd whisper-hotkey

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure (Optional)

**For local-only mode** (works immediately, no setup needed):
```bash
# Already configured! .env file is pre-configured for local-only mode
# ENABLE_POLISHING=false (default)
```

**To enable GPT polishing** (optional):
```bash
# Edit .env file
nano .env

# Set these values:
ENABLE_POLISHING=true
OPENAI_API_KEY=sk-...your-key-here
```

### 4. Setup Hotkey

```bash
# Automatic setup for i3/sway
./scripts/setup-hotkey.sh

# Or follow manual instructions for KDE/GNOME
```

## Usage

### Simple Mode (Manual Testing)

```bash
# Activate virtual environment
source venv/bin/activate

# Run transcription script
python transcribe.py

# Speak into microphone
# Press Enter when done recording
```

### Daemon Mode (Recommended for Daily Use)

**Setup hotkey:**

For **i3/sway**, edit config to use the wrapper script:
```bash
bindsym $mod+Shift+v exec ~/repos/whisper-hotkey/scripts/hotkey-wrapper.sh
```

For **KDE/GNOME**, configure custom shortcut with command:
```bash
~/repos/whisper-hotkey/scripts/hotkey-wrapper.sh
```

**Usage:**
1. Press hotkey once → Recording starts (notification appears)
2. Speak into microphone
3. Press hotkey again → Recording stops, transcription begins
4. Wait for notifications:
   - **Local-only**: Transcribing → Copied
   - **With polishing**: Transcribing → Polishing → Copied
5. Result automatically in clipboard

**No terminal window required!**

## Configuration

Edit `.env` to customize:

| Variable | Default | Options | Notes |
|----------|---------|---------|-------|
| `ENABLE_POLISHING` | `false` | `true`, `false` | Enable GPT polishing (requires API key) |
| `OPENAI_API_KEY` | - | Your API key | Only needed if polishing enabled |
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` | Affects accuracy and speed |
| `DEVICE` | `cpu` | `cpu`, `cuda` | Use `cuda` for GPU acceleration |
| `COMPUTE_TYPE` | `int8` | `int8`, `float16`, `float32` | Lower = faster, less accurate |
| `OPENAI_MODEL` | `gpt-4o-mini` | `gpt-4o-mini`, `gpt-4o` | Only used if polishing enabled |

### Model Comparison

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 75 MB | Fastest | Low |
| base | 142 MB | Fast | Medium |
| small | 466 MB | Balanced | Good |
| medium | 1.5 GB | Slow | Better |
| large-v3 | 2.9 GB | Slowest | Best |

**Recommended**: `small` for best speed/accuracy balance

## GPU Acceleration (Optional)

For faster transcription with NVIDIA GPU:

```bash
# Install CUDA support
pip install faster-whisper[cuda]

# Update .env
DEVICE=cuda
COMPUTE_TYPE=float16
```

## Cost Estimation

**Local-only mode (default):**
- **100% Free** - No API costs, runs entirely on your machine

**With polishing enabled:**
- **Whisper**: Free (runs locally)
- **GPT-4o mini**: ~$0.00001 per transcription
  - Input: $0.15/1M tokens
  - Output: $0.60/1M tokens
  - Average 100-word correction: ~150 tokens ≈ $0.00001
  - **Example**: 1000 transcriptions ≈ $0.01

## Troubleshooting

### No audio detected

```bash
# Test PulseAudio
parecord --format=s16le --rate=16000 --channels=1 test.wav
# Speak for a few seconds, then Ctrl+C

# Play back
paplay test.wav
```

### xclip not working

```bash
# Verify xclip is installed
which xclip

# Test manually
echo "test" | xclip -selection clipboard
```

### Whisper model download fails

Models download automatically on first run. If it fails:

```bash
# Manual download
python -c "from faster_whisper import WhisperModel; WhisperModel('small')"
```

### Permission denied on transcribe.py

```bash
chmod +x transcribe.py
```

## Project Structure

```
whisper-hotkey/
├── transcribe.py           # Main transcription script
├── requirements.txt        # Python dependencies
├── .env.example           # Configuration template
├── .env                   # Your API keys (git-ignored)
├── scripts/
│   └── setup-hotkey.sh    # Automatic hotkey configuration
├── config/                # Future configuration files
└── README.md
```

## Advanced Usage

### Custom Polishing Prompt

Edit `transcribe.py:82` to customize GPT behavior:

```python
"content": "Your custom instructions here"
```

### Integration with Other Apps

Pipe output to other commands:

```bash
python transcribe.py | your-command
```

### Background Recording

For continuous monitoring (advanced):

```bash
# Run in background
nohup python transcribe.py &

# Check output
tail -f nohup.out
```

## Performance

Typical latency breakdown (Intel i5, 10s audio):
- Recording: 10s (user-controlled)
- Transcription (small model): ~2s
- GPT-4o mini API: ~1-2s
- **Total**: ~13-14s

With GPU (CUDA):
- Transcription: ~0.5-1s
- **Total**: ~11-13s

## License

MIT

## Contributing

Pull requests welcome. For major changes, open an issue first.

## Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [OpenAI API](https://platform.openai.com/)
