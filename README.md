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
- **Auto-paste**: Automatically pastes transcription into active window (enabled by default)
- **Pre-recording buffer**: Captures audio before hotkey press (2s default)
- **Works offline**: Local-only mode requires no internet or API keys
- **Two usage modes**: Simple (terminal-based) or Daemon (push-to-talk)
- **Hotkey trigger**: Integrates with i3/sway/KDE/GNOME
- **Desktop notifications**: Visual feedback during processing (daemon mode)
- **X11 and Wayland support**: Auto-detects session type and uses appropriate tools
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
- **X11**: xclip (clipboard) + xdotool (auto-paste)
- **Wayland**: wl-clipboard (clipboard) + ydotool (auto-paste)

**Optional (for polishing):**
- OpenAI API key (get one at https://platform.openai.com/api-keys)

## Installation

### 1. Install System Dependencies

```bash
# Debian/Ubuntu (X11)
sudo apt install python3-venv python3-pip portaudio19-dev xclip xdotool libnotify-bin

# Debian/Ubuntu (Wayland)
sudo apt install python3-venv python3-pip portaudio19-dev wl-clipboard ydotool libnotify-bin

# Arch Linux (X11)
sudo pacman -S python python-pip portaudio xclip xdotool libnotify

# Arch Linux (Wayland)
sudo pacman -S python python-pip portaudio wl-clipboard ydotool libnotify

# Fedora (X11)
sudo dnf install python3-virtualenv portaudio-devel xclip xdotool libnotify

# Fedora (Wayland)
sudo dnf install python3-virtualenv portaudio-devel wl-clipboard ydotool libnotify
```

**Note**: The script auto-detects X11 vs Wayland. For Wayland, `ydotool` may require additional setup:
```bash
# Enable ydotool service (Wayland only)
sudo systemctl enable --now ydotool
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
1. Click in text field where you want transcription to appear
2. Press hotkey once → Recording starts (notification appears)
3. Speak into microphone
4. Press hotkey again → Recording stops, transcription begins
5. Wait for notifications:
   - **Local-only**: Transcribing → Pasted
   - **With polishing**: Transcribing → Polishing → Pasted
6. Transcription automatically appears in your text field!

**No terminal window, no manual pasting required!**

## Configuration

Edit `.env` to customize:

| Variable | Default | Options | Notes |
|----------|---------|---------|-------|
| `ENABLE_POLISHING` | `false` | `true`, `false` | Enable GPT polishing (requires API key) |
| `AUTO_PASTE` | `true` | `true`, `false` | Automatically paste transcription (requires xdotool/ydotool) |
| `PRE_RECORDING_BUFFER` | `2` | `0-10` (seconds) | Capture audio before hotkey press (0 = disabled) |
| `OPENAI_API_KEY` | - | Your API key | Only needed if polishing enabled |
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` | Affects accuracy and speed |
| `DEVICE` | `cpu` | `cpu`, `cuda` | Use `cuda` for GPU acceleration |
| `COMPUTE_TYPE` | `int8` | `int8`, `float16`, `float32` | Lower = faster, less accurate |
| `OPENAI_MODEL` | `gpt-4o-mini` | `gpt-4o-mini`, `gpt-4o` | Only used if polishing enabled |
| `IDLE_TIMEOUT` | `600` | `0` or seconds | Model stays in VRAM (0 = never timeout) |

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

For **3-5x faster** transcription with NVIDIA GPU:

### Prerequisites

- NVIDIA GPU with CUDA support (GTX 1060+, RTX series, etc.)
- NVIDIA drivers installed (check with `nvidia-smi`)
- CUDA 12.x installed (comes with driver)

### Ubuntu 24.04 Setup

**1. Install cuDNN libraries:**

```bash
# Ubuntu 24.04 includes cuDNN 8 in official repos
sudo apt install nvidia-cudnn

# Verify installation
ldconfig -p | grep cudnn
```

**2. Install compatible CTranslate2 version:**

The default `faster-whisper` installs CTranslate2 4.6.0+ which requires cuDNN 9.
Ubuntu 24.04 ships with cuDNN 8, so we need to downgrade CTranslate2:

```bash
source venv/bin/activate

# Downgrade to CTranslate2 4.4.0 (compatible with cuDNN 8)
pip install --force-reinstall ctranslate2==4.4.0
```

**3. Update .env for GPU acceleration:**

```bash
# Edit .env file
DEVICE=cuda
COMPUTE_TYPE=float16  # Best for CUDA (int8 is for CPU only)
```

**4. Verify CUDA is working:**

```bash
source venv/bin/activate

python3 -c "
from faster_whisper import WhisperModel
import os
os.environ['DEVICE'] = 'cuda'
print('Loading model on CUDA...')
model = WhisperModel('small', device='cuda', compute_type='float16')
print('✅ CUDA working! GPU acceleration enabled.')
"
```

Check GPU memory usage while running:
```bash
watch -n 1 nvidia-smi
```

You should see the model loaded in GPU memory (~600MB for `small` model).

### Other Ubuntu Versions

**Ubuntu 22.04 and earlier:**
- Install CUDA Toolkit from NVIDIA website
- May need to manually install cuDNN from NVIDIA developer site

**For newer cuDNN 9+ (if available):**
```bash
# If your system has cuDNN 9, you can use latest CTranslate2
pip install ctranslate2  # Will auto-install latest version
```

### Performance Gains

With CUDA enabled on a GTX 1660:

| Model | CPU Time (Intel i5) | GPU Time | Speedup |
|-------|---------------------|----------|---------|
| tiny  | ~0.5s | ~0.2s | 2.5x |
| small | ~2s | ~0.5s | 4x |
| medium | ~8s | ~2s | 4x |
| large-v3 | ~20s | ~5s | 4x |

### Troubleshooting CUDA

**Error: "Unable to load libcudnn_ops.so.9"**
- You have CTranslate2 4.6.0+ but cuDNN 8 installed
- Solution: Downgrade to CTranslate2 4.4.0 (see step 2 above)

**Error: "CUDA out of memory"**
- Your GPU doesn't have enough VRAM for the model
- Solution: Use smaller model (`small` instead of `medium`) or switch to CPU

**Verify cuDNN version:**
```bash
# Check installed cuDNN version
apt-cache policy nvidia-cudnn

# Check what CTranslate2 is looking for
ldconfig -p | grep cudnn
```

**GPU not being used:**
```bash
# Check if CUDA device is detected
python3 -c "import ctranslate2; print(f'CUDA devices: {ctranslate2.get_cuda_device_count()}')"

# Should output: "CUDA devices: 1" (or more)
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
