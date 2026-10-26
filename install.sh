#!/bin/bash
# Whisper Hotkey installer
# Usage: git clone https://github.com/Kieldro/whisper-hotkey.git && cd whisper-hotkey && ./install.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
OS_TYPE=$(uname -s)  # "Linux" or "Darwin"

echo "Whisper Hotkey Installer"
echo "========================"
echo ""

# --- Detect session type ---
if [ "$OS_TYPE" = "Darwin" ]; then
    SESSION="macos"
elif [ "$XDG_SESSION_TYPE" = "wayland" ] || [ -n "$WAYLAND_DISPLAY" ]; then
    SESSION="wayland"
else
    SESSION="x11"
fi
echo "Session: $SESSION"

# --- Detect distro ---
if [ "$OS_TYPE" = "Darwin" ]; then
    PKG_MGR="brew"
elif command -v apt &>/dev/null; then
    PKG_MGR="apt"
elif command -v pacman &>/dev/null; then
    PKG_MGR="pacman"
elif command -v dnf &>/dev/null; then
    PKG_MGR="dnf"
else
    PKG_MGR="unknown"
fi
echo "Package manager: $PKG_MGR"
echo ""

# --- Install system dependencies ---
echo "Installing system dependencies..."

# python3-gi/GTK: the status overlay runs on the system python3, not the venv
COMMON_DEPS_APT="python3-venv python3-pip pulseaudio-utils libnotify-bin python3-gi gir1.2-gtk-3.0"
COMMON_DEPS_PACMAN="python python-pip portaudio libnotify python-gobject gtk3"
COMMON_DEPS_DNF="python3-virtualenv portaudio-devel libnotify python3-gobject gtk3"

if [ "$SESSION" = "wayland" ]; then
    SESSION_DEPS_APT="wl-clipboard ydotool"
    SESSION_DEPS_PACMAN="wl-clipboard ydotool"
    SESSION_DEPS_DNF="wl-clipboard ydotool"
elif [ "$SESSION" = "x11" ]; then
    SESSION_DEPS_APT="xclip xdotool"
    SESSION_DEPS_PACMAN="xclip xdotool"
    SESSION_DEPS_DNF="xclip xdotool"
fi

case $PKG_MGR in
    brew)
        # macOS: skhd for global hotkeys (pbcopy/osascript are built-in)
        if ! command -v brew &>/dev/null; then
            echo "❌ Homebrew not found. Install from https://brew.sh first."
            exit 1
        fi
        brew tap koekeishiya/formulae 2>/dev/null || true
        brew install skhd 2>/dev/null || true
        ;;
    apt)
        sudo apt update -qq
        sudo apt install -y $COMMON_DEPS_APT $SESSION_DEPS_APT
        ;;
    pacman)
        sudo pacman -S --needed --noconfirm $COMMON_DEPS_PACMAN $SESSION_DEPS_PACMAN
        ;;
    dnf)
        sudo dnf install -y $COMMON_DEPS_DNF $SESSION_DEPS_DNF
        ;;
    *)
        echo "Unknown package manager. Install these manually:"
        echo "  python3, pip, portaudio, pulseaudio-utils, libnotify"
        if [ "$SESSION" = "wayland" ]; then
            echo "  wl-clipboard, ydotool"
        elif [ "$SESSION" = "x11" ]; then
            echo "  xclip, xdotool"
        fi
        echo ""
        read -p "Press Enter once dependencies are installed..."
        ;;
esac

if [ "$SESSION" = "wayland" ]; then
    echo "Enabling ydotool service..."
    sudo systemctl enable --now ydotool 2>/dev/null || true
fi

echo ""

# --- Create virtual environment ---
echo "Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --- Detect engine ---
echo ""
echo "Choose transcription engine:"
echo "  1) Parakeet TDT (recommended - faster, better punctuation)"
echo "  2) faster-whisper (traditional, configurable model sizes)"
if [ "$OS_TYPE" = "Darwin" ]; then
    echo "  3) Apple on-device streaming (live transcript, no model download)"
fi
read -p "Choice [1]: " ENGINE_CHOICE
ENGINE_CHOICE="${ENGINE_CHOICE:-1}"

if [ "$ENGINE_CHOICE" = "2" ]; then
    ENGINE="whisper"
    echo "Installing faster-whisper..."
    pip install -q -r "$PROJECT_DIR/requirements.txt"
elif [ "$ENGINE_CHOICE" = "3" ] && [ "$OS_TYPE" = "Darwin" ]; then
    ENGINE="apple-streaming"
    echo "Installing Apple streaming dependencies..."
    pip install -q openai python-dotenv
else
    ENGINE="parakeet"
    echo "Installing Parakeet TDT..."
    pip install -q "onnx-asr[hub]" "onnxruntime!=1.21" openai python-dotenv "silero-vad>=5.1"
fi

# macOS pyobjc bridges: AVFoundation records audio, Quartz drives the CGEvent
# paste and shift-to-submit (osascript fallback is slower and has no
# shift-to-submit), Speech powers ENGINE=apple-streaming.
if [ "$OS_TYPE" = "Darwin" ]; then
    echo "Installing macOS audio and paste dependencies..."
    pip install -q numpy pyobjc-framework-Quartz pyobjc-framework-AVFoundation pyobjc-framework-Speech
fi

echo ""

# --- Create .env if it doesn't exist ---
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    if [ "$OS_TYPE" = "Darwin" ]; then
        sed -i '' "s/^ENGINE=.*/ENGINE=$ENGINE/" "$PROJECT_DIR/.env"
    else
        sed -i "s/^ENGINE=.*/ENGINE=$ENGINE/" "$PROJECT_DIR/.env"
    fi
    echo "Created .env from template (engine=$ENGINE)"
else
    echo ".env already exists, skipping"
fi

# --- GPU detection ---
# (faster-whisper's ctranslate2 has no Apple GPU backend; macOS stays on CPU.)
if command -v nvidia-smi &>/dev/null; then
    echo ""
    read -p "NVIDIA GPU detected. Enable CUDA acceleration? [Y/n]: " USE_CUDA
    USE_CUDA="${USE_CUDA:-Y}"
    if [[ "$USE_CUDA" =~ ^[Yy] ]]; then
        sed -i "s/^DEVICE=.*/DEVICE=cuda/" "$PROJECT_DIR/.env"
        sed -i "s/^COMPUTE_TYPE=.*/COMPUTE_TYPE=float16/" "$PROJECT_DIR/.env"
        echo "CUDA enabled"
    fi
fi

# --- Setup hotkey ---
echo ""
echo "Setting up hotkey..."
bash "$PROJECT_DIR/scripts/setup-hotkey.sh"

# --- Pre-download model ---
echo ""
echo "Downloading model (this may take a minute)..."
if [ "$ENGINE" = "parakeet" ]; then
    "$VENV_DIR/bin/python" -c "import onnx_asr; onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3'); print('Model downloaded')"
elif [ "$ENGINE" = "whisper" ]; then
    "$VENV_DIR/bin/python" -c "from faster_whisper import WhisperModel; WhisperModel('small'); print('Model downloaded')"
else
    echo "No model needed; macOS asks for Speech Recognition permission on first use"
fi

echo ""
echo "========================"
echo "Installation complete!"
echo ""
echo "Usage: Press your hotkey, speak, press again. Text appears where your cursor is."
echo ""
echo "Config: $PROJECT_DIR/.env"
if [ "$OS_TYPE" = "Darwin" ]; then
    echo "Logs:   \$TMPDIR/whisper-hotkey.log"
else
    echo "Logs:   \${XDG_RUNTIME_DIR:-/tmp}/whisper-hotkey.log"
fi
