#!/bin/bash
# Setup hotkey for voice transcription
# Supports i3, sway, KDE, GNOME

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER_SCRIPT="$PROJECT_DIR/scripts/hotkey-wrapper.sh"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
OS_TYPE=$(uname -s)

echo "🔧 Voice Transcription Hotkey Setup"
echo "===================================="
echo ""

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found at $VENV_PYTHON"
    echo "Run: python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# --- macOS: use Hammerspoon ---
if [ "$OS_TYPE" = "Darwin" ]; then
    HS_CONFIG="$HOME/.hammerspoon/init.lua"
    mkdir -p "$(dirname "$HS_CONFIG")"

    echo "📝 macOS setup using Hammerspoon:"
    echo "Hotkey: Option+Space (press once to start, again to stop)"
    echo ""

    if grep -q "whisper-hotkey" "$HS_CONFIG" 2>/dev/null; then
        echo "⚠️  Hotkey already exists in Hammerspoon config"
    else
        cat >> "$HS_CONFIG" << LUAEOF

-- Whisper Hotkey: Option+Space → toggle push-to-talk transcription
local wrapper = "$WRAPPER_SCRIPT"
hs.hotkey.bind({"alt"}, "space", function()
    hs.task.new(wrapper, nil):start()
end)
LUAEOF
        echo "✅ Added to $HS_CONFIG"
    fi

    if [ -d "/Applications/Hammerspoon.app" ]; then
        open /Applications/Hammerspoon.app
        echo "✅ Hammerspoon launched — reload config with Cmd+Shift+R in the Hammerspoon console"
    else
        echo "⚠️  Hammerspoon not installed. Run: brew install --cask hammerspoon"
    fi

    echo ""
    echo "⚠️  Grant Accessibility permissions: System Settings → Privacy & Security → Accessibility → add Hammerspoon"
    echo ""
    echo "✅ Setup complete!"
    exit 0
fi

# Detect desktop environment
detect_de() {
    if [ -n "$SWAYSOCK" ]; then
        echo "sway"
    elif [ "$XDG_CURRENT_DESKTOP" = "KDE" ]; then
        echo "kde"
    elif [ "$XDG_CURRENT_DESKTOP" = "GNOME" ]; then
        echo "gnome"
    elif pgrep -x "i3" > /dev/null; then
        echo "i3"
    else
        echo "unknown"
    fi
}

DE=$(detect_de)

echo "Detected desktop environment: $DE"
echo ""

# Setup based on desktop environment
case $DE in
    i3)
        CONFIG_FILE="$HOME/.config/i3/config"
        HOTKEY_LINE="bindsym \$mod+Shift+v exec $WRAPPER_SCRIPT"

        echo "📝 Adding hotkey to i3 config: $CONFIG_FILE"
        echo "Hotkey: Mod+Shift+V (daemon mode - press once to start, again to stop)"
        echo ""

        if grep -q "exec.*whisper-hotkey" "$CONFIG_FILE" 2>/dev/null; then
            echo "⚠️  Hotkey already exists in config"
        else
            echo "" >> "$CONFIG_FILE"
            echo "# Voice transcription hotkey (daemon mode)" >> "$CONFIG_FILE"
            echo "$HOTKEY_LINE" >> "$CONFIG_FILE"
            echo "✅ Added to config"
            echo "⚠️  Reload i3 with Mod+Shift+R"
        fi
        ;;

    sway)
        CONFIG_FILE="$HOME/.config/sway/config"
        HOTKEY_LINE="bindsym \$mod+Shift+v exec $WRAPPER_SCRIPT"

        echo "📝 Adding hotkey to sway config: $CONFIG_FILE"
        echo "Hotkey: Mod+Shift+V (daemon mode - press once to start, again to stop)"
        echo ""

        if grep -q "exec.*whisper-hotkey" "$CONFIG_FILE" 2>/dev/null; then
            echo "⚠️  Hotkey already exists in config"
        else
            echo "" >> "$CONFIG_FILE"
            echo "# Voice transcription hotkey (daemon mode)" >> "$CONFIG_FILE"
            echo "$HOTKEY_LINE" >> "$CONFIG_FILE"
            echo "✅ Added to config"
            echo "⚠️  Reload sway with Mod+Shift+C"
        fi
        ;;

    kde)
        echo "📝 KDE Plasma setup (manual steps required):"
        echo ""
        echo "1. Open System Settings"
        echo "2. Navigate to: Shortcuts → Custom Shortcuts"
        echo "3. Click: Edit → New → Global Shortcut → Command/URL"
        echo "4. Name: Voice Transcription"
        echo "5. Trigger: Set your preferred hotkey (e.g., Meta+Shift+V)"
        echo "6. Action: $WRAPPER_SCRIPT"
        echo ""
        echo "Usage: Press once to start recording, again to stop"
        ;;

    gnome)
        echo "📝 GNOME setup (manual steps required):"
        echo ""
        echo "1. Open Settings"
        echo "2. Navigate to: Keyboard → View and Customize Shortcuts → Custom Shortcuts"
        echo "3. Click: +"
        echo "4. Name: Voice Transcription"
        echo "5. Command: $WRAPPER_SCRIPT"
        echo "6. Set Shortcut: Click 'Set Shortcut' and press your desired key combo"
        echo ""
        echo "Usage: Press once to start recording, again to stop"
        ;;

    *)
        echo "❌ Unknown desktop environment"
        echo ""
        echo "Manual setup required. Add this command to your hotkey manager:"
        echo ""
        echo "  $WRAPPER_SCRIPT"
        echo ""
        echo "Usage: Press once to start recording, again to stop"
        ;;
esac

echo ""
echo "✅ Setup complete!"
