#!/bin/bash
# Switch between CPU and CUDA mode for whisper-hotkey

ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found"
    exit 1
fi

# Check current mode
CURRENT_DEVICE=$(grep "^DEVICE=" "$ENV_FILE" | cut -d= -f2)

if [ "$CURRENT_DEVICE" = "cuda" ]; then
    # Switch to CPU
    sed -i 's/^DEVICE=cuda$/DEVICE=cpu/' "$ENV_FILE"
    sed -i 's/^COMPUTE_TYPE=float16$/COMPUTE_TYPE=int8/' "$ENV_FILE"
    echo "✅ Switched to CPU mode"
    echo "   (Restart daemon: pkill -f transcribe-daemon)"
elif [ "$CURRENT_DEVICE" = "cpu" ]; then
    # Switch to CUDA
    sed -i 's/^DEVICE=cpu$/DEVICE=cuda/' "$ENV_FILE"
    sed -i 's/^COMPUTE_TYPE=int8$/COMPUTE_TYPE=float16/' "$ENV_FILE"
    echo "✅ Switched to CUDA mode"
    echo "   (Restart daemon: pkill -f transcribe-daemon)"
else
    echo "Error: Could not determine current device mode"
    exit 1
fi

# Kill daemon so it picks up new settings on next use
pkill -f transcribe-daemon 2>/dev/null && echo "   Daemon killed, will restart with new settings on next hotkey press"
