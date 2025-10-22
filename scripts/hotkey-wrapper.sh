#!/bin/bash
# Wrapper script for hotkey binding
# Press once to start recording, press again to stop and transcribe

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
DAEMON_SCRIPT="$PROJECT_DIR/transcribe-daemon.py"

# Run daemon in toggle mode
"$VENV_PYTHON" "$DAEMON_SCRIPT" &

exit 0
