#!/bin/bash
# Wrapper script for hotkey binding
# Press once to start recording, press again to stop and transcribe

# Logging
LOG_FILE="/tmp/whisper-hotkey.log"
echo "=== $(date) ===" >> "$LOG_FILE"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
DAEMON_SCRIPT="$PROJECT_DIR/transcribe-daemon.py"

echo "PROJECT_DIR: $PROJECT_DIR" >> "$LOG_FILE"
echo "VENV_PYTHON: $VENV_PYTHON" >> "$LOG_FILE"
echo "DAEMON_SCRIPT: $DAEMON_SCRIPT" >> "$LOG_FILE"

# Run daemon in toggle mode (don't background - runs in foreground or sends signal)
"$VENV_PYTHON" "$DAEMON_SCRIPT" >> "$LOG_FILE" 2>&1 &
DAEMON_PID=$!

echo "Daemon started/signaled with PID: $DAEMON_PID" >> "$LOG_FILE"

# Detach from parent process so GNOME doesn't kill it
disown

exit 0
