#!/bin/bash
# Wrapper script for hotkey binding
# Press once to start recording, press again to stop and transcribe
#
# Fast path: when the daemon is already running, signal it directly from
# bash (~5ms) instead of spawning a Python interpreter (~120ms).

RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"
LOCK_FILE="$RUNTIME_DIR/whisper-hotkey-daemon.lock"
STATE_FILE="$RUNTIME_DIR/whisper-hotkey-state"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
DAEMON_SCRIPT="$PROJECT_DIR/transcribe-daemon.py"

# Fast path: if daemon is alive, signal it directly
DAEMON_PID=$(cat "$LOCK_FILE" 2>/dev/null)
if [ -n "$DAEMON_PID" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
    if [ -f "$STATE_FILE" ]; then
        # Currently recording → stop
        kill -USR1 "$DAEMON_PID"
        rm -f "$STATE_FILE"
    else
        # Idle → start recording
        touch "$STATE_FILE"
        kill -USR2 "$DAEMON_PID"
    fi
    exit 0
fi

# Cold start: no daemon running, launch Python to load model
"$VENV_PYTHON" "$DAEMON_SCRIPT" 2>/dev/null &
disown
exit 0
