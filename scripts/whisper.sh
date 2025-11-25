#!/bin/bash
# Fast hotkey script - sends signals directly without starting Python
# Falls back to full daemon script if no daemon is running

LOCK_FILE="/tmp/whisper-hotkey-daemon.lock"
STATE_FILE="/tmp/whisper-hotkey-state"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check if daemon is running
if [ -f "$LOCK_FILE" ]; then
    DAEMON_PID=$(cat "$LOCK_FILE" 2>/dev/null)
    if [ -n "$DAEMON_PID" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
        # Daemon is alive - send signal directly
        if [ -f "$STATE_FILE" ]; then
            # Currently recording -> stop
            kill -USR1 "$DAEMON_PID"
            rm -f "$STATE_FILE"
        else
            # Not recording -> start
            touch "$STATE_FILE"
            kill -USR2 "$DAEMON_PID"
        fi
        exit 0
    fi
fi

# No daemon running - start one (this is the slow path, only happens once)
exec "$PROJECT_DIR/venv/bin/python" "$PROJECT_DIR/transcribe-daemon.py" &
disown

