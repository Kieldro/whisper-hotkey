#!/bin/bash
# Wrapper script for hotkey binding
# Press once to start recording, press again to stop and transcribe
#
# Fast path: when the daemon is already running, signal it directly from
# bash (~5ms) instead of spawning a Python interpreter (~120ms).

# Must match RUNTIME_DIR in transcribe-daemon.py.
case "$OSTYPE" in
    darwin*) RUNTIME_DIR="${TMPDIR:-/tmp}" ;;
    *)       RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" ;;
esac
RUNTIME_DIR="${RUNTIME_DIR%/}"  # strip trailing slash
LOCK_FILE="$RUNTIME_DIR/whisper-hotkey-daemon.lock"
STATE_FILE="$RUNTIME_DIR/whisper-hotkey-state"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
DAEMON_SCRIPT="$PROJECT_DIR/transcribe-daemon.py"

# Fast path: signal the daemon directly. Check the PID really is the daemon:
# after a hard crash the lock keeps a stale PID that another app may now own,
# and SIGUSR1/2 would kill it.
{ read -r DAEMON_PID < "$LOCK_FILE"; } 2>/dev/null
case "$(ps -p "${DAEMON_PID:-0}" -o args= 2>/dev/null)" in
    *transcribe-daemon.py*)
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
        ;;
esac

# Cold start: no daemon running, launch Python to load model.
# Append stderr to the log file so uncaught Python exceptions during
# model load are visible (otherwise the daemon can die silently).
LOG_FILE="$RUNTIME_DIR/whisper-hotkey.log"
"$VENV_PYTHON" "$DAEMON_SCRIPT" >>"$LOG_FILE" 2>>"$LOG_FILE" &
disown
exit 0
