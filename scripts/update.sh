#!/bin/bash
# Fast-forward this checkout to its upstream, then SIGHUP the daemon so it
# restarts on the new code once idle. Run by the daemon; safe to run by hand.
# Usage: update.sh [daemon_pid]
set -u

REPO_DIR="${WHISPER_HOTKEY_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
case "$OSTYPE" in
    darwin*) RUNTIME_DIR="${TMPDIR:-/tmp}" ;;
    *)       RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" ;;
esac
STAMP="${RUNTIME_DIR%/}/whisper-hotkey-update-check"
INTERVAL_MIN=$(( ${AUTO_UPDATE_INTERVAL:-86400} / 60 ))
DAEMON_PID="${1:-}"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] [update] $*"; }

# The daemon restarts often (idle timeout), so throttle across restarts.
[ -n "$(find "$STAMP" -mmin -"$INTERVAL_MIN" 2>/dev/null)" ] && exit 0
touch "$STAMP"

cd "$REPO_DIR" || exit 1
branch=$(git symbolic-ref -q --short HEAD) || { log "detached HEAD; skipping"; exit 0; }
upstream=$(git rev-parse -q --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null) || upstream="origin/$branch"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    log "local changes in $REPO_DIR; skipping"
    exit 0
fi

GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" \
    git fetch -q "${upstream%%/*}" 2>&1 || { log "fetch failed; skipping"; exit 0; }
old=$(git rev-parse HEAD)
new=$(git rev-parse -q --verify "$upstream^{commit}") || { log "no $upstream; skipping"; exit 0; }
[ "$old" = "$new" ] && exit 0
if ! git merge-base --is-ancestor "$old" "$new"; then
    log "$branch has diverged from $upstream; skipping"
    exit 0
fi
git merge -q --ff-only "$new" || exit 1

# Import the new daemon before restarting onto it: a missing dependency would
# otherwise leave the hotkey dead until someone notices. Scratch runtime dir so
# the import doesn't touch the live daemon's log and status files.
py="$REPO_DIR/venv/bin/python"; [ -x "$py" ] || py=python3
scratch=$(mktemp -d "${RUNTIME_DIR%/}/whisper-hotkey-check.XXXXXX") || { git reset -q --keep "$old"; exit 1; }
if ! TMPDIR="$scratch/" XDG_RUNTIME_DIR="$scratch" "$py" -c '
import importlib.util, sys
spec = importlib.util.spec_from_file_location("daemon_check", sys.argv[1])
spec.loader.exec_module(importlib.util.module_from_spec(spec))
' "$REPO_DIR/transcribe-daemon.py" >/dev/null 2>"$scratch/err"; then
    log "${new:0:7} failed to import, rolling back to ${old:0:7}: $(tail -1 "$scratch/err")"
    git reset -q --keep "$old"
    rm -rf "$scratch"
    exit 1
fi
rm -rf "$scratch"

log "updated $branch ${old:0:7}..${new:0:7} ($(git rev-list --count "$old..$new") commits)"
[ -n "$DAEMON_PID" ] && kill -HUP "$DAEMON_PID" 2>/dev/null
exit 0
