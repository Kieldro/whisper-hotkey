#!/bin/bash
# Tests scripts/update.sh against throwaway git repos (no network, no live daemon).
#   bash test_update.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T=$(mktemp -d "${TMPDIR:-/tmp}/whisper-update-test.XXXXXX")
trap 'kill "$HUP_PID" 2>/dev/null; rm -rf "$T"' EXIT
export TMPDIR="$T/rt/" XDG_RUNTIME_DIR="$T/rt" AUTO_UPDATE_INTERVAL=0 WHISPER_HOTKEY_DIR="$T/install"
mkdir -p "$T/rt"
fails=0
check() { if eval "$2"; then echo "  ok: $1"; else echo "  FAIL: $1"; fails=$((fails + 1)); fi; }

git init -q --bare "$T/origin.git"
git clone -q "$T/origin.git" "$T/dev" 2>/dev/null
cd "$T/dev" && git checkout -q -b main
mkdir scripts && cp "$HERE/scripts/update.sh" scripts/
echo "x = 1" > transcribe-daemon.py
git add -A && git -c user.name=t -c user.email=t@t commit -qm init && git push -q origin main
git clone -q -b main "$T/origin.git" "$T/install"

# Stand-in daemon: records each SIGHUP it receives.
bash -c "trap 'echo hup >> $T/hups' HUP; while :; do sleep 0.1; done" &
HUP_PID=$!
sleep 0.2
run() { bash "$T/install/scripts/update.sh" "$HUP_PID" >> "$T/log" 2>&1; sleep 1; }  # the stand-in handles HUP between its sleeps
push() { (cd "$T/dev" && echo "$1" > transcribe-daemon.py && git -c user.name=t -c user.email=t@t commit -qam "$2" && git push -q origin main); }
head_of() { git -C "$T/$1" rev-parse HEAD; }
hups() { [ -f "$T/hups" ] && wc -l < "$T/hups" | tr -d ' ' || echo 0; }

echo "up to date"
run
check "no restart requested" '[ "$(hups)" = 0 ]'

echo "new commit"
push "x = 2" "second"
run
check "fast-forwarded" '[ "$(head_of install)" = "$(head_of dev)" ]'
check "restart requested once" '[ "$(hups)" = 1 ]'
check "logged" 'grep -q "\[update\] updated main" "$T/log"'

echo "throttled"
push "x = 3" "third"
AUTO_UPDATE_INTERVAL=3600 run
check "skipped within the interval" '[ "$(head_of install)" != "$(head_of dev)" ]'
rm -f "$T/rt/whisper-hotkey-update-check"

echo "local edits"
echo "x = 99" > "$T/install/transcribe-daemon.py"
run
check "left alone" '[ "$(cat "$T/install/transcribe-daemon.py")" = "x = 99" ] && [ "$(hups)" = 1 ]'
git -C "$T/install" checkout -q transcribe-daemon.py
run
check "updates once clean" '[ "$(head_of install)" = "$(head_of dev)" ] && [ "$(hups)" = 2 ]'

echo "broken update"
good=$(head_of install)
push "import no_such_module" "broken"
run
check "rolled back" '[ "$(head_of install)" = "$good" ] && grep -q "rolling back" "$T/log"'
check "no restart for a broken update" '[ "$(hups)" = 2 ]'
check "tree clean after rollback" '[ -z "$(git -C "$T/install" status --porcelain --untracked-files=no)" ]'

echo "diverged"
push "x = 4" "fixed"
(cd "$T/install" && echo "y = 1" > local.py && git add local.py && git -c user.name=t -c user.email=t@t commit -qm local)
local_head=$(head_of install)
run
check "local commits kept" '[ "$(head_of install)" = "$local_head" ] && grep -q "diverged" "$T/log"'

echo "--- update log ---"; cat "$T/log"
[ "$fails" = 0 ] && echo "All update tests passed" || { echo "$fails update test(s) failed"; exit 1; }
