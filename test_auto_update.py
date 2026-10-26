"""Daemon side of auto-update: SIGHUP from scripts/update.sh makes the daemon
exit once it is idle, never mid-recording. Runs main() with a fake recorder in
a child process with its own runtime dir; no mic, model, overlay or paste.

    venv/bin/python test_auto_update.py
"""
import atexit
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DAEMON = os.getenv("DAEMON_PATH", str(Path(__file__).parent / "transcribe-daemon.py"))


def child(auto_update):
    spec = importlib.util.spec_from_file_location("transcribe_daemon_update", DAEMON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class FakeRecorder:
        is_recording = False
        recording_start_time = None

    class FakeDaemon:
        idle_timeout = 0

        def __init__(self):
            self.recorder = FakeRecorder()
            self.last_activity_time = time.time()

        def start_recording(self):
            self.recorder.is_recording = True
            return True

        def stop_and_process(self):
            self.recorder.is_recording = False
            mod.logger.info("fake: stopped")

        def is_idle_timeout_exceeded(self):
            return False

        def cleanup(self):
            pass

    def fake_updater():
        mod.logger.info("fake: updater spawned")
        os.kill(os.getpid(), signal.SIGHUP)

    mod.TranscriptionDaemon = FakeDaemon
    mod.spawn_updater = fake_updater
    mod.AUTO_UPDATE = auto_update
    mod.ENGINE = "apple-streaming"  # no loading chime
    mod.ENABLE_OVERLAY = False
    mod.ENABLE_NOTIFICATIONS = False
    mod.main()


def run_case(auto_update):
    rt = tempfile.mkdtemp(prefix="whisper-hotkey-test-")
    atexit.register(shutil.rmtree, rt, ignore_errors=True)
    env = dict(os.environ, TMPDIR=rt + "/", XDG_RUNTIME_DIR=rt)
    proc = subprocess.Popen([sys.executable, __file__, "child", str(auto_update)], env=env)
    log = Path(rt, "whisper-hotkey.log")

    def wait_for(text, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if log.exists() and text in log.read_text():
                return True
            time.sleep(0.05)
        return False

    assert wait_for("Daemon loop started"), "daemon never started"
    time.sleep(1.5)
    assert proc.poll() is None, "exited mid-recording"
    proc.send_signal(signal.SIGUSR1)  # finish the dictation
    assert wait_for("fake: stopped"), "stop not handled"
    try:
        proc.wait(timeout=3)
        exited = True
    except subprocess.TimeoutExpired:
        exited = False
    text = log.read_text()
    if not exited:
        proc.terminate()
        proc.wait(timeout=5)
    return exited, text


def main():
    exited, text = run_case(True)
    assert exited, "did not exit after the update once idle"
    assert "Update installed" in text and text.count("fake: updater spawned") == 1, text
    print("  ok: waits for the recording, then exits to load the update")

    exited, text = run_case(False)
    assert not exited and "fake: updater spawned" not in text, text
    print("  ok: AUTO_UPDATE=false never checks")
    print("All auto-update tests passed")


if __name__ == "__main__":
    if sys.argv[1:2] == ["child"]:
        child(sys.argv[2] == "True")
    else:
        main()
