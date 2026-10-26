"""End-to-end check of the apple-streaming path against the real
SFSpeechRecognizer, fed from `say`-generated audio instead of the mic.

No mic, speakers, clipboard or paste involved. Needs macOS with Speech
Recognition already granted to the terminal; skips otherwise.

    venv/bin/python test_sfs_live.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Keep the module's log/status files away from a live daemon's.
_RT = tempfile.mkdtemp(prefix="whisper-hotkey-test-")
os.environ["TMPDIR"] = _RT + "/"

DAEMON = os.getenv("DAEMON_PATH", str(Path(__file__).parent / "transcribe-daemon.py"))


def _load_daemon_module():
    spec = importlib.util.spec_from_file_location("transcribe_daemon_live", DAEMON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pump(seconds):
    import CoreFoundation as CF
    CF.CFRunLoopRunInMode(CF.kCFRunLoopDefaultMode, seconds, False)


def _speech_file(text):
    path = os.path.join(_RT, f"speech-{abs(hash(text))}.aiff")
    subprocess.run(["say", "-o", path, text], check=True)
    return path


def _transcribe(mod, audio_path):
    """Stream audio_path through a recorder in real time, then stop it.
    Returns (text, seconds spent in stop after the trailing-speech delay)."""
    from AVFoundation import AVAudioFile, AVAudioPCMBuffer
    from Foundation import NSURL
    from Speech import SFSpeechRecognizer

    rec = mod.AudioRecorderDaemon()
    rec._sfs_recognizer = SFSpeechRecognizer.alloc().init()
    rec._sfs_handler = lambda r, e: rec._sfs_handle(r, e)
    rec.is_recording = True
    rec._start_next_sfs_segment()

    f, err = AVAudioFile.alloc().initForReading_error_(NSURL.fileURLWithPath_(audio_path), None)
    assert f is not None, err
    fmt = f.processingFormat()
    while True:
        buf = AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(fmt, 1024)
        ok, _ = f.readIntoBuffer_error_(buf, None)
        if not ok or buf.frameLength() == 0:
            break
        if rec._sfs_request is not None:
            rec._sfs_request.appendAudioPCMBuffer_(buf)
        _pump(buf.frameLength() / fmt.sampleRate())

    t0 = time.monotonic()
    text = rec._stop_recording_apple_streaming() or ""
    stop_s = time.monotonic() - t0 - min(mod.TRAILING_SPEECH_DELAY, 0.3)
    return text, stop_s


def _words(s):
    return s.lower().replace(".", "").replace(",", "").split()


def test_single_utterance_keeps_last_word(mod):
    text, stop_s = _transcribe(mod, _speech_file("Please schedule the review for Thursday afternoon [[slnc 300]]"))
    print(f"  single: {text!r} (stop {stop_s * 1000:.0f}ms)")
    assert _words(text)[-1] == "afternoon", text
    return stop_s


def test_pause_chains_segments_without_duplicates(mod):
    text, stop_s = _transcribe(mod, _speech_file(
        "The first part of the message is here. [[slnc 2500]] "
        "And the second part ends with pineapple. [[slnc 300]]"))
    print(f"  paused: {text!r} (stop {stop_s * 1000:.0f}ms)")
    w = _words(text)
    assert "first" in w and w[-1] == "pineapple", text
    assert w.count("first") == 1 and w.count("second") == 1, f"duplicated: {text!r}"
    return stop_s


def main():
    if sys.platform != "darwin":
        print("SKIP: macOS only")
        return
    from Speech import SFSpeechRecognizer
    if SFSpeechRecognizer.authorizationStatus() != 3:  # Authorized
        print("SKIP: Speech Recognition not authorized for this terminal")
        return
    from AppKit import NSApplication
    NSApplication.sharedApplication()
    mod = _load_daemon_module()
    print(f"daemon: {DAEMON}")
    stops = [test_single_utterance_keeps_last_word(mod),
             test_pause_chains_segments_without_duplicates(mod)]
    print(f"All live tests passed (mean stop {sum(stops) / len(stops) * 1000:.0f}ms)")


if __name__ == "__main__":
    main()
