"""Test the SFSpeechRecognizer event handler logic for utterance
restart detection. Reproduces the duplicate-words bug observed in
production logs (2026-05-11 16:21:24 session).
"""
import importlib.util
import sys
from pathlib import Path


def _load_daemon_module():
    spec = importlib.util.spec_from_file_location(
        "transcribe_daemon", Path(__file__).parent / "transcribe-daemon.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["transcribe_daemon"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeTranscription:
    def __init__(self, text):
        self._t = text

    def formattedString(self):
        return self._t


class FakeResult:
    def __init__(self, text, is_final=False):
        self._b = FakeTranscription(text)
        self._final = is_final

    def bestTranscription(self):
        return self._b

    def isFinal(self):
        return self._final


def feed(rec, events):
    for text, is_final in events:
        rec._sfs_handle(FakeResult(text, is_final), None)


def assemble(rec):
    pieces = list(rec._sfs_segments)
    if rec._sfs_current_partial and (not pieces or pieces[-1] != rec._sfs_current_partial):
        pieces.append(rec._sfs_current_partial)
    return " ".join(pieces).strip()


def test_refinement_not_promoted_to_segments():
    """Apple refining the same utterance must not create duplicate segments.

    Real log sequence from 2026-05-11 16:21:24 — Apple converged on
    'Have the tester run these in integration now' but went through
    intermediate hypotheses that the head_changed heuristic
    misinterpreted as fresh utterances.
    """
    mod = _load_daemon_module()
    rec = mod.AudioRecorderDaemon.__new__(mod.AudioRecorderDaemon)
    rec._sfs_segments = []
    rec._sfs_current_partial = ""
    rec._sfs_stopped = False
    rec._sfs_recognizer = None  # _start_next_sfs_segment will early-return
    rec.live_text = ""

    events = [
        ("Have", False),
        ("Have the test", False),
        ("Have the test tester", False),
        ("I have the tester", False),
        ("Have the tester run these in integration now", False),
        ("Have the tester run these in integration now", True),
        ("I'm not sure if the flag is on or off", True),
    ]
    feed(rec, events)
    final = assemble(rec)
    print("FINAL:", repr(final))
    assert "Have the test tester" not in final, f"intermediate refinement leaked: {final!r}"
    assert "I have the tester " not in final, f"intermediate refinement leaked: {final!r}"
    assert final == "Have the tester run these in integration now I'm not sure if the flag is on or off", final


def test_real_utterance_restart_still_caught():
    """Don't regress the 87eefe7 case: silent restart after speech pause.

    Apple shrinks the partial drastically when it silently begins a
    new utterance. We must promote the prior partial so it isn't lost.
    """
    mod = _load_daemon_module()
    rec = mod.AudioRecorderDaemon.__new__(mod.AudioRecorderDaemon)
    rec._sfs_segments = []
    rec._sfs_current_partial = ""
    rec._sfs_stopped = False
    rec._sfs_recognizer = None
    rec.live_text = ""

    events = [
        ("The first phrase goes here now", False),
        ("The", False),
        ("The second phrase comes after silence", False),
        ("The second phrase comes after silence", True),
    ]
    feed(rec, events)
    final = assemble(rec)
    print("FINAL:", repr(final))
    assert final == "The first phrase goes here now The second phrase comes after silence", final


def test_unrelated_silent_restart_caught():
    """Head changes AND words are mostly disjoint: still treat as restart."""
    mod = _load_daemon_module()
    rec = mod.AudioRecorderDaemon.__new__(mod.AudioRecorderDaemon)
    rec._sfs_segments = []
    rec._sfs_current_partial = ""
    rec._sfs_stopped = False
    rec._sfs_recognizer = None
    rec.live_text = ""

    events = [
        ("The quick brown fox jumps", False),
        ("Hello world today is great", False),
        ("Hello world today is great", True),
    ]
    feed(rec, events)
    final = assemble(rec)
    print("FINAL:", repr(final))
    assert "The quick brown fox jumps" in final, final
    assert "Hello world today is great" in final, final


if __name__ == "__main__":
    test_refinement_not_promoted_to_segments()
    test_real_utterance_restart_still_caught()
    test_unrelated_silent_restart_caught()
    print("All tests passed")
