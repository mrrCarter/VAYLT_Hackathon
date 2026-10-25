"""ENGRAM Assist hand-signal rules (engram/vision/assist_signals.py). No camera, network or clock."""

import json

import pytest

from engram.vision.assist_signals import (
    DEFAULT_MAP, NO_SIGNAL, ApproveGuard, Decider, MapError, Reading, load_map, no_signal, parse_map,
    reading_from, signal_from_hands,
)
from engram.vision.gestures import Hand

DIGEST = "ab" * 32
BRAIN_NOW = 1_800_000_000.0  # the brain's clock in the approve tests


def default_map(**changes):
    data = json.loads(DEFAULT_MAP.read_text())
    data.update(changes)
    return parse_map(data)


def play(decider, frames, start=0.0, step=0.25, busy=False):
    """Feed frames (signal, or (signal, probability)) one step apart. Returns [(time, decision)]."""
    out = []
    for i, frame in enumerate(frames):
        signal, p = (frame, 0.9) if isinstance(frame, str) else frame
        now = start + i * step
        d = decider.decide(no_signal("failed") if signal == "FAIL" else Reading(signal, p), now, busy=busy)
        if d is not None:
            out.append((now, d))
    return out


def brief(decisions):
    return [(t, d.signal, d.action.kind, d.go) for t, d in decisions]


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------
def test_default_map_is_the_agreed_mapping():
    m = load_map()
    kinds = {name: (a.kind, a.text, a.say) for name, a in m.actions.items()}
    assert kinds == {
        "open_palm": ("stop", "", ""),
        "thumbs_up": ("approve", "", ""),
        "thumbs_down": ("cancel", "", "Okay, cancelled"),
        "peace": ("command", "Bring me my pills", ""),
        "fist": ("command", "Pick it up", ""),
        "pointing": ("none", "", ""),
        "waving": ("none", "", ""),
    }
    assert (m.hold_frames, m.hold_seconds, m.min_probability, m.cooldown_s) == (2, 0.6, 0.6, 3.0)


def test_the_map_comes_from_the_json_file(tmp_path):
    path = tmp_path / "verna.json"
    path.write_text(json.dumps({"signals": {
        "waving": {"action": "stop", "say": "Stopping"},
        "fist": {"action": "approve"},
        "peace": {"action": "command", "text": "Bring me water"},
    }, "hold_frames": 3, "cooldown_s": 1.5}))
    m = load_map(path)
    assert m.action("waving").kind == "stop" and m.action("waving").say == "Stopping"
    assert m.action("fist").kind == "approve"
    assert m.action("peace").text == "Bring me water"
    assert m.action("open_palm").kind == "none"         # left out: does nothing
    assert m.action("thumbs_up").kind == "none"
    assert (m.hold_frames, m.cooldown_s, m.hold_seconds) == (3, 1.5, 0.6)
    (_, d), = play(Decider(m), ["waving"])               # the map's stop signal stops at once
    assert d.action.kind == "stop" and d.go


BAD_MAPS = {
    "not_an_object": [],
    "no_signals": {"hold_frames": 2},
    "unknown_signal": {"signals": {"open_palm": {"action": "stop"}, "thumbs up": {"action": "approve"}}},
    "unknown_action": {"signals": {"open_palm": {"action": "stop"}, "fist": {"action": "grab"}}},
    "command_without_text": {"signals": {"open_palm": {"action": "stop"}, "fist": {"action": "command"}}},
    "text_on_a_non_command": {"signals": {"open_palm": {"action": "stop", "text": "Stop"}}},
    "none_that_speaks": {"signals": {"open_palm": {"action": "stop"}, "waving": {"action": "none", "say": "Hi"}}},
    "nothing_stops": {"signals": {"thumbs_up": {"action": "approve"}}},
    "typo_in_a_key": {"signals": {"open_palm": {"acton": "stop"}}},
    "unknown_top_key": {"signals": {"open_palm": {"action": "stop"}}, "cooldown": 3},
    "one_frame_hold": {"signals": {"open_palm": {"action": "stop"}}, "hold_frames": 1},
    "low_threshold": {"signals": {"open_palm": {"action": "stop"}}, "min_probability": 0.3},
    "negative_cooldown": {"signals": {"open_palm": {"action": "stop"}}, "cooldown_s": -1},
    "empty_command": {"signals": {"open_palm": {"action": "stop"}, "fist": {"action": "command", "text": " "}}},
}


@pytest.mark.parametrize("name", BAD_MAPS)
def test_a_bad_map_is_refused(name):
    with pytest.raises(MapError):
        parse_map(BAD_MAPS[name])


def test_a_signal_written_twice_is_refused(tmp_path):
    path = tmp_path / "twice.json"
    path.write_text('{"signals": {"open_palm": {"action": "stop"}, "fist": {"action": "approve"},'
                    ' "fist": {"action": "command", "text": "Pick it up"}}}')
    with pytest.raises(MapError, match="twice"):
        load_map(path)


def test_a_missing_or_broken_map_file_is_refused(tmp_path):
    with pytest.raises(MapError):
        load_map(tmp_path / "missing.json")
    (tmp_path / "broken.json").write_text('{"signals": {')
    with pytest.raises(MapError, match="not valid JSON"):
        load_map(tmp_path / "broken.json")


# ---------------------------------------------------------------------------
# Classifier answers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("answer", [
    None, "thumbs_up", {"signal": "ok_sign", "probability": 0.9}, {"signal": None, "probability": 0.9},
    {"signal": "thumbs_up", "probability": float("nan")}, {"signal": "thumbs_up", "probability": 1.5},
    {"signal": "thumbs_up", "probability": -0.1}, {"signal": "thumbs_up", "probability": "0.9"},
    {"signal": "thumbs_up", "probability": True}, {"signal": "thumbs_up"},
])
def test_a_malformed_answer_is_no_signal(answer):
    reading = reading_from(answer)
    assert reading.signal == NO_SIGNAL and reading.problem


def test_a_good_answer_is_read_as_is():
    assert reading_from({"signal": "peace", "probability": 0.75, "confidence": 0.7}) == Reading("peace", 0.75)
    assert reading_from({"signal": "none", "probability": 0.99}) == Reading(NO_SIGNAL, 0.99)


# ---------------------------------------------------------------------------
# Rule 2: the hold
# ---------------------------------------------------------------------------
def test_the_hold_must_span_the_hold_time():
    # Two frames 0.5 s apart are enough frames but not enough time; the third, at 0.6 s, is.
    got = play(Decider(default_map()), ["thumbs_up", "thumbs_up", "thumbs_up"], step=0.3)
    assert brief(got) == [(0.6, "thumbs_up", "approve", True)]
    assert play(Decider(default_map()), ["thumbs_up", "thumbs_up"], step=0.5) == []


def test_the_hold_needs_the_number_of_frames():
    # With no hold time, the count alone decides: 2 frames by default, 3 when the map says so.
    assert brief(play(Decider(default_map(hold_seconds=0)), ["fist", "fist"])) == [(0.25, "fist", "command", True)]
    assert brief(play(Decider(default_map(hold_seconds=0, hold_frames=3)), ["fist"] * 3)) == [
        (0.5, "fist", "command", True)]


def test_no_signal_starts_the_hold_over_and_is_never_held():
    got = play(Decider(default_map()), ["thumbs_up", "thumbs_up", ("none", 0.95)] + ["thumbs_up"] * 4)
    assert brief(got) == [(1.5, "thumbs_up", "approve", True)]   # counted again from 0.75 s
    assert play(Decider(default_map()), [("none", 0.95)] * 8) == []


def test_a_failed_classification_starts_the_hold_over():
    got = play(Decider(default_map()), ["peace", "peace", "FAIL"] + ["peace"] * 4)
    assert brief(got) == [(1.5, "peace", "command", True)]


def test_a_low_probability_starts_the_hold_over():
    got = play(Decider(default_map()), ["thumbs_up", "thumbs_up", ("thumbs_up", 0.5)] + ["thumbs_up"] * 4)
    assert brief(got) == [(1.5, "thumbs_up", "approve", True)]
    # Exactly the threshold counts.
    assert brief(play(Decider(default_map()), [("thumbs_up", 0.6)] * 4)) == [(0.75, "thumbs_up", "approve", True)]


def test_another_signal_starts_the_hold_over():
    got = play(Decider(default_map()), ["thumbs_up", "thumbs_up", "fist"] + ["thumbs_up"] * 4)
    assert brief(got) == [(1.5, "thumbs_up", "approve", True)]


def test_one_hold_makes_one_action():
    d = Decider(default_map())
    assert len(play(d, ["fist"] * 40)) == 1                       # 10 s held: still one "Pick it up"
    assert brief(play(d, ["none"] + ["fist"] * 4, start=10.0)) == [(11.0, "fist", "command", True)]


def test_a_signal_with_no_action_is_recognised_but_does_nothing():
    (_, d), = play(Decider(default_map()), ["pointing"] * 4)
    assert d.action.kind == "none" and not d.go


# ---------------------------------------------------------------------------
# Rule 3: the pause after an action, and a request still running
# ---------------------------------------------------------------------------
def test_the_pause_after_an_action_drops_other_signals():
    d = Decider(default_map())
    got = play(d, ["fist"] * 4 + ["none"] + ["peace"] * 16 + ["none"] + ["peace"] * 4)
    # fist at 0.75 s; peace held from 1.25 s is complete at 2.0 s, inside the pause (until 3.75 s):
    # dropped, and holding it on past the pause does not bring it back. Shown again, it works.
    assert brief(got) == [(0.75, "fist", "command", True), (2.0, "peace", "command", False),
                          (6.25, "peace", "command", True)]
    assert "pause" in got[1][1].why


def test_a_signal_is_dropped_while_the_last_request_runs():
    (_, d), = play(Decider(default_map()), ["fist"] * 4, busy=True)
    assert not d.go and "still running" in d.why


# ---------------------------------------------------------------------------
# Rule 1: STOP first
# ---------------------------------------------------------------------------
def test_one_confident_open_palm_frame_stops_at_once():
    (t, d), = play(Decider(default_map()), ["open_palm"])
    assert t == 0.0 and d.action.kind == "stop" and d.go and not d.repeat


def test_stop_is_never_held_back_by_the_pause_or_a_running_request():
    d = Decider(default_map())
    play(d, ["fist"] * 4)                                          # an action: the pause runs to 3.75 s
    (_, stop), = play(d, ["open_palm"], start=1.0)
    assert stop.action.kind == "stop" and stop.go
    (_, stop), = play(Decider(default_map()), ["open_palm"], busy=True)
    assert stop.action.kind == "stop" and stop.go


def test_an_unsure_open_palm_does_not_stop():
    assert play(Decider(default_map()), [("open_palm", 0.59)]) == []


def test_stop_is_sent_again_while_the_palm_stays_up():
    got = play(Decider(default_map()), ["open_palm", "open_palm", "open_palm", "none", "open_palm"])
    assert [(d.action.kind, d.go, d.repeat) for _, d in got] == [
        ("stop", True, False), ("stop", True, True), ("stop", True, True), ("stop", True, False)]


def test_a_stop_starts_the_pause_for_other_signals():
    got = play(Decider(default_map()), ["open_palm"] + ["fist"] * 4)
    assert brief(got) == [(0.0, "open_palm", "stop", True), (1.0, "fist", "command", False)]


def test_a_stop_breaks_a_hold():
    got = play(Decider(default_map(cooldown_s=0)), ["thumbs_up", "thumbs_up", "open_palm", "thumbs_up"])
    assert brief(got) == [(0.5, "open_palm", "stop", True)]


# ---------------------------------------------------------------------------
# Rules 4 and 5: what a thumbs-up may approve
# ---------------------------------------------------------------------------
def plan(op="ep_1", age=5.0, digest=DIGEST, **extra):
    return {"operation_id": op, "request": "Bring me my pills", "plan": {"goal": "x", "steps": []},
            "digest": digest, "source": "rules", "proposed_at": BRAIN_NOW - age, **extra}


def test_approves_the_one_recent_plan_with_its_own_digest():
    choice, why = ApproveGuard().check([plan()], BRAIN_NOW, now=0.0)
    assert why == "" and (choice.operation_id, choice.digest, choice.request, choice.age_s) == (
        "ep_1", DIGEST, "Bring me my pills", 5.0)


def test_no_plan_waiting_approves_nothing():
    assert ApproveGuard().check([], BRAIN_NOW, 0.0) == (None, "no plan is waiting for approval")


def test_two_plans_waiting_approve_nothing():
    choice, why = ApproveGuard().check([plan("ep_1"), plan("ep_2", age=1.0)], BRAIN_NOW, 0.0)
    assert choice is None and "2 plans" in why


def test_a_stale_plan_is_not_approved():
    guard = ApproveGuard()
    assert guard.check([plan(age=59.9)], BRAIN_NOW, 0.0)[0] is not None
    for age in (60.0, 61.0, 299.0):
        choice, why = guard.check([plan(age=age)], BRAIN_NOW, 0.0)
        assert choice is None and "old" in why


def test_a_plan_dated_in_the_brains_future_is_not_approved():
    choice, why = ApproveGuard().check([plan(age=-10.0)], BRAIN_NOW, 0.0)
    assert choice is None and "ahead" in why
    assert ApproveGuard().check([plan(age=-1.0)], BRAIN_NOW, 0.0)[0] is not None  # within the Date slack


def test_without_the_brains_clock_nothing_is_approved():
    choice, why = ApproveGuard().check([plan()], None, 0.0)
    assert choice is None and "Date" in why


@pytest.mark.parametrize("bad", [{"digest": "AB" * 32}, {"digest": "ab" * 31}, {"digest": None}, {"digest": 7},
                                 {"operation_id": "ep_1/../stop"}, {"operation_id": ""}, {"operation_id": None},
                                 {"proposed_at": None}, {"proposed_at": "yesterday"}, {"proposed_at": True}])
def test_a_plan_with_a_bad_field_is_not_approved(bad):
    assert ApproveGuard().check([{**plan(), **bad}], BRAIN_NOW, 0.0)[0] is None


def test_a_list_that_is_not_a_list_approves_nothing():
    assert ApproveGuard().check({"operation_id": "ep_1"}, BRAIN_NOW, 0.0)[0] is None
    assert ApproveGuard().check(["ep_1"], BRAIN_NOW, 0.0)[0] is None


def test_a_cancelled_plan_is_never_approved():
    guard = ApproveGuard()
    assert guard.cancel([plan("ep_1")], now=0.0) == ["ep_1"]
    choice, why = guard.check([plan("ep_1")], BRAIN_NOW, 1.0)
    assert choice is None and "cancelled" in why
    assert guard.check([plan("ep_2")], BRAIN_NOW, 1.0)[0] is not None   # a new plan still can be


def test_a_cancel_that_could_not_read_the_plans_turns_approvals_off_until_they_are_too_old():
    guard = ApproveGuard()
    assert guard.cancel(None, now=100.0) == []
    choice, why = guard.check([plan(age=1.0)], BRAIN_NOW, now=161.9)
    assert choice is None and "off" in why
    assert guard.check([plan(age=1.0)], BRAIN_NOW, now=162.0)[0] is not None
    guard2 = ApproveGuard()
    guard2.cancel({"error": "not a list"}, now=0.0)
    assert guard2.check([plan(age=1.0)], BRAIN_NOW, now=10.0)[0] is None


# ---------------------------------------------------------------------------
# MediaPipe hands -> signal names
# ---------------------------------------------------------------------------
def hand(builtin="None", score=0.0, label="other"):
    return Hand([], label, builtin, score)


@pytest.mark.parametrize("builtin, signal", [("Open_Palm", "open_palm"), ("Thumb_Up", "thumbs_up"),
                                             ("Thumb_Down", "thumbs_down"), ("Closed_Fist", "fist"),
                                             ("Victory", "peace"), ("Pointing_Up", "pointing"), ("ILoveYou", "none")])
def test_mediapipe_gestures_map_to_signal_names(builtin, signal):
    answer = signal_from_hands([hand(builtin, 0.8)])
    assert answer["signal"] == signal
    assert reading_from(answer).signal == signal


def test_mediapipe_shapes_without_a_label():
    assert signal_from_hands([hand(label="open_palm")])["signal"] == "open_palm"   # the shape that stops today
    assert signal_from_hands([hand(label="pointing")])["probability"] == 1.0
    assert signal_from_hands([hand(label="thumb")])["signal"] == NO_SIGNAL        # up or down? can't tell
    assert signal_from_hands([])["signal"] == NO_SIGNAL


def test_mediapipe_an_open_palm_on_any_hand_wins():
    answer = signal_from_hands([hand("Thumb_Up", 0.95), hand("Open_Palm", 0.7)])
    assert (answer["signal"], answer["probability"]) == ("open_palm", 0.7)


def test_mediapipe_hands_that_disagree_are_no_signal():
    assert signal_from_hands([hand("Thumb_Up", 0.9), hand("Closed_Fist", 0.9)])["signal"] == NO_SIGNAL
    agree = signal_from_hands([hand("Thumb_Up", 0.7), hand("Thumb_Up", 0.9)])
    assert (agree["signal"], agree["probability"]) == ("thumbs_up", 0.9)
