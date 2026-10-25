"""The ENGRAM Assist loop (scripts/assist_gestures.py) against a fake brain, the real brain API on
the mock robot, and a fake classifier. No camera, no network, no intent model."""

import email.utils
import io
import json
import os
import subprocess
import sys
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import assist_gestures as ag  # noqa: E402

from engram.vision.assist_signals import Action, Decider, Decision, Reading, load_map  # noqa: E402

DIGEST = "cd" * 32
WHO = "patient (hand signal)"
PAUSE = ["none"] * 13   # 13 frames at --rate 4: past the 3 s pause after an action
JPEG = b"\xff\xd8"


class VisionError(Exception):
    """Stands in for engram/vision/intent_vision.py's error."""


def fake_classifier(jpeg: bytes) -> dict:
    """Frame bytes name the signal: b"fist" is a fist at 0.9, b"fist:0.4" at 0.4, b"error" fails."""
    text = jpeg.removeprefix(JPEG).decode()
    if text == "error":
        raise VisionError("the intent model did not answer")
    signal, _, p = text.partition(":")
    p = float(p) if p else 0.9
    return {"signal": signal, "probability": p, "confidence": p, "probs": {signal: p}, "latency_ms": 7}


def frames(tmp_path, signals):
    folder = tmp_path / "frames"
    folder.mkdir()
    for i, s in enumerate(signals):
        (folder / f"{i:03d}.jpg").write_bytes(s.encode())
    return folder


class FakeTime:
    def __init__(self):
        self.t = 1000.0

    def clock(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def brain_date():
    return {"date": email.utils.formatdate(time.time(), usegmt=True)}   # what uvicorn sends


def waiting_plan(op="ep_1", age=5.0, request="Bring me my pills", age_s=True):
    """A plan as GET /operations/pending lists it. age_s=False: as from a brain that sends no age_s."""
    proposed_at = time.time() - age
    plan = {"operation_id": op, "request": request, "plan": {"goal": request, "steps": []}, "digest": DIGEST,
            "source": "rules", "planner_note": "", "proposed_at": proposed_at}
    if age_s:
        plan["created_at"] = datetime.fromtimestamp(proposed_at, timezone.utc).isoformat(timespec="milliseconds")
        plan["age_s"] = age
    return plan


class FakeBrain:
    """Stands in for engram/api.py and records every call it gets. As in the real brain, a command
    adds a waiting plan, and approving or rejecting one takes it off the waiting list."""

    def __init__(self, pending=None, down=(), down_once=(), stop_reply=None, reject_reply=None, date=True):
        self.calls = []
        self.pending = list(pending or [])
        self.rejected = set()
        self.made = 0
        self.stop_reply = stop_reply or {"adapter": "mock", "stopped": True}
        self.reject_reply = reject_reply  # (status, body) to answer every reject with, rejecting nothing
        self.down = set(down)             # paths that fail as if the brain were unreachable
        self.down_once = set(down_once)   # paths that fail the first time only
        self.date = date                  # False: no Date header, as from a server that sends none
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle), base_url="http://brain.test")

    def handle(self, request):
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, body))
        path = request.url.path
        if path in self.down or path in self.down_once:
            self.down_once.discard(path)
            raise httpx.ConnectError("no route to host", request=request)
        route = (request.method, path)
        if route == ("GET", "/health"):
            return httpx.Response(200, json={"ok": True, "robot": "mock"})
        if route == ("POST", "/stop"):
            return httpx.Response(200, json=self.stop_reply)
        if route == ("GET", "/operations/pending"):
            return httpx.Response(200, json=self.pending, headers=brain_date() if self.date else {})
        if route == ("POST", "/commands"):
            self.made += 1
            op = f"ep_new{self.made}"
            self.pending.insert(0, waiting_plan(op, age=0.0, request=body["text"]))
            return httpx.Response(200, json={"kind": "proposal", "status": "proposed", "operation_id": op})
        _, kind, op, verb = (path.split("/") + ["", "", "", ""])[:4]
        waiting = next((p for p in self.pending if p["operation_id"] == op), None)
        if (request.method, kind, verb) == ("POST", "operations", "approve"):
            if waiting is None:
                return httpx.Response(409, json={"detail": "operation is not waiting for approval"})
            self.pending.remove(waiting)
            return httpx.Response(200, json={"operation_id": op, "outcome": "success"})
        if (request.method, kind, verb) == ("POST", "operations", "reject"):
            if self.reject_reply is not None:
                return httpx.Response(self.reject_reply[0], json=self.reject_reply[1])
            if waiting is None and op not in self.rejected:
                return httpx.Response(404, json={"detail": f"no operation {op}"})
            if waiting is not None:
                self.pending.remove(waiting)
            again = op in self.rejected
            self.rejected.add(op)
            return httpx.Response(200, json={"operation_id": op, "outcome": "cancelled", "already_rejected": again,
                                             "failure": f"rejected by {body['rejected_by']}: {body['reason']}"})
        return httpx.Response(404, json={"detail": "Not Found"})

    def brain(self):
        return ag.Brain("http://brain.test", client=self.client)


def run_main(argv, *, brain=None, said=None, classifier=fake_classifier, **kw):
    out, t = io.StringIO(), FakeTime()
    said = [] if said is None else said
    kw.setdefault("clock", t.clock)
    kw.setdefault("sleep", t.sleep)
    code = ag.main(argv, classifier=classifier, brain=brain, speaker=lambda text: said.append(text) or True,
                   out=out, background=False, **kw)
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def actions(lines):
    return [(x["action"], x["status"]) for x in lines if x["event"] == "action"]


def action_line(lines, kind):
    return next(x for x in lines if x["event"] == "action" and x["action"] == kind)


REJECT_BODY = {"rejected_by": "hand-signal", "reason": "patient signalled no"}


# ---------------------------------------------------------------------------
# What the brain receives
# ---------------------------------------------------------------------------
def test_the_brain_gets_exactly_the_expected_calls(tmp_path):
    fb, said = FakeBrain(pending=[waiting_plan("ep_1")]), []
    folder = frames(tmp_path, ["thumbs_up"] * 4 + PAUSE + ["peace"] * 4 + PAUSE + ["open_palm"] + PAUSE
                    + ["thumbs_down"] * 4 + PAUSE + ["fist"] * 4)
    code, lines = run_main(["--images", str(folder), "--rate", "4", "--brain", "http://brain.test"],
                           brain=fb.brain(), said=said)
    assert code == 0
    assert fb.calls == [
        ("GET", "/health", None),
        ("GET", "/operations/pending", None),
        ("POST", "/operations/ep_1/approve", {"digest": DIGEST, "approved_by": WHO}),
        ("POST", "/commands", {"text": "Bring me my pills", "who": WHO}),
        ("POST", "/stop", None),
        ("GET", "/operations/pending", None),
        ("POST", "/operations/ep_new1/reject", REJECT_BODY),   # the pills plan, the one waiting by then
        ("POST", "/commands", {"text": "Pick it up", "who": WHO}),
    ]
    assert said == ["Okay, cancelled"]
    assert actions(lines) == [("approve", "sent"), ("command", "sent"), ("stop", "sent"), ("cancel", "sent"),
                              ("command", "sent")]
    approve = action_line(lines, "approve")
    assert approve["operation_id"] == "ep_1" and approve["outcome"] == "success"
    assert approve["plan_age_s"] == 5.0                      # the brain's own age_s, not the Date header
    cancel = action_line(lines, "cancel")
    assert (cancel["operation_id"], cancel["outcome"], cancel["already_rejected"]) == ("ep_new1", "cancelled", False)
    assert [x["signal"] for x in lines if x["event"] == "signal"] == ["thumbs_up", "peace", "open_palm",
                                                                      "thumbs_down", "fist"]


@pytest.mark.parametrize("case, why", [
    ("none waiting", "no plan is waiting"),
    ("two waiting", "2 plans"),
    ("stale by age_s", "75 s old"),
    ("age_s stale, proposed_at fresh", "75 s old"),   # the brain's age_s decides, not the Date header
    ("no age_s, stale by the Date header", "old"),
    ("no age_s and no Date header", "no Date header"),
])
def test_hand_signals_act_only_on_exactly_one_recent_plan(tmp_path, case, why):
    pending = {   # built here, not at import, so the Date-header ages are what they say
        "none waiting": lambda: [],
        "two waiting": lambda: [waiting_plan("ep_1"), waiting_plan("ep_2", age=1.0)],
        "stale by age_s": lambda: [waiting_plan("ep_1", age=75.0)],
        "age_s stale, proposed_at fresh": lambda: [{**waiting_plan("ep_1", age=5.0), "age_s": 75.0}],
        "no age_s, stale by the Date header": lambda: [waiting_plan("ep_1", age=75.0, age_s=False)],
        "no age_s and no Date header": lambda: [waiting_plan("ep_1", age=5.0, age_s=False)],
    }[case]()
    fb, said = FakeBrain(pending=pending, date=case != "no age_s and no Date header"), []
    folder = frames(tmp_path, ["thumbs_up"] * 4 + PAUSE + ["thumbs_down"] * 4)
    code, lines = run_main(["--images", str(folder), "--rate", "4"], brain=fb.brain(), said=said)
    assert code == 0
    assert [c[:2] for c in fb.calls] == [("GET", "/health"), ("GET", "/operations/pending"),
                                         ("GET", "/operations/pending")]   # no approve, no reject
    assert actions(lines) == [("approve", "refused"), ("cancel", "refused")]
    assert all(why in x["reason"] for x in lines if x["event"] == "action")
    assert said == []


def test_without_age_s_the_date_header_ages_the_plan(tmp_path):
    fb = FakeBrain(pending=[waiting_plan("ep_1", age=5.0, age_s=False)])
    code, lines = run_main(["--images", str(frames(tmp_path, ["thumbs_up"] * 4))], brain=fb.brain())
    approve = action_line(lines, "approve")
    assert approve["status"] == "sent" and 5.0 <= approve["plan_age_s"] < 10.0   # rounded up by the slack


def test_a_thumbs_down_rejects_the_waiting_plan_and_only_then_says_so(tmp_path):
    fb = FakeBrain(pending=[waiting_plan("ep_1")])
    folder = frames(tmp_path, ["thumbs_down"] * 4 + PAUSE + ["thumbs_up"] * 4)
    code, lines = run_main(["--images", str(folder), "--rate", "4"], brain=fb.brain(), said=fb.calls)
    assert fb.calls[:5] == [("GET", "/health", None), ("GET", "/operations/pending", None),
                            ("POST", "/operations/ep_1/reject", REJECT_BODY),
                            "Okay, cancelled",                      # said after the brain confirmed the reject
                            ("GET", "/operations/pending", None)]
    assert actions(lines) == [("cancel", "sent"), ("approve", "refused")]
    assert "no plan is waiting" in action_line(lines, "approve")["reason"]   # the plan left the waiting list


@pytest.mark.parametrize("failure", ["unreachable", "409", "answer is not cancelled"])
def test_a_failed_reject_says_nothing_and_no_thumbs_up_approves_that_plan(tmp_path, failure):
    how = {"unreachable": {"down": {"/operations/ep_1/reject"}},
           "409": {"reject_reply": (409, {"detail": "operation is running, not waiting for approval"})},
           "answer is not cancelled": {"reject_reply": (200, {"operation_id": "ep_1", "outcome": "proposed"})}}
    fb, said = FakeBrain(pending=[waiting_plan("ep_1")], **how[failure]), []
    folder = frames(tmp_path, ["thumbs_down"] * 4 + PAUSE + ["thumbs_up"] * 4)
    code, lines = run_main(["--images", str(folder), "--rate", "4"], brain=fb.brain(), said=said)
    assert said == []
    assert actions(lines) == [("cancel", "error"), ("approve", "refused")]
    assert action_line(lines, "cancel")["remembered"] == ["ep_1"]
    assert "cancelled with a thumbs-down" in action_line(lines, "approve")["reason"]   # the backstop
    assert not [c for c in fb.calls if c[1].endswith("/approve")]


def test_another_thumbs_down_tries_a_failed_reject_again(tmp_path):
    fb, said = FakeBrain(pending=[waiting_plan("ep_1")], down_once={"/operations/ep_1/reject"}), []
    folder = frames(tmp_path, ["thumbs_down"] * 4 + PAUSE + ["thumbs_down"] * 4)
    code, lines = run_main(["--images", str(folder), "--rate", "4"], brain=fb.brain(), said=said)
    assert actions(lines) == [("cancel", "error"), ("cancel", "sent")]
    assert [c[:2] for c in fb.calls].count(("POST", "/operations/ep_1/reject")) == 2
    assert said == ["Okay, cancelled"]


def test_a_thumbs_down_that_cannot_read_the_plans_turns_thumbs_up_approvals_off(tmp_path):
    fb, said = FakeBrain(pending=[waiting_plan("ep_1")], down_once={"/operations/pending"}), []
    folder = frames(tmp_path, ["thumbs_down"] * 4 + PAUSE + ["thumbs_up"] * 4)
    code, lines = run_main(["--images", str(folder), "--rate", "4"], brain=fb.brain(), said=said)
    assert actions(lines) == [("cancel", "error"), ("approve", "refused")]
    assert "approvals are off" in action_line(lines, "cancel")["warning"] and said == []
    assert "off for" in action_line(lines, "approve")["reason"]   # the plan list was readable by then
    assert [c[:2] for c in fb.calls].count(("GET", "/operations/pending")) == 2
    assert not [c for c in fb.calls if c[1].endswith(("/approve", "/reject"))]


# ---------------------------------------------------------------------------
# Against the real brain API (engram/api.py) on the mock robot, with the default map
# ---------------------------------------------------------------------------
@pytest.fixture
def real_brain(engine):
    from fastapi.testclient import TestClient

    from engram.api import create_app

    app = create_app(engine)

    @app.middleware("http")
    async def date_header(request, call_next):   # uvicorn adds this; the test client does not
        response = await call_next(request)
        response.headers["date"] = email.utils.formatdate(time.time(), usegmt=True)
        return response

    return ag.Brain("http://testserver", client=TestClient(app))


def test_with_the_real_brain_a_thumbs_up_approves_and_runs_the_plan(tmp_path, engine, real_brain):
    folder = frames(tmp_path, ["peace"] * 4 + PAUSE + ["thumbs_up"] * 4)
    code, lines = run_main(["--images", str(folder), "--rate", "4"], brain=real_brain)
    assert actions(lines) == [("command", "sent"), ("approve", "sent")]
    command, approve = [x for x in lines if x["event"] == "action"]
    assert command["text"] == "Bring me my pills" and command["result"] == "proposed"
    assert approve["operation_id"] == command["operation_id"]
    assert approve["outcome"] == "success"
    row = engine.conn.execute("SELECT approved_by, digest FROM approval WHERE target_id = ?",
                              (approve["operation_id"],)).fetchone()
    assert row["approved_by"] == WHO
    assert engine.robot.state()["at_zone"] == "patient"   # "me" is the patient's place


def test_with_the_real_brain_a_thumbs_down_rejects_the_plan(tmp_path, engine, real_brain):
    said = []
    folder = frames(tmp_path, ["peace"] * 4 + PAUSE + ["thumbs_down"] * 4 + PAUSE + ["thumbs_up"] * 4)
    code, lines = run_main(["--images", str(folder), "--rate", "4"], brain=real_brain, said=said)
    assert actions(lines) == [("command", "sent"), ("cancel", "sent"), ("approve", "refused")]
    command, cancel, approve = [x for x in lines if x["event"] == "action"]
    assert cancel["operation_id"] == command["operation_id"] and cancel["outcome"] == "cancelled"
    assert "no plan is waiting" in approve["reason"]                   # the rejected plan left the waiting list
    ep = engine.conn.execute("SELECT outcome, failure FROM episode WHERE id = ?", (command["operation_id"],)).fetchone()
    assert (ep["outcome"], ep["failure"]) == ("cancelled", "rejected by hand-signal: patient signalled no")
    assert engine.conn.execute("SELECT COUNT(*) FROM approval").fetchone()[0] == 0
    assert said == ["Okay, cancelled"]


# ---------------------------------------------------------------------------
# Rehearsal: --images with --dry-run
# ---------------------------------------------------------------------------
def test_an_images_dry_run_logs_the_actions_and_never_calls_the_brain(tmp_path):
    fb, said = FakeBrain(pending=[waiting_plan("ep_1")]), []
    signals = ["none"] + ["thumbs_up"] * 4 + PAUSE + ["thumbs_down"] * 4 + PAUSE + ["peace"] * 4 + ["open_palm"]
    folder = frames(tmp_path, signals)
    code, lines = run_main(["--images", str(folder), "--dry-run", "--rate", "4"], brain=fb.brain(), said=said)
    assert code == 0
    assert fb.calls == []
    assert lines[0]["event"] == "start" and lines[0]["dry_run"] is True and "brain" not in lines[0]
    assert actions(lines) == [("approve", "dry_run"), ("cancel", "dry_run"), ("command", "dry_run"),
                              ("stop", "dry_run")]   # the stop came 0.25 s after the command: no pause for STOP
    assert [x["file"] for x in lines if x["event"] == "signal"] == ["004.jpg", "021.jpg", "038.jpg", "039.jpg"]
    assert said == []                                   # "Okay, cancelled" only after a real reject
    assert action_line(lines, "cancel")["would_say"] == "Okay, cancelled"
    assert lines[-1] == {**lines[-1], "event": "exit", "reason": "no more images", "frames": len(signals)}


def test_the_map_file_given_with_map_is_the_one_used(tmp_path):
    path = tmp_path / "waving_stops.json"
    path.write_text(json.dumps({"signals": {"waving": {"action": "stop"}}}))
    code, lines = run_main(["--images", str(frames(tmp_path, ["open_palm", "waving"])), "--dry-run",
                            "--map", str(path)])
    assert actions(lines) == [("stop", "dry_run")]
    assert lines[0]["signals"]["waving"] == "stop" and lines[0]["signals"]["open_palm"] == "none"


def test_a_broken_map_stops_the_loop_before_it_starts(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"signals": {"thumbs_up": {"action": "approve"}}}))   # nothing stops
    code, lines = run_main(["--images", str(frames(tmp_path, ["none"])), "--dry-run", "--map", str(path)])
    assert code == 2 and lines == [{**lines[0], "event": "error", "where": "setup"}]


# ---------------------------------------------------------------------------
# Never crashes
# ---------------------------------------------------------------------------
def test_failures_and_a_dead_brain_never_stop_the_loop(tmp_path):
    fb = FakeBrain(down={"/stop", "/commands"})
    folder = frames(tmp_path, ["error", "ok_sign", "thumbs_up:nan", "open_palm", "open_palm"] + PAUSE
                    + ["fist"] * 4 + ["none"])
    code, lines = run_main(["--images", str(folder), "--rate", "4"], brain=fb.brain())
    assert code == 0 and lines[-1]["event"] == "exit" and lines[-1]["reason"] == "no more images"
    assert [c[:2] for c in fb.calls] == [("GET", "/health"), ("POST", "/stop"), ("POST", "/stop"),
                                         ("POST", "/commands")]   # a failed STOP is tried again next frame
    assert actions(lines) == [("stop", "error"), ("stop", "error"), ("command", "error")]
    assert [x["where"] for x in lines if x["event"] == "error"] == ["classifier"] * 3


@pytest.mark.parametrize("reply, warned", [
    ({"adapter": "mock", "stopped": True}, False),
    ({"adapter": "brian8200", "connected": False, "stopped": True, "error": "timed out"}, True),   # robot unreachable
    ({"adapter": "brian8200", "connected": True, "stopped": False}, True),                        # stop did not latch
])
def test_a_stop_the_brain_does_not_confirm_is_flagged(tmp_path, reply, warned):
    fb = FakeBrain(stop_reply=reply)
    code, lines = run_main(["--images", str(frames(tmp_path, ["open_palm"]))], brain=fb.brain())
    stop = action_line(lines, "stop")
    assert stop["status"] == "sent" and ("warning" in stop) is warned


def test_a_stop_says_its_line_once_per_palm(tmp_path):
    path, said = tmp_path / "say_stop.json", []
    path.write_text(json.dumps({"signals": {"open_palm": {"action": "stop", "say": "Stopping"}}}))
    code, lines = run_main(["--images", str(frames(tmp_path, ["open_palm"] * 3 + ["none", "open_palm"])),
                            "--dry-run", "--map", str(path)], said=said)
    assert actions(lines) == [("stop", "dry_run")] * 4 and said == ["Stopping", "Stopping"]


def test_an_action_that_raises_is_logged_not_fatal():
    class BrokenBrain:
        def stop(self):
            raise RuntimeError("bug")

    out = io.StringIO()
    ag.Assist(BrokenBrain(), ag.Log(out), say=lambda text: None, background=False).do(
        Decision("open_palm", 0.9, Action("stop"), True))
    (line,) = [json.loads(x) for x in out.getvalue().splitlines()]
    assert (line["action"], line["status"]) == ("stop", "error") and "bug" in line["error"]


def test_a_failed_classification_breaks_a_hold_in_the_loop(tmp_path):
    folder = frames(tmp_path, ["thumbs_up", "error"] + ["thumbs_up"] * 4)
    code, lines = run_main(["--images", str(folder), "--dry-run", "--rate", "4"])
    assert [x["file"] for x in lines if x["event"] == "signal"] == ["005.jpg"]   # held again from 002.jpg


def test_an_unexpected_error_in_one_frame_is_logged_and_the_loop_goes_on(tmp_path, monkeypatch):
    real, calls = Decider.decide, []

    def flaky(self, reading, now, busy=False):
        calls.append(now)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return real(self, reading, now, busy)

    monkeypatch.setattr(Decider, "decide", flaky)
    code, lines = run_main(["--images", str(frames(tmp_path, ["none", "open_palm"])), "--dry-run"])
    assert code == 0 and [x["where"] for x in lines if x["event"] == "error"] == ["loop"]
    assert actions(lines) == [("stop", "dry_run")]


def test_robot_camera_trouble_is_no_signal_and_the_loop_goes_on():
    replies = [httpx.Response(503, json={"error": "E_CAMERA"}), httpx.Response(200, content=b"<html>"), None,
               httpx.Response(200, content=JPEG + b"open_palm")]

    def camera(request):
        assert request.url.path == "/robot/frame" and request.url.params["annotate"] == "false"
        reply = replies.pop(0)
        if reply is None:
            raise httpx.ConnectTimeout("timed out", request=request)
        return reply

    source = ag.RobotCamera("http://pi.test:8200", client=httpx.Client(
        transport=httpx.MockTransport(camera), base_url="http://pi.test:8200"))
    t, out = FakeTime(), io.StringIO()

    def sleep(seconds):   # Ctrl+C once the four replies are used up
        t.sleep(seconds)
        if not replies:
            raise KeyboardInterrupt

    args = ag.parse_args(["--robot", "http://pi.test:8200", "--dry-run"])
    code = ag.run(args, load_map(), source, fake_classifier, None, ag.Log(out), None, t.clock, sleep, False)
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert code == 0
    assert [x["where"] for x in lines if x["event"] == "error"] == ["frame"] * 3
    assert actions(lines) == [("stop", "dry_run")]
    assert lines[-1]["event"] == "exit" and lines[-1]["reason"] == "ctrl+c" and lines[-1]["frames"] == 1


@pytest.mark.parametrize("where", ["sleep", "classifier"])
def test_ctrl_c_exits_cleanly(tmp_path, where):
    t, seen = FakeTime(), []

    def classifier(jpeg):
        seen.append(jpeg)
        if where == "classifier" and len(seen) == 3:
            raise KeyboardInterrupt
        return fake_classifier(jpeg)

    def sleep(seconds):
        t.sleep(seconds)
        if where == "sleep" and len(seen) == 3:
            raise KeyboardInterrupt

    code, lines = run_main(["--images", str(frames(tmp_path, ["none"] * 10)), "--dry-run"],
                           classifier=classifier, clock=t.clock, sleep=sleep)
    assert code == 0 and lines[-1]["event"] == "exit" and lines[-1]["reason"] == "ctrl+c"
    assert lines[-1]["frames"] == 3


# ---------------------------------------------------------------------------
# STOP never waits for a slow request
# ---------------------------------------------------------------------------
def test_stop_goes_through_while_an_approve_is_still_running():
    calls, entered, release = [], threading.Event(), threading.Event()

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/operations/pending":
            return httpx.Response(200, json=[waiting_plan("ep_1")], headers=brain_date())
        if request.url.path == "/operations/ep_1/approve":   # the brain runs the plan inside this call
            entered.set()
            release.wait(5)
            return httpx.Response(200, json={"operation_id": "ep_1", "outcome": "aborted"})
        return httpx.Response(200, json={"adapter": "mock", "stopped": True})

    brain = ag.Brain("http://brain.test", client=httpx.Client(transport=httpx.MockTransport(handler),
                                                              base_url="http://brain.test"))
    out = io.StringIO()
    assist = ag.Assist(brain, ag.Log(out), say=lambda text: None, background=True)
    try:
        assist.do(Decision("thumbs_up", 0.9, Action("approve"), True))
        assert entered.wait(5) and assist.busy
        decider = Decider(load_map())
        late = [decider.decide(Reading("fist", 0.9), 10.0 + i, busy=assist.busy) for i in range(4)]
        assert late[1].go is False and "still running" in late[1].why    # other signals wait their turn...
        assist.do(Decision("open_palm", 0.9, Action("stop"), True))      # ...STOP does not
        assert calls[-1] == ("POST", "/stop") and assist.busy, "the STOP waited for the approve"
    finally:
        release.set()
    assert assist.wait(5)
    assert calls == [("GET", "/operations/pending"), ("POST", "/operations/ep_1/approve"), ("POST", "/stop")]
    done = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [(x["action"], x["status"]) for x in done] == [("stop", "sent"), ("approve", "sent")]


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------
def test_the_brains_clock_comes_from_its_date_header():
    r = httpx.Response(200, headers={"date": "Sun, 25 Oct 2026 12:00:00 GMT"})
    noon = datetime(2026, 10, 25, 12, tzinfo=timezone.utc).timestamp()
    assert ag.brain_clock(r) == noon + 2.0                          # rounded up by the header's slack
    assert ag.brain_clock(httpx.Response(200)) is None
    assert ag.brain_clock(httpx.Response(200, headers={"date": "yesterday"})) is None


def test_brain_errors_carry_the_reason():
    def handler(request):
        if request.url.path == "/operations/ep_1/approve":
            return httpx.Response(409, json={"detail": "digest does not match the proposed plan"})
        return httpx.Response(200, content=b"<html>")

    brain = ag.Brain("http://brain.test", client=httpx.Client(transport=httpx.MockTransport(handler),
                                                              base_url="http://brain.test"))
    with pytest.raises(ag.BrainError, match="409.*digest does not match"):
        brain.approve("ep_1", DIGEST)
    with pytest.raises(ag.BrainError, match="not JSON"):
        brain.pending()


def test_no_voice_is_not_a_problem():
    out = io.StringIO()

    def broken(text):
        raise OSError("no speaker")

    ag.Voice(ag.Log(out), broken)("Okay, cancelled")
    ag.Voice(ag.Log(out), None)("Okay, cancelled")
    assert [json.loads(line)["spoken"] for line in out.getvalue().splitlines()] == [False, False]


def test_the_intent_classifier_is_called_with_the_frame(tmp_path, monkeypatch):
    got = []
    fake = types.ModuleType("engram.vision.intent_vision")
    fake.VisionError = VisionError
    fake.classify_hand_signal = lambda jpeg, *, timeout_s=4.0: got.append(jpeg) or fake_classifier(jpeg)
    monkeypatch.setitem(sys.modules, "engram.vision.intent_vision", fake)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-value")
    monkeypatch.setenv("TYPESAFE_INTENT_MODEL", "test-model")
    out = io.StringIO()
    code = ag.main(["--images", str(frames(tmp_path, ["open_palm"])), "--dry-run", "--classifier", "intent"],
                   speaker=lambda text: True, out=out, sleep=lambda s: None, background=False)
    assert code == 0 and got == [b"open_palm"]
    assert '"action": "stop"' in out.getvalue()


def test_the_intent_classifier_needs_its_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_INTENT_MODEL", "test-model")
    code, lines = run_main(["--images", str(frames(tmp_path, ["none"])), "--dry-run"], classifier=None)
    assert code == 2 and "TYPESAFE_API_KEY" in lines[0]["error"] and "TYPESAFE_INTENT_MODEL" not in lines[0]["error"]


def test_mediapipe_missing_is_reported(tmp_path, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "mediapipe" else real(name, *a))
    code, lines = run_main(["--images", str(frames(tmp_path, ["none"])), "--classifier", "mediapipe"],
                           classifier=None)
    assert code == 2 and "pip install mediapipe" in lines[0]["error"]


def test_it_runs_as_a_script(tmp_path):
    script = str(ROOT / "scripts/assist_gestures.py")
    out = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "--robot" in out.stdout and "--dry-run" in out.stdout
    env = {k: v for k, v in os.environ.items() if k not in ("TYPESAFE_API_KEY", "TYPESAFE_INTENT_MODEL")}
    out = subprocess.run([sys.executable, script, "--images", str(frames(tmp_path, ["none"])), "--dry-run"],
                         capture_output=True, text=True, timeout=60, env=env)
    assert out.returncode == 2
    assert "TYPESAFE_API_KEY and TYPESAFE_INTENT_MODEL" in json.loads(out.stdout.splitlines()[-1])["error"]
