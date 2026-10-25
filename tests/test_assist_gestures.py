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


def waiting_plan(op="ep_1", age=5.0):
    return {"operation_id": op, "request": "Bring me my pills", "plan": {"goal": "Bring me my pills", "steps": []},
            "digest": DIGEST, "source": "rules", "planner_note": "", "proposed_at": time.time() - age}


class FakeBrain:
    """Stands in for engram/api.py and records every call it gets."""

    def __init__(self, pending=None, down=(), down_once=(), stop_reply=None):
        self.calls = []
        self.pending = pending if pending is not None else []
        self.stop_reply = stop_reply or {"adapter": "mock", "stopped": True}
        self.down = set(down)            # paths that fail as if the brain were unreachable
        self.down_once = set(down_once)  # paths that fail the first time only
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle), base_url="http://brain.test")

    def handle(self, request):
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, body))
        if request.url.path in self.down or request.url.path in self.down_once:
            self.down_once.discard(request.url.path)
            raise httpx.ConnectError("no route to host", request=request)
        route = (request.method, request.url.path)
        if route == ("GET", "/health"):
            return httpx.Response(200, json={"ok": True, "robot": "mock"})
        if route == ("POST", "/stop"):
            return httpx.Response(200, json=self.stop_reply)
        if route == ("GET", "/operations/pending"):
            return httpx.Response(200, json=self.pending, headers=brain_date())
        if route == ("POST", "/commands"):
            return httpx.Response(200, json={"kind": "proposal", "status": "proposed", "operation_id": "ep_new"})
        if request.method == "POST" and request.url.path.endswith("/approve"):
            return httpx.Response(200, json={"operation_id": request.url.path.split("/")[2], "outcome": "success"})
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
        ("GET", "/operations/pending", None),          # the thumbs-down notes which plan it turned down
        ("POST", "/commands", {"text": "Pick it up", "who": WHO}),
    ]
    assert said == ["Okay, cancelled"]
    assert actions(lines) == [("approve", "sent"), ("command", "sent"), ("stop", "sent"), ("cancel", "done"),
                              ("command", "sent")]
    approve = action_line(lines, "approve")
    assert approve["operation_id"] == "ep_1" and approve["outcome"] == "success" and 4 < approve["plan_age_s"] < 10
    cancel = action_line(lines, "cancel")
    assert cancel["cancelled"] == ["ep_1"] and "no reject endpoint" in cancel["note"]
    assert [x["signal"] for x in lines if x["event"] == "signal"] == ["thumbs_up", "peace", "open_palm",
                                                                      "thumbs_down", "fist"]


@pytest.mark.parametrize("pending, why", [
    ([], "no plan is waiting"),
    ([waiting_plan("ep_1"), waiting_plan("ep_2", age=1.0)], "2 plans"),
    ([waiting_plan("ep_1", age=75.0)], "old"),
])
def test_a_thumbs_up_approves_nothing_unless_exactly_one_recent_plan_waits(tmp_path, pending, why):
    fb = FakeBrain(pending=pending)
    code, lines = run_main(["--images", str(frames(tmp_path, ["thumbs_up"] * 4))], brain=fb.brain())
    assert code == 0
    assert [c[:2] for c in fb.calls] == [("GET", "/health"), ("GET", "/operations/pending")]   # no approve
    (refused,) = [x for x in lines if x["event"] == "action"]
    assert refused["status"] == "refused" and why in refused["reason"]


def test_a_thumbs_down_keeps_that_plan_from_being_approved(tmp_path):
    fb = FakeBrain(pending=[waiting_plan("ep_1")])
    folder = frames(tmp_path, ["thumbs_down"] * 4 + PAUSE + ["thumbs_up"] * 4)
    code, lines = run_main(["--images", str(folder), "--rate", "4"], brain=fb.brain())
    assert not [c for c in fb.calls if c[1].endswith("/approve")]
    assert actions(lines) == [("cancel", "done"), ("approve", "refused")]
    assert "cancelled with a thumbs-down" in lines[-2]["reason"]


def test_a_thumbs_down_that_cannot_read_the_plans_turns_thumbs_up_approvals_off(tmp_path):
    fb = FakeBrain(pending=[waiting_plan("ep_1")], down_once={"/operations/pending"})
    folder = frames(tmp_path, ["thumbs_down"] * 4 + PAUSE + ["thumbs_up"] * 4)
    code, lines = run_main(["--images", str(folder), "--rate", "4"], brain=fb.brain())
    assert actions(lines) == [("cancel", "done"), ("approve", "refused")]
    cancel = action_line(lines, "cancel")
    assert cancel["cancelled"] == [] and "approvals are off" in cancel["warning"]
    assert "off for" in lines[-2]["reason"]                       # the plan list was readable by then
    assert [c[:2] for c in fb.calls].count(("GET", "/operations/pending")) == 2
    assert not [c for c in fb.calls if c[1].endswith("/approve")]


# ---------------------------------------------------------------------------
# Against the real brain API (engram/api.py) on the mock robot
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


def beaker_map(tmp_path):
    data = json.loads((ROOT / "engram/vision/assist_signals.default.json").read_text())
    data["signals"]["peace"]["text"] = "Bring me the beaker"   # the demo arena has no pills yet
    path = tmp_path / "beaker.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_with_the_real_brain_a_thumbs_up_approves_and_runs_the_plan(tmp_path, engine, real_brain):
    folder = frames(tmp_path, ["peace"] * 4 + PAUSE + ["thumbs_up"] * 4)
    code, lines = run_main(["--images", str(folder), "--rate", "4", "--map", beaker_map(tmp_path)], brain=real_brain)
    assert actions(lines) == [("command", "sent"), ("approve", "sent")]
    command, approve = [x for x in lines if x["event"] == "action"]
    assert command["result"] == "proposed" and approve["operation_id"] == command["operation_id"]
    assert approve["outcome"] == "success"
    row = engine.conn.execute("SELECT approved_by, digest FROM approval WHERE target_id = ?",
                              (approve["operation_id"],)).fetchone()
    assert row["approved_by"] == WHO
    assert engine.robot.state()["at_zone"] == "bench"   # the beaker run went to the bench


def test_with_the_real_brain_a_thumbs_down_means_nothing_runs(tmp_path, engine, real_brain):
    folder = frames(tmp_path, ["peace"] * 4 + PAUSE + ["thumbs_down"] * 4 + PAUSE + ["thumbs_up"] * 4)
    code, lines = run_main(["--images", str(folder), "--rate", "4", "--map", beaker_map(tmp_path)], brain=real_brain)
    assert actions(lines) == [("command", "sent"), ("cancel", "done"), ("approve", "refused")]
    op = lines[[x["event"] for x in lines].index("action")]["operation_id"]
    assert engine.conn.execute("SELECT outcome FROM episode WHERE id = ?", (op,)).fetchone()["outcome"] == "proposed"
    assert engine.conn.execute("SELECT COUNT(*) FROM approval").fetchone()[0] == 0


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
    assert said == ["Okay, cancelled"]
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
    from datetime import datetime, timezone

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
