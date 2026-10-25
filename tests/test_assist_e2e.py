"""ENGRAM Assist end to end: the brain's HTTP API drives Brian's robot server on the simulator.

Only the hardware is pretend. The brain is engram.api.default_engine() with ENGRAM_ROBOT_API=brian,
which is what scripts/cds_up.sh --real starts, served on a local port. Brian's server is
robot/engram_robot (RobotService and its HTTP app, the code the Pi runs) with the simulator driver
and the forward-only ultrasonic sensor this robot has, on another port with a test secret. Every
test reads the robot server's own journal to see what it was asked to do and what happened.
"""

import json
import math
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from conftest import ROOT

pytest.importorskip("cv2")   # the simulator renders real ArUco tags
import uvicorn  # noqa: E402

SECRET = "assist-e2e-test-secret"


def _robot_server():
    """Brian's server is its own project with a top-level package called robot. Import it without
    leaving its folder on sys.path, where its voice and tools packages could shadow others."""
    path = str(ROOT / "robot/engram_robot")
    if getattr(sys.modules.get("robot"), "__file__", "") is None:   # the repo's robot/ folder as a namespace
        del sys.modules["robot"]
    sys.path.insert(0, path)
    try:
        from robot.config import Config
        from robot.server import create_app
        from robot.service import RobotService
        from robot.sim import SimDriver, SimTag
    finally:
        sys.path.remove(path)
    return Config, create_app, RobotService, SimDriver, SimTag


Config, robot_app, RobotService, SimDriver, SimTag = _robot_server()


def call(url, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def serve(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.02)
    return f"http://127.0.0.1:{sock.getsockname()[1]}", server, thread


def wait_until(cond, timeout_s):
    deadline = time.monotonic() + timeout_s
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


class Rig:
    def __init__(self, brain, robot, service, sim):
        self.brain, self.robot, self.service, self.sim = brain, robot, service, sim

    def command(self, text):
        return call(self.brain + "/commands", {"text": text, "who": "patient"})

    def approve(self, proposal):
        assert proposal["status"] == "proposed", proposal
        return call(f"{self.brain}/operations/{proposal['operation_id']}/approve",
                    {"digest": proposal["digest"], "approved_by": "nurse"})

    def journal(self):
        """(op, args as the brain sent them, status, code) for every operation the robot server ran, in order."""
        rows = call(self.robot + "/robot/journal?limit=500")
        done = {r["operation_id"]: r["result"] for r in rows if r["phase"] == "finished"}
        return [(r["op"], r["args"], done[r["operation_id"]]["status"], done[r["operation_id"]]["code"])
                for r in rows if r["phase"] == "started"]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """The pill bottle (tag 5) 0.8 m in front of the robot, the patient's marker (tag 15) on a post beyond it."""
    cfg = Config()
    sim = SimDriver(cfg, tags=[SimTag(5, 0.8, 0.05), SimTag(15, 1.6, 0.4, kind="zone", lidar_radius=0.04)],
                    range_sensor="ultrasonic")
    service = RobotService(sim, cfg, SECRET, str(tmp_path / "robot_journal.jsonl"))
    robot_url, robot_server, robot_thread = serve(robot_app(service))
    key = tmp_path / "robot.env"
    key.write_text(f"ENGRAM_ROBOT_SECRET={SECRET}\n")
    monkeypatch.setenv("ENGRAM_ROBOT_API", "brian")
    monkeypatch.setenv("ENGRAM_ROBOT_URL", robot_url)
    monkeypatch.setenv("ENGRAM_ROBOT_KEY_FILE", str(key))
    monkeypatch.setenv("ENGRAM_DB", str(tmp_path / "engram.db"))
    for name in ("ENGRAM_ARENA", "ENGRAM_LLM_URL", "ENGRAM_VOICE_PRESETS"):
        monkeypatch.delenv(name, raising=False)
    from engram.api import create_app, default_engine

    engine = default_engine()
    assert engine.robot.state()["adapter"] == "brian8200"
    brain_url, brain_server, brain_thread = serve(create_app(engine))
    yield Rig(brain_url, robot_url, service, sim)
    brain_server.should_exit = robot_server.should_exit = True
    brain_thread.join(10)
    robot_thread.join(10)
    engine.conn.close()
    service.close()


def test_pick_it_up_then_put_it_down(rig):
    p = rig.command("Pick it up")
    assert [s["op"] for s in p["plan"]["steps"]] == ["arm_pose", "gripper_close", "arm_pose"]
    assert rig.journal() == []                                          # nothing moves before approval
    out = rig.approve(p)
    assert out["outcome"] == "success", out
    assert rig.journal() == [("arm_pose", {"pose": "reach"}, "UNKNOWN", "E_UNVERIFIED"),
                             ("gripper_close", {}, "SUCCEEDED", "OK"),
                             ("arm_pose", {"pose": "carry"}, "UNKNOWN", "E_UNVERIFIED")]
    assert rig.sim.arm_pose == "carry" and not rig.sim.claw_open
    # The arm has no joint sensor: the brain says so instead of claiming it was checked.
    assert all("not sensor checked" in s["detail"] for s in out["steps"] if s["op"] == "arm_pose")
    out = rig.approve(rig.command("Put it down"))
    assert out["outcome"] == "success", out
    assert rig.journal()[3:] == [("arm_pose", {"pose": "reach"}, "UNKNOWN", "E_UNVERIFIED"),
                                 ("gripper_open", {}, "SUCCEEDED", "OK"),
                                 ("arm_pose", {"pose": "carry"}, "UNKNOWN", "E_UNVERIFIED")]
    assert rig.sim.arm_pose == "carry" and rig.sim.claw_open


def test_move_forward_drives_20_cm_and_back_up_comes_back(rig):
    out = rig.approve(rig.command("Move forward"))
    assert out["outcome"] == "success", out
    assert rig.journal() == [("drive_bounded", {"distance_m": 0.2}, "SUCCEEDED", "OK")]
    assert abs(rig.sim.odometry()[0] - 0.2) < 0.02
    out = rig.approve(rig.command("Back up"))   # blind: the server allows it only slowly and at most 0.20 m
    assert out["outcome"] == "success", out
    assert rig.journal()[1:] == [("drive_bounded", {"distance_m": -0.2}, "SUCCEEDED", "OK")]
    assert abs(rig.sim.odometry()[0]) < 0.03


def test_turn_left_is_counter_clockwise_and_turn_right_clockwise(rig):
    assert rig.approve(rig.command("Turn left"))["outcome"] == "success"
    assert abs(math.degrees(rig.sim.odometry()[2]) - 45) < 3
    assert rig.approve(rig.command("Turn right"))["outcome"] == "success"
    assert abs(math.degrees(rig.sim.odometry()[2])) < 3
    assert rig.journal() == [("turn_bounded", {"angle_deg": 45.0}, "SUCCEEDED", "OK"),
                             ("turn_bounded", {"angle_deg": -45.0}, "SUCCEEDED", "OK")]


def test_stop_stops_a_drive_in_progress_and_stays_stopped(rig):
    p = rig.command("Move forward 50 centimeters")
    assert p["plan"]["steps"] == [{"op": "drive_bounded", "args": {"distance_m": 0.5}}]
    result = {}
    run = threading.Thread(target=lambda: result.update(rig.approve(p)))
    run.start()
    wait_until(lambda: rig.sim.velocity()[0] > 0, 15)                   # the robot is driving
    stopped = rig.command("Stop")
    run.join(30)
    assert stopped["kind"] == "stopped" and stopped["stopped"] is True
    assert result["outcome"] == "aborted", result
    assert rig.journal() == [("drive_bounded", {"distance_m": 0.5}, "ABORTED", "E_ESTOP")]
    assert rig.service.safety.estop_latched and rig.sim.velocity() == (0.0, 0.0)
    assert rig.sim.odometry()[0] < 0.45                                 # it did not finish the 0.5 m
    again = rig.approve(rig.command("Move forward"))                    # latched until a person clears it
    assert again["outcome"] == "aborted" and len(rig.journal()) == 1


def test_bring_me_my_pills_picks_tag_5_and_places_it_at_tag_15(rig):
    # Where things are comes from memory: this is the sighting the overhead camera would post.
    call(rig.brain + "/observations/tag", {"tag_id": 5, "zone": "shelf_B"})
    p = rig.command("Bring me my pills")
    assert [(s["op"], s["args"]) for s in p["plan"]["steps"]] == [
        ("locate_tag", {"object_id": "pill_bottle"}), ("pick_tagged_object", {"object_id": "pill_bottle"}),
        ("place_at_zone", {"zone": "patient"}), ("speak", {"text": "Done. Pill bottle at patient."})]
    out = rig.approve(p)
    assert out["outcome"] == "success", out
    assert rig.journal() == [("pick_tagged", {"tag_id": 5}, "SUCCEEDED", "OK"),
                             ("place_at", {"zone_tag_id": 15}, "SUCCEEDED", "OK")]
    assert rig.sim.held is None and math.hypot(rig.sim.tags[5].x - 1.6, rig.sim.tags[5].y - 0.4) < 0.4


def test_a_rejected_plan_never_reaches_the_robot(rig):
    p = rig.command("Move forward")
    out = call(f"{rig.brain}/operations/{p['operation_id']}/reject", {"rejected_by": "patient", "reason": "thumbs down"})
    assert out["outcome"] == "cancelled"
    with pytest.raises(urllib.error.HTTPError) as late:
        rig.approve(p)
    assert late.value.code == 409
    assert rig.journal() == [] and rig.sim.odometry()[:2] == (0.0, 0.0)


def test_without_a_recent_sighting_the_pill_fetch_stops_before_the_robot_moves(rig):
    out = rig.approve(rig.command("Bring me my pills"))
    assert out["outcome"] == "failed" and "has not seen pill_bottle recently" in out["failure"], out
    assert rig.journal() == [] and rig.sim.odometry()[:2] == (0.0, 0.0)
