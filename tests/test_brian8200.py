import base64
import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from engram.robot.brian8200 import BrianRobot, read_secret

SECRET = "test-secret"


class FakeBrian:
    """Checks signatures like Brian's robot/auth.py and answers like his server."""

    def __init__(self):
        self.calls, self.stops, self.beats = [], 0, 0
        self.reply = {}  # op -> (status, code) or (status, code, data)
        self.hooks = {}  # op -> called when that op arrives, before the reply (e.g. a stop pressed meanwhile)

    def verify(self, body):
        tok = body.get("authorization") or ""
        b, sig = tok.rsplit(".", 1)
        assert hmac.compare_digest(sig, hmac.new(SECRET.encode(), b.encode(), hashlib.sha256).hexdigest())
        p = json.loads(base64.urlsafe_b64decode(b + "=" * (-len(b) % 4)))
        canon = json.dumps(body["args"], sort_keys=True, separators=(",", ":"))
        assert p["op"] == body["op"] and p["operation_id"] == body["operation_id"]
        assert p["args_sha256"] == hashlib.sha256(canon.encode()).hexdigest()

    def handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"busy": None, "gripper": "unknown", "safety": {"estop_latched": fake.stops > 0},
                                 "driver": {"battery_v": 8.1}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                if self.path == "/robot/execute":
                    if body["op"] != "observe":
                        fake.verify(body)
                    fake.calls.append((body["op"], body["args"]))
                    status, code, *data = fake.reply.get(body["op"], ("SUCCEEDED", None))
                    if body["op"] in fake.hooks:
                        fake.hooks[body["op"]]()
                    self._send(200 if status != "REJECTED" else 409,
                               {"status": status, "code": code, "message": "m", "data": data[0] if data else {}})
                elif self.path == "/robot/stop":
                    fake.stops += 1
                    self._send(200, {"stopped": True})
                elif self.path == "/robot/heartbeat":
                    fake.beats += 1
                    self._send(200, {"ok": True})
                else:
                    self._send(200, {})

            def log_message(self, *a):
                pass

        return H


@pytest.fixture
def brian(engine):
    fake = FakeBrian()
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    robot = BrianRobot(f"http://127.0.0.1:{server.server_address[1]}", engine.memory, SECRET,
                       {"home": 10, "shelf_A": 11, "shelf_B": 12, "ppe_station": 13, "bench": 14})
    yield robot, fake
    server.shutdown()


def test_maps_and_signs_ops(brian):
    robot, fake = brian
    assert robot.execute("drive_bounded", {"distance_m": 0.2}).status == "done"
    assert robot.execute("turn_bounded", {"degrees": 90}).status == "done"
    assert robot.execute("arm_pose", {"pose": "reach_down"}).status == "done"
    assert robot.execute("pick_tagged_object", {"object_id": "beaker"}).status == "done"
    assert robot.execute("place_at_zone", {"zone": "bench"}).status == "done"
    assert fake.calls == [("drive_bounded", {"distance_m": 0.2}), ("turn_bounded", {"angle_deg": 90.0}),
                          ("arm_pose", {"pose": "reach"}),
                          ("detect_tag", {"tag_id": 4}), ("pick_tagged", {"tag_id": 4}),        # in view: no search
                          ("detect_tag", {"tag_id": 14}), ("place_at", {"zone_tag_id": 14})]


def test_unsensed_moves_count_but_unknown_pick_stops(brian):
    robot, fake = brian
    fake.reply = {"drive_bounded": ("UNKNOWN", "E_UNVERIFIED"), "pick_tagged": ("UNKNOWN", "E_GRASP_UNVERIFIED")}
    r = robot.execute("drive_bounded", {"distance_m": 0.1})
    assert r.status == "done" and "not sensor checked" in r.detail
    assert robot.execute("pick_tagged_object", {"object_id": "beaker"}).status == "unknown"
    assert fake.stops == 1


def test_rejected_and_unmappable_fail_without_moving(brian):
    robot, fake = brian
    fake.reply = {"gripper_close": ("REJECTED", "E_AUTH")}
    assert robot.execute("gripper_close", {}).status == "failed"
    out = robot.execute("vacuum_floor", {})
    assert out.status == "failed" and "no action" in out.detail
    assert robot.execute("arm_pose", {"pose": "wave"}).status == "failed"
    assert [c[0] for c in fake.calls] == ["gripper_close"]


def ops(fake):
    return [op for op, _ in fake.calls]


HIDDEN = {"detect_tag": ("FAILED", "E_TAG_NOT_FOUND")}          # the robot's camera does not see the tag now
SEARCH_4 = ("search_tag", {"tag_id": 4, "max_turn_deg": 360.0})


def test_locate_uses_a_fresh_sighting_then_the_robot_camera_and_never_moves(brian, engine):
    robot, fake = brian
    assert robot.execute("locate_tag", {"object_id": "beaker"}).status == "done" and fake.calls == []
    engine.memory.freshness_s = 1e-9   # no fresh sighting any more: there is no overhead camera
    fake.reply = {"detect_tag": ("SUCCEEDED", "OK", {"detection": {"distance_m": 0.62, "bearing_deg": -4.0}})}
    seen = robot.execute("locate_tag", {"object_id": "beaker"})
    assert seen.status == "done" and "sees tag 4 at 0.62 m, -4.0 deg" in seen.detail
    fake.reply = HIDDEN
    hidden = robot.execute("locate_tag", {"object_id": "beaker"})
    assert hidden.status == "done" and "not in view now; the robot will turn to look for it" in hidden.detail
    fake.reply = {"detect_tag": ("FAILED", "E_CAMERA")}
    assert robot.execute("locate_tag", {"object_id": "beaker"}).status == "failed"   # cannot look is not "not there"
    assert fake.calls == [("detect_tag", {"tag_id": 4})] * 3 and fake.stops == 0


def test_a_pick_with_the_tag_in_view_does_not_search(brian):
    robot, fake = brian
    assert robot.execute("pick_tagged_object", {"object_id": "beaker"}).status == "done"
    assert fake.calls == [("detect_tag", {"tag_id": 4}), ("pick_tagged", {"tag_id": 4})]


def test_a_pick_with_the_tag_hidden_turns_to_find_it_then_picks(brian):
    robot, fake = brian
    fake.reply = HIDDEN
    out = robot.execute("pick_tagged_object", {"object_id": "beaker"})
    assert out.status == "done", out
    assert fake.calls == [("detect_tag", {"tag_id": 4}), SEARCH_4, ("pick_tagged", {"tag_id": 4})]
    assert fake.beats > 0 and fake.stops == 0                    # heartbeats ran through the step


def test_a_search_that_finds_nothing_fails_the_step_and_never_picks(brian):
    robot, fake = brian
    fake.reply = {**HIDDEN, "search_tag": ("FAILED", "E_TAG_NOT_FOUND")}
    out = robot.execute("pick_tagged_object", {"object_id": "beaker"})
    assert out.status == "failed" and "looked for tag 4: FAILED E_TAG_NOT_FOUND" in out.detail
    assert ops(fake) == ["detect_tag", "search_tag"] and fake.stops == 0


def test_a_stop_latched_between_the_search_and_the_pick_means_no_pick(brian):
    robot, fake = brian
    fake.reply = HIDDEN
    fake.hooks = {"search_tag": lambda: setattr(fake, "stops", fake.stops + 1)}   # Stop pressed while it turned
    out = robot.execute("pick_tagged_object", {"object_id": "beaker"})
    assert out.status == "failed" and "stop pressed after the robot found tag 4" in out.detail
    assert ops(fake) == ["detect_tag", "search_tag"]


def test_a_search_cut_short_is_unknown_stops_the_robot_and_is_never_retried(brian):
    robot, fake = brian
    fake.reply = {**HIDDEN, "search_tag": ("ABORTED", "E_ESTOP")}
    out = robot.execute("pick_tagged_object", {"object_id": "beaker"})
    assert out.status == "unknown" and ops(fake) == ["detect_tag", "search_tag"] and fake.stops == 1


def test_nothing_is_sent_while_the_robot_is_stopped(brian):
    robot, fake = brian
    fake.stops = 1
    for op, args in (("pick_tagged_object", {"object_id": "beaker"}), ("place_at_zone", {"zone": "bench"})):
        out = robot.execute(op, args)
        assert out.status == "failed" and "robot is stopped" in out.detail
    assert fake.calls == []


def test_a_place_turns_to_find_the_zone_marker_first(brian):
    robot, fake = brian
    fake.reply = HIDDEN
    assert robot.execute("place_at_zone", {"zone": "bench"}).status == "done"
    assert fake.calls == [("detect_tag", {"tag_id": 14}), ("search_tag", {"tag_id": 14, "max_turn_deg": 360.0}),
                          ("place_at", {"zone_tag_id": 14})]


def test_lost_robot_is_unknown(engine):
    robot = BrianRobot("http://127.0.0.1:9", engine.memory, SECRET, {}, timeout_s=0.5)
    assert robot.execute("drive_bounded", {"distance_m": 0.1}).status == "unknown"
    assert robot.state()["stopped"]


def test_read_secret(tmp_path):
    env = tmp_path / "robot.env"
    env.write_text("# robot\nENGRAM_ROBOT_SECRET=abc123\n")
    assert read_secret(env) == "abc123"
    raw = tmp_path / "key"
    raw.write_text("xyz\n")
    assert read_secret(raw) == "xyz"


def test_planner_ignores_qwen3_thinking():
    from engram.planner import extract_json

    assert extract_json('<think>maybe {"goal": "x"} no</think>\n{"goal": "bring", "steps": []}') == {"goal": "bring", "steps": []}
    assert extract_json("<think>never closed {\"a\": 1}") is None


def test_model_failure_is_reported_as_fallback(engine):
    from engram.planner import LlamaCppPlanner

    engine.planner = LlamaCppPlanner(engine.zones, "http://127.0.0.1:9/completion", timeout_s=0.5)
    out = engine.propose("bring the beaker to the bench")
    assert out["source"] == "fallback" and "model call failed" in out["planner_note"]
