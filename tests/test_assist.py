"""ENGRAM Assist (engram/assist.py, docs/ASSIST.md): fixed commands for a helper robot, and the pill fetch.

Every command plans exact steps that the checker accepts and that wait for a person's approval.
Amounts past a bound, negated or later wording, and questions never become a plan.
"""

import json
import os
import subprocess
import sys

import pytest

from conftest import ROOT
from engram.api import create_app
from engram.assist import COMMANDS, MAX_REVERSE_M, plan_request, register_assist
from engram.capabilities import ARM_POSES, CAPABILITIES, MAX_DRIVE_M, MAX_TURN_DEG
from engram.checker import check_plan
from engram.engine import EngineError
from engram.planner import find_zone
from engram.robot.brian8200 import BrianRobot
from engram.robot.mock import MockRobot
from engram.spark.presets import PRESETS, register_presets

OBSERVED_ONLY = [("observe", {})]   # the fixture's overhead scan: nothing else reached the robot
REACH = {"op": "arm_pose", "args": {"pose": "reach"}}
CARRY = {"op": "arm_pose", "args": {"pose": "carry"}}
EXPECTED = {
    "Move forward": [{"op": "drive_bounded", "args": {"distance_m": 0.2}}],
    "Move back": [{"op": "drive_bounded", "args": {"distance_m": -0.2}}],
    "Back up": [{"op": "drive_bounded", "args": {"distance_m": -0.2}}],
    "Turn left": [{"op": "turn_bounded", "args": {"degrees": 45.0}}],
    "Turn right": [{"op": "turn_bounded", "args": {"degrees": -45.0}}],
    "Pick it up": [REACH, {"op": "gripper_close", "args": {}}, CARRY],
    "Put it down": [REACH, {"op": "gripper_open", "args": {}}, CARRY],
}
PILLS = [{"op": "locate_tag", "args": {"object_id": "pill_bottle"}},
         {"op": "pick_tagged_object", "args": {"object_id": "pill_bottle"}},
         {"op": "place_at_zone", "args": {"zone": "patient"}},
         {"op": "speak", "args": {"text": "Done. Pill bottle at patient."}}]


@pytest.fixture
def assist(engine):
    register_assist(engine)
    return engine


def counts(engine):
    episodes = engine.conn.execute("SELECT COUNT(*) FROM episode").fetchone()[0]
    tasks = engine.conn.execute("SELECT COUNT(*) FROM prospective_task").fetchone()[0]
    return episodes, tasks, len(engine.robot.calls)


def steps_of(out):
    return out["plan"]["steps"]


# ---- arm_pose: capability, checker, mock robot, Brian's adapter -------------------------------

def test_arm_pose_is_a_physical_capability_naming_a_saved_pose():
    cap = CAPABILITIES["arm_pose"]
    assert cap.physical and cap.args == {"pose": str}
    assert ARM_POSES == {"home", "look", "reach", "carry"}


def test_checker_accepts_saved_poses_and_rejects_others(engine):
    def check(args):
        return check_plan({"steps": [{"op": "arm_pose", "args": args}]}, engine.memory, engine.zones)

    for pose in ("home", "look", "reach", "carry"):
        res = check({"pose": pose})
        assert res.ok and res.physical, res.errors
    assert "unknown arm pose 'wave'" in check({"pose": "wave"}).errors[0]
    assert "must be str" in check({"pose": 3}).errors[0]
    assert "missing" in check({}).errors[0]
    assert "does not take" in check({"pose": "reach", "speed": 1}).errors[0]


def test_mock_robot_moves_the_arm_only_to_saved_poses(arena):
    robot = MockRobot(arena)
    assert robot.state()["arm_pose"] == "home"
    assert robot.execute("arm_pose", {"pose": "reach"}).status == "done"
    assert robot.state()["arm_pose"] == "reach"
    assert robot.execute("arm_pose", {"pose": "wave"}).status == "failed"
    assert robot.state()["arm_pose"] == "reach"


def test_brian_robot_sends_the_assist_steps_with_his_names_and_signs(engine):
    robot = BrianRobot("http://127.0.0.1:9", engine.memory, "unused", {})
    assert robot._translate("drive_bounded", {"distance_m": 0.2}) == ("drive_bounded", {"distance_m": 0.2})
    assert robot._translate("drive_bounded", {"distance_m": -0.2}) == ("drive_bounded", {"distance_m": -0.2})
    # Brian's turn_bounded: + is left (counter-clockwise), so right is a negative angle.
    assert robot._translate("turn_bounded", {"degrees": 45.0}) == ("turn_bounded", {"angle_deg": 45.0})
    assert robot._translate("turn_bounded", {"degrees": -45.0}) == ("turn_bounded", {"angle_deg": -45.0})
    assert robot._translate("arm_pose", {"pose": "reach"}) == ("arm_pose", {"pose": "reach"})
    assert robot._translate("arm_pose", {"pose": "carry"}) == ("arm_pose", {"pose": "carry"})
    assert robot._translate("arm_pose", {"pose": "reach_down"}) == ("arm_pose", {"pose": "reach"})  # older name
    assert isinstance(robot._translate("arm_pose", {"pose": "wave"}), str)                          # never sent
    assert robot._translate("gripper_close", {}) == ("gripper_close", {})
    assert robot._translate("gripper_open", {}) == ("gripper_open", {})


# ---- the fixed commands --------------------------------------------------------------------

@pytest.mark.parametrize("text,command", [
    *((t, t) for t in EXPECTED),
    ("move forward.", "Move forward"), ("MOVE FORWARD!", "Move forward"), ("Please move forward", "Move forward"),
    ("Go forward", "Move forward"), ("Move forward a little", "Move forward"),
    ("Move backward", "Move back"), ("Go back", "Move back"), ("Back up, please.", "Back up"),
    ("Turn to the left", "Turn left"), ("Robot, turn right now", "Turn right"),
    ("Pick it up, please.", "Pick it up"), ("Could you pick that up?", "Pick it up"), ("Pick up", "Pick it up"),
    ("Set it down", "Put it down"), ("Put that down for me", "Put it down"),
])
def test_each_assist_command_plans_its_exact_steps_and_waits(assist, text, command):
    out = assist.command(text)
    assert out["kind"] == "proposal" and out["status"] == "proposed", out
    assert steps_of(out) == EXPECTED[command]
    assert out["check"]["ok"] and out["check"]["physical"] and out["source"] == "fixed" and out["planner_calls"] == 0
    assert assist.robot.calls == OBSERVED_ONLY                                       # proposed, not run
    assert assist.conn.execute("SELECT COUNT(*) FROM approval").fetchone()[0] == 0


@pytest.mark.parametrize("command", list(EXPECTED))
def test_an_assist_command_runs_only_after_a_person_approves_its_digest(assist, command):
    out = assist.command(command)
    with pytest.raises(EngineError, match="digest"):
        assist.approve_and_run(out["operation_id"], "0" * 64, "nurse")
    assert assist.robot.calls == OBSERVED_ONLY
    run = assist.approve_and_run(out["operation_id"], out["digest"], "nurse")
    assert run["outcome"] == "success"
    assert assist.robot.calls[1:] == [(s["op"], s["args"]) for s in EXPECTED[command]]


def test_pick_it_up_then_put_it_down_on_the_mock(assist):
    for command, claw_open in (("Pick it up", False), ("Put it down", True)):
        out = assist.command(command)
        assert assist.approve_and_run(out["operation_id"], out["digest"], "nurse")["outcome"] == "success"
        state = assist.robot.state()
        assert state["arm_pose"] == "carry" and state["gripper_open"] is claw_open


def test_the_parser_and_the_fixed_commands_agree():
    assert set(COMMANDS) == set(EXPECTED)
    for text, steps in EXPECTED.items():
        assert COMMANDS[text] == steps and plan_request(text) == steps


# ---- distances and angles ------------------------------------------------------------------

@pytest.mark.parametrize("text,op,arg,value", [
    ("Move forward 30 centimeters", "drive_bounded", "distance_m", 0.3),
    ("Move forward 0.5 meters", "drive_bounded", "distance_m", 0.5),               # the bound itself
    ("Move forward 50 cm", "drive_bounded", "distance_m", 0.5),
    ("Move forward 25cm", "drive_bounded", "distance_m", 0.25),
    ("Move 30 centimeters forward", "drive_bounded", "distance_m", 0.3),
    ("Move forward thirty centimeters", "drive_bounded", "distance_m", 0.3),
    ("Move forward 1 foot", "drive_bounded", "distance_m", 0.305),
    ("Move forward half a meter", "drive_bounded", "distance_m", 0.5),
    ("Can you move forward 20 centimeters?", "drive_bounded", "distance_m", 0.2),
    ("Move back 10 centimeters", "drive_bounded", "distance_m", -0.1),
    ("Back up 20 centimeters", "drive_bounded", "distance_m", -0.2),               # the reverse bound itself
    ("Go backwards 15 cm", "drive_bounded", "distance_m", -0.15),
    ("Turn left 90 degrees", "turn_bounded", "degrees", 90.0),
    ("Turn right 90 degrees", "turn_bounded", "degrees", -90.0),
    ("Turn left 180 degrees", "turn_bounded", "degrees", 180.0),                   # the bound itself
    ("Turn 30 degrees to the right", "turn_bounded", "degrees", -30.0),
    ("Turn left ninety degrees", "turn_bounded", "degrees", 90.0),
    ("Turn right forty five degrees", "turn_bounded", "degrees", -45.0),
    ("Turn left a hundred and eighty degrees", "turn_bounded", "degrees", 180.0),
    ("Turn left 90°", "turn_bounded", "degrees", 90.0),
    ("Turn right by 15", "turn_bounded", "degrees", -15.0),                       # a bare number is degrees
])
def test_explicit_amounts_within_the_bounds_plan_that_amount(assist, text, op, arg, value):
    out = assist.command(text)
    assert out["status"] == "proposed" and out["check"]["ok"], out
    assert steps_of(out) == [{"op": op, "args": {arg: value}}]
    assert assist.robot.calls == OBSERVED_ONLY


@pytest.mark.parametrize("text,says", [
    ("Move forward 80 centimeters", "at most 50 cm at a time, and you asked for 80 cm"),
    ("Move forward 1 meter", "at most 50 cm"),
    ("Move forward 51 cm", "at most 50 cm"),
    ("Move back 30 centimeters", "back up at most 20 cm"),
    ("Back up half a meter", "can see behind it"),
    ("Turn left 270 degrees", "at most 180 degrees at a time, and you asked for 270"),
    ("Turn right 181 degrees", "at most 180 degrees"),
    ("Turn left 0 degrees", "no turn at all"),
    ("Move forward 0 cm", "too short"),
    ("Move forward 30", "with a unit"),
    ("Move forward 5 steps", "centimeters or meters, not 'steps'"),
    ("Turn left 30 centimeters", "by degrees"),
])
def test_amounts_past_a_bound_or_unclear_are_refused_with_the_reason(assist, text, says):
    out = assist.command(text)
    assert out["kind"] == "proposal" and out["status"] == "refused" and says in out["reason"], out
    assert not out.get("plan") and assist.robot.calls == OBSERVED_ONLY
    row = assist.conn.execute("SELECT outcome, failure FROM episode WHERE id = ?", (out["operation_id"],)).fetchone()
    assert (row["outcome"], row["failure"]) == ("refused", out["reason"])


def test_the_limits_are_the_capability_bounds_and_the_robots_blind_reverse_limit():
    assert plan_request(f"Move forward {MAX_DRIVE_M * 100:g} cm")[0]["args"]["distance_m"] == MAX_DRIVE_M
    assert isinstance(plan_request(f"Move forward {MAX_DRIVE_M * 100 + 1:g} cm"), str)
    assert plan_request(f"Turn left {MAX_TURN_DEG:g} degrees")[0]["args"]["degrees"] == MAX_TURN_DEG
    assert isinstance(plan_request(f"Turn left {MAX_TURN_DEG + 1:g} degrees"), str)
    assert plan_request(f"Back up {MAX_REVERSE_M * 100:g} cm")[0]["args"]["distance_m"] == -MAX_REVERSE_M
    assert isinstance(plan_request(f"Back up {MAX_REVERSE_M * 100 + 1:g} cm"), str)
    # Never more than Brian's robot server allows backing up with its forward-only ultrasonic sensor.
    robot_cfg = json.loads((ROOT / "robot/engram_robot/configs/ultrasonic.json").read_text())
    assert MAX_REVERSE_M <= robot_cfg["blind_reverse_max_m"] and MAX_REVERSE_M <= MAX_DRIVE_M


# ---- what must never plan ------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Don't move forward", "Never pick it up", "Do not turn left", "No need to put it down",
    "Move forward after lunch", "Turn left in 5 minutes", "Put it down later",
])
def test_negated_or_later_assist_commands_never_plan(assist, text):
    out = assist.command(text)
    assert out["kind"] == "proposal" and out["status"] == "refused", out
    assert assist.robot.calls == OBSERVED_ONLY


def test_a_when_assist_command_is_scheduled_then_proposed_when_it_fires(assist):
    assert assist.command("When the nurse leaves, move forward")["kind"] == "scheduled"
    (fired,) = assist.fire_trigger("the nurse leaves")
    assert fired["proposal"]["status"] == "proposed" and steps_of(fired["proposal"]) == EXPECTED["Move forward"]
    assert assist.robot.calls == OBSERVED_ONLY


@pytest.mark.parametrize("text", ["Stop", "Stop moving forward", "Wait, pick it up", "Hold on, turn left"])
def test_stop_wins_over_every_assist_command(assist, text):
    assert assist.command(text)["kind"] == "stopped"
    assert assist.robot.state()["stopped"] and assist.robot.calls == OBSERVED_ONLY


@pytest.mark.parametrize("text", [
    "Where are my pills?", "Should I move forward?", "Can the robot turn left?", "Did you pick it up?",
    "Turn left?", "Where is my medicine",
])
def test_questions_are_answered_never_planned(assist, text):
    before = counts(assist)
    out = assist.command(text)
    assert out["kind"] == "answer" and "reply" in out
    assert counts(assist) == before


def test_where_are_my_pills_is_answered_from_memory(assist):
    out = assist.command("Where are my pills?")
    assert out["kind"] == "answer" and out["kinds"] == ["spatial"]
    assert out["reply"] == "Pill bottle is at shelf_B"


@pytest.mark.parametrize("text,picks", [
    ("Pick up the beaker", "beaker"),
    ("Move the beaker back to the bench", "beaker"),
    ("Pick up the pill bottle", "pill_bottle"),
])
def test_requests_that_name_an_object_still_go_to_the_planner(assist, text, picks):
    out = assist.command(text)
    assert out["status"] == "proposed" and out["source"] == "rules", out
    assert [s["args"]["object_id"] for s in steps_of(out) if s["op"] == "pick_tagged_object"] == [picks]


@pytest.mark.parametrize("text", [
    "Go back home", "Go ahead", "Turn on the light", "Turn around", "Back up the data", "Move forward to the bench",
    "Bring me my pills", "Move forward 30 centimeters and turn left",
])
def test_other_requests_are_not_assist_commands(text):
    assert plan_request(text) is None


# ---- "Bring me my pills": the planner's fetch, with names from the arena --------------------

@pytest.mark.parametrize("text", ["Bring me my pills", "Get my pills", "Bring me the pill bottle",
                                  "Please bring me my medicine"])
def test_bring_me_my_pills_plans_the_fetch_to_the_patient(assist, text):
    out = assist.command(text)
    assert out["kind"] == "proposal" and out["status"] == "proposed" and out["check"]["ok"], out
    assert steps_of(out) == PILLS and out["source"] == "rules"
    assert assist.robot.calls == OBSERVED_ONLY


def test_the_pill_fetch_runs_on_the_mock_only_after_approval(assist):
    out = assist.command("Bring me my pills")
    assert assist.approve_and_run(out["operation_id"], out["digest"], "nurse")["outcome"] == "success"
    assert assist.robot.state()["objects"]["pill_bottle"] == "patient"


def test_a_named_zone_wins_over_me_and_my(assist):
    to_bench = assist.command("Bring my pills to the bench")
    assert [s for s in steps_of(to_bench) if s["op"] == "place_at_zone"] == [
        {"op": "place_at_zone", "args": {"zone": "bench"}}]
    to_me = assist.command("Bring me Sample A")
    assert [s for s in steps_of(to_me) if s["op"] == "place_at_zone"] == [
        {"op": "place_at_zone", "args": {"zone": "patient"}}]


def test_aliases_match_whole_words_only(engine):
    zones, aliases = {"bench", "patient"}, {"me": "patient", "sofa": "nowhere"}
    assert find_zone("bring it to me", zones, aliases) == "patient"
    assert find_zone("bring the medium beaker", zones, aliases) == "bench"      # "medium" is not "me"
    assert find_zone("bring it to the sofa", zones, aliases) == "bench"         # an alias to no known zone
    assert [o.object_id for o in engine.memory.resolve_object("my pills")] == ["pill_bottle"]
    assert engine.memory.resolve_object("my pillow") == []                    # "pillow" is not "pill"


def test_the_arena_holds_the_pill_bottle_and_the_patient_tags(arena, engine):
    (pills,) = [o for o in arena["objects"] if o["id"] == "pill_bottle"]
    assert pills["tag"] == 5 and pills["graspable"] and "pills" in pills["aliases"]
    assert arena["zone_tags"]["15"] == "patient" and "patient" in arena["zones"]
    assert "me" in arena["zone_aliases"]["patient"]
    assert all(z in arena["zones"] for z in arena["zone_aliases"])
    object_tags = [o["tag"] for o in arena["objects"]]
    assert len(set(object_tags)) == len(object_tags) and not set(object_tags) & {int(t) for t in arena["zone_tags"]}
    # The zone tags reach Brian's adapter the way engram/api.py builds them.
    zone_tags = {zone: int(tag) for tag, zone in arena.get("zone_tags", {}).items()}
    robot = BrianRobot("http://127.0.0.1:9", engine.memory, "unused", zone_tags)
    assert robot._translate("pick_tagged_object", {"object_id": "pill_bottle"}) == ("pick_tagged", {"tag_id": 5})
    assert robot._translate("place_at_zone", {"zone": "patient"}) == ("place_at", {"zone_tag_id": 15})


# ---- the fast voice path and the HTTP API ---------------------------------------------------

@pytest.mark.parametrize("name,request_text", [
    ("move_forward", "Move forward"), ("move_back", "Move back"), ("turn_left", "Turn left"),
    ("turn_right", "Turn right"), ("pick_up", "Pick it up"), ("put_down", "Put it down"),
])
def test_assist_presets_carry_the_assist_commands(engine, name, request_text):
    preset = PRESETS[name]
    assert preset.request == request_text and preset.plan_steps() == EXPECTED[request_text]
    assert check_plan({"steps": preset.plan_steps()}, engine.memory, engine.zones).ok


def test_presets_alone_plan_the_assist_commands(engine):
    register_presets(engine)   # what VoiceIntegration does; no register_assist here
    for name in ("move_forward", "move_back", "turn_left", "turn_right", "pick_up", "put_down"):
        out = engine.command(PRESETS[name].request, who="voice")
        assert out["status"] == "proposed" and steps_of(out) == EXPECTED[PRESETS[name].request]
    assert engine.robot.calls == OBSERVED_ONLY


@pytest.mark.parametrize("flag", [None, "0"], ids=["presets on", "presets off"])
def test_the_api_plans_assist_commands_with_or_without_voice_presets(engine, monkeypatch, flag):
    from fastapi.testclient import TestClient

    if flag is None:
        monkeypatch.delenv("ENGRAM_VOICE_PRESETS", raising=False)
    else:
        monkeypatch.setenv("ENGRAM_VOICE_PRESETS", flag)
    client = TestClient(create_app(engine))
    for text, steps in (("Move forward", EXPECTED["Move forward"]),
                        ("Turn left 90 degrees", [{"op": "turn_bounded", "args": {"degrees": 90.0}}])):
        out = client.post("/commands", json={"text": text, "who": "patient"}).json()
        assert out["status"] == "proposed" and steps_of(out) == steps
    assert engine.robot.calls == OBSERVED_ONLY


def test_pose_names_come_from_the_environment():
    code = ("import json; from engram.assist import COMMANDS; from engram.capabilities import ARM_POSES; "
            "from engram.robot.brian8200 import POSE_NAMES; "
            "print(json.dumps([COMMANDS['Pick it up'], sorted(ARM_POSES), POSE_NAMES['low_reach']]))")
    env = {**os.environ, "ENGRAM_REACH_POSE": "low_reach", "ENGRAM_CARRY_POSE": "lift"}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    steps, poses, sent = json.loads(out.stdout)
    assert steps == [{"op": "arm_pose", "args": {"pose": "low_reach"}}, {"op": "gripper_close", "args": {}},
                     {"op": "arm_pose", "args": {"pose": "lift"}}]
    assert poses == ["home", "lift", "look", "low_reach"] and sent == "low_reach"
