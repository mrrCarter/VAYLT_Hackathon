"""The robot presets a voice request can select.

A preset is a short, fixed request that maps to steps made only of actions in
engram/capabilities.py. The intent model may pick one by name; it never writes
steps. The steps still go through Engine.propose, the checker and a person's
digest-bound approval, like every other plan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..assist import COMMANDS, STEP_M, TURN_DEG
from ..capabilities import CAPABILITIES


@dataclass(frozen=True)
class Preset:
    name: str
    request: str              # canonical request text sent to Engine.propose
    steps: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]
    description: str          # what the intent model is told this preset means

    def plan_steps(self) -> list[dict[str, Any]]:
        return [{"op": op, "args": dict(args)} for op, args in self.steps]


def _assist(request: str) -> tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]:
    """An assist command's fixed steps (engram/assist.py), in a preset's frozen form."""
    return tuple((s["op"], tuple(s["args"].items())) for s in COMMANDS[request])


_STEP = f"{STEP_M * 100:g} cm"
_TURN = f"{TURN_DEG:g} degrees"

PRESETS: dict[str, Preset] = {
    p.name: p
    for p in [
        Preset("open_gripper", "Open the gripper", (("gripper_open", ()),),
               "Open the robot's gripper (also called the claw, hand or grip), or release what it holds."),
        Preset("close_gripper", "Close the gripper", (("gripper_close", ()),),
               "Close the robot's gripper (also called the claw, hand or grip)."),
        Preset("look_around", "Look around", (("observe", ()),),
               "Look around with the camera and update what the robot knows about where things are."),
        # ENGRAM Assist (engram/assist.py): the same canonical requests and steps the engine plans.
        Preset("move_forward", "Move forward", _assist("Move forward"),
               f"Drive the robot straight forward one short step ({_STEP}). Only when no distance is said."),
        Preset("move_back", "Move back", _assist("Move back"),
               f"Drive the robot straight backward one short step ({_STEP}), also called back up. "
               "Only when no distance is said."),
        Preset("turn_left", "Turn left", _assist("Turn left"),
               f"Turn the robot in place to the left by {_TURN}. Only when no angle is said."),
        Preset("turn_right", "Turn right", _assist("Turn right"),
               f"Turn the robot in place to the right by {_TURN}. Only when no angle is said."),
        Preset("pick_up", "Pick it up", _assist("Pick it up"),
               "Pick up whatever is in front of the claw: lower the arm, close the claw and lift it. "
               "Not for fetching or bringing a named object."),
        Preset("put_down", "Put it down", _assist("Put it down"),
               "Put down what the claw is holding: lower the arm, open the claw and lift the arm again."),
    ]
}


def _validate() -> None:
    """Every preset step must be an existing capability with exactly its arguments."""
    for p in PRESETS.values():
        for op, args in p.steps:
            cap = CAPABILITIES.get(op)
            if cap is None or op == "stop":
                raise ValueError(f"preset {p.name}: '{op}' is not an allowed preset action")
            if set(dict(args)) != set(cap.args):
                raise ValueError(f"preset {p.name}: {op} needs arguments {sorted(cap.args)}")


_validate()


def voice_presets_enabled() -> bool:
    """ENGRAM_VOICE_PRESETS: on unless it is 0, false, no or off. The one reading of the setting, used by
    the HTTP API (engram/api.py) and by VoiceIntegration alike."""
    return os.environ.get("ENGRAM_VOICE_PRESETS", "1").strip().lower() not in ("0", "false", "no", "off")


def register_presets(engine: Any) -> None:
    """Let the engine plan each preset's canonical request ("Open the gripper") as its fixed steps.

    Uses Engine.register_fixed_request: only planning is skipped. Refusal rules, the checker and a
    person's digest-bound approval still apply to every proposal. Used by VoiceIntegration and,
    at startup, by the HTTP API (engram/api.py).
    """
    for p in PRESETS.values():
        engine.register_fixed_request(p.request, p.plan_steps())
