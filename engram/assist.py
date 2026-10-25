"""ENGRAM Assist: short commands for a helper robot ("Move forward", "Pick it up").

For a patient who cannot get up. Each command becomes fixed steps made only of actions in
capabilities.py, and nothing here moves the robot: the steps go through Engine.propose like every
plan, so the refusal rules (negation, "later"), the checker and a person's digest-bound approval
still apply. "Stop" never gets here: the engine stops on a stop word before anything else.

    Move forward [N]           drive_bounded +0.20 m, or N up to 0.5 m
    Move back / Back up [N]    drive_bounded -0.20 m, or N up to 0.20 m (nothing senses behind it)
    Turn left / Turn right [N] turn_bounded +45 / -45 degrees (+ is left), or N up to 180
    Pick it up                 arm_pose reach, gripper_close, arm_pose carry
    Put it down                arm_pose reach, gripper_open, arm_pose carry

"Bring me my pills" is not here: it is the planner's ordinary fetch. The pill bottle, the
patient's place and their other names ("pills", "me") are in the arena file. See docs/ASSIST.md.
"""

from __future__ import annotations

import re
from typing import Any

from .capabilities import CARRY_POSE, MAX_DRIVE_M, MAX_TURN_DEG, REACH_POSE

STEP_M = 0.20        # "Move forward" / "Move back" when no distance is said
TURN_DEG = 45.0      # "Turn left" / "Turn right" when no angle is said
MIN_DRIVE_M = 0.01   # the robot server refuses shorter drives
# The robot's range sensor looks forward only, so backing up is blind. The robot server allows at
# most 0.20 m of that per drive (blind_reverse_max_m); a longer back-up is refused here first.
MAX_REVERSE_M = 0.20

Steps = list[dict[str, Any]]


def drive(distance_m: float) -> Steps:
    """+ forward, - back."""
    return [{"op": "drive_bounded", "args": {"distance_m": distance_m}}]


def turn(degrees: float) -> Steps:
    """+ left (counter-clockwise), - right: the robot server's turn_bounded convention."""
    return [{"op": "turn_bounded", "args": {"degrees": degrees}}]


def pick_up() -> Steps:
    """Lower the arm to what is in front of the claw, close the claw, lift it."""
    return [{"op": "arm_pose", "args": {"pose": REACH_POSE}}, {"op": "gripper_close", "args": {}},
            {"op": "arm_pose", "args": {"pose": CARRY_POSE}}]


def put_down() -> Steps:
    """Lower the arm, open the claw, lift the arm clear."""
    return [{"op": "arm_pose", "args": {"pose": REACH_POSE}}, {"op": "gripper_open", "args": {}},
            {"op": "arm_pose", "args": {"pose": CARRY_POSE}}]


# The canonical commands and their fixed steps. The voice presets (spark/presets.py) use these too.
COMMANDS: dict[str, Steps] = {
    "Move forward": drive(STEP_M),
    "Move back": drive(-STEP_M),
    "Back up": drive(-STEP_M),
    "Turn left": turn(TURN_DEG),
    "Turn right": turn(-TURN_DEG),
    "Pick it up": pick_up(),
    "Put it down": put_down(),
}

# ---- other wordings, and amounts ---------------------------------------------

_LEAD = [("hey", "robot"), ("ok", "robot"), ("okay", "robot"), ("robot",), ("engram",), ("hey",), ("ok",),
         ("okay",), ("can", "you"), ("could", "you"), ("would", "you"), ("will", "you"), ("please",), ("just",),
         ("now",), ("then",), ("and",)]
_TAIL = [("thank", "you"), ("thanks",), ("please",), ("now",), ("for", "me")]
_PICK = {("pick", "it", "up"), ("pick", "that", "up"), ("pick", "this", "up"), ("pick", "up")}
_PUT = {("put", "it", "down"), ("put", "that", "down"), ("put", "this", "down"), ("put", "down"),
        ("set", "it", "down"), ("set", "that", "down"), ("set", "this", "down"), ("set", "down")}
_DRIVE_VERBS = {"move", "go", "drive", "roll"}
# No "ahead": "Go ahead" usually means "yes, do it", and must not become a new plan to drive.
_DIRECTIONS = {"forward": 1.0, "forwards": 1.0, "back": -1.0, "backward": -1.0, "backwards": -1.0}
_TURN_VERBS = {"turn", "rotate"}
_SIDES = {"left": 1.0, "right": -1.0}
_A_LITTLE = [["a", "little", "bit"], ["a", "little"], ["a", "bit"], ["slightly"]]   # the default step
_METERS = {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
           "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01, "centimetre": 0.01, "centimetres": 0.01,
           "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001, "millimetre": 0.001, "millimetres": 0.001,
           "inch": 0.0254, "inches": 0.0254, "foot": 0.3048, "feet": 0.3048, "ft": 0.3048}
_DEGREES = {"degree", "degrees", "deg"}
_SMALL = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                     "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}


def _words(text: str) -> list[str]:
    """Lower case words and numbers ("0.3" stays one number, "30cm" is two words); "90°" is 90 degrees."""
    return re.findall(r"\d+(?:\.\d+)?|[a-z]+", text.lower().replace("°", " degrees "))


def _strip(words: list[str]) -> list[str]:
    """Drop politeness and address around the command: "Robot, please move forward now" -> move forward."""
    changed = True
    while changed:
        changed = False
        for lead in _LEAD:
            if len(words) > len(lead) and tuple(words[:len(lead)]) == lead:
                words, changed = words[len(lead):], True
                break
        for tail in _TAIL:
            if len(words) > len(tail) and tuple(words[-len(tail):]) == tail:
                words, changed = words[:-len(tail)], True
                break
    return words


def _number(words: list[str]) -> tuple[float, int] | None:
    """The number the words start with and how many words it took: "30", "0.5", "forty five",
    "a hundred and eighty", "half a" (meter). None when they do not start with a number."""
    if not words:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", words[0]):
        return float(words[0]), 1
    if words[0] == "half":
        return 0.5, 2 if words[1:2] in (["a"], ["an"]) else 1
    value, used = None, 0
    for w in words:
        if w in ("a", "an") and value is None:
            n = 1
        elif w in _SMALL or w in _TENS:
            n = _SMALL.get(w, _TENS.get(w, 0))
        elif w == "hundred" and value is not None:
            value, used = value * 100, used + 1
            continue
        elif w == "and" and value is not None and value >= 100:   # "a hundred and eighty"
            used += 1
            continue
        else:
            break
        value, used = n if value is None else value + n, used + 1
    if value is None:
        return None
    if words[used - 1] == "and":
        used -= 1
    return float(value), used


def _amount(words: list[str]) -> tuple[float, str | None] | None:
    """(number, unit word or None) when the words are exactly an amount: "30 centimeters", "by 90"."""
    if words[:1] == ["by"]:
        words = words[1:]
    found = _number(words)
    if found is None:
        return None
    value, used = found
    rest = words[used:]
    if len(rest) > 1:
        return None
    return value, rest[0] if rest else None


def _cm(meters: float) -> str:
    return f"{meters * 100:g} cm"


def _drive(sign: float, words: list[str]) -> Steps | str | None:
    way = "forward" if sign > 0 else "back"
    if not words or words in _A_LITTLE:
        return drive(sign * STEP_M)
    amount = _amount(words)
    if amount is None:
        return None   # "go back home": not a distance, so not ours
    value, unit = amount
    if unit is None:
        return f"How far? Say it with a unit, for example 'Move {way} 20 centimeters'."
    if unit not in _METERS:
        return f"I move by centimeters or meters, not '{unit}'. For example: 'Move {way} 20 centimeters'."
    meters = round(value * _METERS[unit], 3)
    if meters < MIN_DRIVE_M:
        return f"That is too short to drive: the shortest move is {_cm(MIN_DRIVE_M)}."
    if sign > 0 and meters > MAX_DRIVE_M:
        return f"I can move forward at most {_cm(MAX_DRIVE_M)} at a time, and you asked for {_cm(meters)}."
    if sign < 0 and meters > MAX_REVERSE_M:
        return (f"I can back up at most {_cm(MAX_REVERSE_M)} at a time, because nothing on the robot can see "
                f"behind it, and you asked for {_cm(meters)}.")
    return drive(sign * meters)


def _turn(sign: float, words: list[str]) -> Steps | str | None:
    if not words or words in _A_LITTLE:
        return turn(sign * TURN_DEG)
    amount = _amount(words)
    if amount is None:
        return None
    value, unit = amount
    if unit is not None and unit not in _DEGREES:
        return f"I turn by degrees, not '{unit}'. For example: 'Turn left 90 degrees'."
    degrees = round(value, 1)
    if degrees <= 0:
        return "That is no turn at all. For example: 'Turn left 90 degrees'."
    if degrees > MAX_TURN_DEG:
        return f"I can turn at most {MAX_TURN_DEG:g} degrees at a time, and you asked for {degrees:g}."
    return turn(sign * degrees)


def _side(words: list[str]) -> tuple[float, list[str]] | None:
    """The side and the rest: from "left 90", "to the right", or "90 degrees to the left"."""
    first = words[2:] if words[:2] == ["to", "the"] else words
    if first and first[0] in _SIDES:
        return _SIDES[first[0]], first[1:]
    if len(words) > 1 and words[-1] in _SIDES:
        rest = words[:-1]
        return _SIDES[words[-1]], rest[:-2] if rest[-2:] == ["to", "the"] else rest
    return None


def plan_request(request: str) -> Steps | str | None:
    """An assist command's steps, a reason to refuse one (an amount past a bound), or None.

    None means "not an assist command": the request goes on to the planner, so "Pick up the
    beaker" or "Go back home" are never taken over here.
    """
    words = _strip(_words(request))
    if tuple(words) in _PICK:
        return pick_up()
    if tuple(words) in _PUT:
        return put_down()
    if len(words) >= 2 and words[0] in _TURN_VERBS:
        side = _side(words[1:])
        return _turn(*side) if side else None
    if words[:2] == ["back", "up"]:
        return _drive(-1.0, words[2:])
    if len(words) >= 2 and words[0] in _DRIVE_VERBS:
        if words[1] in _DIRECTIONS:                                   # "move forward 30 cm"
            return _drive(_DIRECTIONS[words[1]], words[2:])
        if words[-1] in _DIRECTIONS and _amount(words[1:-1]):        # "move 30 cm forward"
            return _drive(_DIRECTIONS[words[-1]], words[1:-1])
    return None


def register_assist(engine: Any) -> None:
    """Let the engine plan the assist commands: the canonical ones as fixed requests
    (Engine.register_fixed_request), other wordings and amounts through Engine.register_request_parser.
    Only planning is skipped; see the module docstring."""
    for request, steps in COMMANDS.items():
        engine.register_fixed_request(request, steps)
    engine.register_request_parser(plan_request)
