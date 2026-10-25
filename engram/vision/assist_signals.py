"""ENGRAM Assist: what a patient's hand signal makes the robot do.

A patient who cannot speak makes a hand signal at the robot's camera. A classifier names the
signal in each frame (the intent model, or MediaPipe hand landmarks on the laptop) and this
module decides what to do about it. It is plain logic, with no camera, network, clock or
threads of its own, so every rule below is tested without a robot (tests/test_assist_signals.py).
scripts/assist_gestures.py is the loop that feeds it frames and carries out its decisions.

Signals: open_palm, thumbs_up, thumbs_down, pointing, fist, peace, waving (or "none").
Which signal does what comes from a JSON map the team edits (assist_signals.default.json):
    open_palm -> stop, thumbs_up -> approve, thumbs_down -> cancel,
    peace -> command "Bring me my pills", fist -> command "Pick it up", the rest -> none.

The safety rules. The map chooses which signal does what; it cannot change these.

1. STOP is immediate. One frame with a stop signal at or above min_probability sends STOP:
   no hold, no cooldown, and not held back by a request that is still running. Every further
   such frame sends it again. A needless stop costs one re-arm; a missed stop can hurt someone.
2. Every other signal must be HELD: the same signal in hold_frames classifications in a row
   (default 2) spanning at least hold_seconds (default 0.6). A frame with no signal, a failed
   classification, or a probability under min_probability (default 0.6) starts the hold over.
   One hold makes at most one action: to repeat, lower the hand and show the signal again.
3. After any action (a stop too) other signals are ignored for cooldown_s (default 3 s), and
   while the last request to the brain is still running. A signal ignored that way is
   dropped, not saved for later: show it again.
4. A thumbs-up approves only when exactly ONE plan is waiting, it was proposed less than 60 s
   ago by the brain's own clock, and no thumbs-down cancelled it. It approves that plan's
   digest and nothing else. Otherwise nothing is approved and the reason is logged.
5. The brain has no endpoint to reject a plan, so a thumbs-down cannot remove one. It says
   "Okay, cancelled", approves nothing, and remembers the plans waiting at that moment so no
   later thumbs-up approves them. If the waiting plans cannot be read, thumbs-up approvals
   stay off until any plan that was waiting then is too old to approve.
6. Anything malformed (an unknown signal name, a bad probability, a broken map) is no signal,
   or keeps the loop from starting. It never turns into an action.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SIGNALS = ("open_palm", "thumbs_up", "thumbs_down", "pointing", "fist", "peace", "waving")
NO_SIGNAL = "none"
ACTIONS = ("stop", "approve", "cancel", "command", "none")
DEFAULT_MAP = Path(__file__).with_name("assist_signals.default.json")

# Defaults the map may change, within the limits parse_map checks.
HOLD_FRAMES = 2
HOLD_SECONDS = 0.6
MIN_PROBABILITY = 0.6
COOLDOWN_S = 3.0

# Fixed: not in the map.
APPROVE_MAX_AGE_S = 60.0  # a thumbs-up approves only a plan proposed less than this long ago
DATE_SLACK_S = 2.0        # the brain's Date header is whole seconds, refreshed about once a second
MAX_COMMAND_CHARS = 500   # the brain's limit for POST /commands
MAX_SAY_CHARS = 200

_OPERATION_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


# ---------------------------------------------------------------------------
# The map: which signal does what
# ---------------------------------------------------------------------------
class MapError(ValueError):
    """The signal map can't be used. The loop refuses to start rather than guess."""


@dataclass(frozen=True)
class Action:
    kind: str       # one of ACTIONS
    text: str = ""  # command only: the words sent to the brain, as if the patient had said them
    say: str = ""   # optional: said out loud on the laptop when the action is carried out


NO_ACTION = Action("none")


@dataclass(frozen=True)
class SignalMap:
    actions: dict[str, Action]  # an entry for every name in SIGNALS
    hold_frames: int = HOLD_FRAMES
    hold_seconds: float = HOLD_SECONDS
    min_probability: float = MIN_PROBABILITY
    cooldown_s: float = COOLDOWN_S

    def action(self, signal: str) -> Action:
        return self.actions.get(signal, NO_ACTION)


def load_map(path: str | Path = DEFAULT_MAP) -> SignalMap:
    """Read and check a signal map file. Raises MapError, naming the problem."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise MapError(f"cannot read the signal map {path}: {e.strerror or e}") from None
    try:
        data = json.loads(text, object_pairs_hook=_unique_keys)
    except MapError:
        raise
    except ValueError as e:
        raise MapError(f"the signal map {path} is not valid JSON: {e}") from None
    return parse_map(data)


def parse_map(data: Any) -> SignalMap:
    """Check a map already read from JSON. assist_signals.default.json shows the shape."""
    if not isinstance(data, dict):
        raise MapError("the signal map must be a JSON object")
    _known_keys(data, ("about", "signals", "hold_frames", "hold_seconds", "min_probability", "cooldown_s"),
                "the map")
    entries = data.get("signals")
    if not isinstance(entries, dict):
        raise MapError('the map needs "signals", e.g. {"signals": {"open_palm": {"action": "stop"}}}')
    actions = {name: NO_ACTION for name in SIGNALS}
    for name, entry in entries.items():
        if name not in SIGNALS:
            raise MapError(f'unknown signal "{name}"; the signals are {", ".join(SIGNALS)}')
        actions[name] = _parse_action(name, entry)
    if not any(a.kind == "stop" for a in actions.values()):
        raise MapError('no signal stops the robot; keep "open_palm": {"action": "stop"}')
    return SignalMap(
        actions,
        hold_frames=int(_number(data, "hold_frames", HOLD_FRAMES, 2, 30, whole=True)),
        hold_seconds=float(_number(data, "hold_seconds", HOLD_SECONDS, 0.0, 10.0)),
        min_probability=float(_number(data, "min_probability", MIN_PROBABILITY, 0.5, 1.0)),
        cooldown_s=float(_number(data, "cooldown_s", COOLDOWN_S, 0.0, 60.0)),
    )


def _parse_action(name: str, entry: Any) -> Action:
    if not isinstance(entry, dict):
        raise MapError(f'"{name}" must be an object, e.g. {{"action": "none"}}')
    _known_keys(entry, ("about", "action", "text", "say"), f'"{name}"')
    kind = entry.get("action")
    if kind not in ACTIONS:
        raise MapError(f'"{name}": "action" must be one of {", ".join(ACTIONS)}, not {json.dumps(kind)}')
    text = _words(entry, "text", name, MAX_COMMAND_CHARS)
    say = _words(entry, "say", name, MAX_SAY_CHARS)
    if (kind == "command") != bool(text):
        raise MapError(f'"{name}": a command needs "text" (the words to send), and only a command has it')
    if kind == "none" and say:
        raise MapError(f'"{name}": "none" does nothing, so it cannot "say" anything')
    return Action(kind, text, say)


def _words(entry: dict, key: str, name: str, limit: int) -> str:
    value = entry.get(key, "")
    if not isinstance(value, str) or (key in entry and not value.strip()) or len(value) > limit:
        raise MapError(f'"{name}": "{key}" must be some words, at most {limit} characters')
    return value.strip()


def _known_keys(obj: dict, allowed: tuple[str, ...], where: str) -> None:
    for key in obj:
        if key not in allowed:
            raise MapError(f'{where}: unknown key "{key}" (allowed: {", ".join(allowed)})')


def _number(data: dict, key: str, default: float, lo: float, hi: float, whole: bool = False) -> float:
    value = data.get(key, default)
    ok = isinstance(value, int) if whole else isinstance(value, (int, float))
    if isinstance(value, bool) or not ok or not math.isfinite(value) or not lo <= value <= hi:
        kind = "a whole number" if whole else "a number"
        raise MapError(f'"{key}" must be {kind} from {lo:g} to {hi:g}, not {json.dumps(value)}')
    return value


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise MapError(f'"{key}" appears twice in the signal map; keep one')
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# One classification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Reading:
    """One frame's classification. signal is one of SIGNALS or NO_SIGNAL."""

    signal: str
    probability: float
    problem: str = ""  # why this frame counts as no signal, when something went wrong


def no_signal(problem: str = "") -> Reading:
    return Reading(NO_SIGNAL, 0.0, problem)


def reading_from(result: Any) -> Reading:
    """Check one classifier answer. Anything malformed counts as no signal (fail closed)."""
    if not isinstance(result, dict):
        return no_signal(f"the classifier answered {type(result).__name__}, not a dict")
    signal, p = result.get("signal"), result.get("probability")
    if not isinstance(signal, str) or signal not in (*SIGNALS, NO_SIGNAL):
        return no_signal(f"unknown signal {repr(signal)[:60]}")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
        return no_signal(f"bad probability {repr(p)[:30]} for {signal}")
    return Reading(signal, float(p))


# ---------------------------------------------------------------------------
# Rules 1 to 3: hold, cooldown, STOP first
# ---------------------------------------------------------------------------
class SignalHold:
    """Rule 2. update() returns the signal once, on the frame its hold is complete, and not
    again until the hold is broken."""

    def __init__(self, frames: int = HOLD_FRAMES, seconds: float = HOLD_SECONDS,
                 min_probability: float = MIN_PROBABILITY):
        self.frames, self.seconds, self.min_probability = frames, seconds, min_probability
        self.reset()

    def reset(self) -> None:
        self.signal: str | None = None
        self.count = 0
        self.since = 0.0
        self.spent = False

    def update(self, reading: Reading, now: float) -> str | None:
        if reading.signal == NO_SIGNAL or reading.probability < self.min_probability:
            self.reset()
            return None
        if reading.signal != self.signal:
            self.reset()
            self.signal, self.since = reading.signal, now
        self.count += 1
        if self.spent or self.count < self.frames or now - self.since < self.seconds:
            return None
        self.spent = True
        return self.signal


@dataclass(frozen=True)
class Decision:
    """What to do about one recognised signal. go=False: recognised, not acted on (see why)."""

    signal: str
    probability: float
    action: Action
    go: bool
    why: str = ""
    repeat: bool = False  # a STOP sent again because the stop signal is still up


class Decider:
    """Rules 1 to 3: one classification in, at most one Decision out."""

    def __init__(self, signal_map: SignalMap):
        self.map = signal_map
        self.hold = SignalHold(signal_map.hold_frames, signal_map.hold_seconds, signal_map.min_probability)
        self.quiet_until = -math.inf  # end of the cooldown after the last action
        self._stopping = False        # the previous frame sent a STOP

    def decide(self, reading: Reading, now: float, busy: bool = False) -> Decision | None:
        """now: when the frame was taken, in seconds (monotonic). busy: the last request to the
        brain has not finished yet."""
        action = self.map.action(reading.signal)
        if action.kind == "stop" and reading.probability >= self.map.min_probability:
            self.hold.reset()
            repeat, self._stopping = self._stopping, True
            self.quiet_until = now + self.map.cooldown_s
            return Decision(reading.signal, reading.probability, action, True, repeat=repeat)
        self._stopping = False
        held = self.hold.update(reading, now)
        if held is None:
            return None
        action = self.map.action(held)
        if action.kind == "none":
            return Decision(held, reading.probability, action, False, "this signal does nothing in the map")
        if now < self.quiet_until:
            return Decision(held, reading.probability, action, False,
                            f"ignored: {self.quiet_until - now:.1f} s of the pause after the last action left;"
                            " show it again")
        if busy:
            return Decision(held, reading.probability, action, False,
                            "ignored: the last request to the brain is still running; show it again")
        self.quiet_until = now + self.map.cooldown_s
        return Decision(held, reading.probability, action, True)


# ---------------------------------------------------------------------------
# Rules 4 and 5: what a thumbs-up may approve
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Approval:
    operation_id: str
    digest: str
    request: str
    age_s: float


class ApproveGuard:
    """Rules 4 and 5: what a thumbs-up may approve, and what a thumbs-down has turned down."""

    def __init__(self, max_age_s: float = APPROVE_MAX_AGE_S):
        self.max_age_s = max_age_s
        self.cancelled: set[str] = set()
        self.locked_until = -math.inf

    def cancel(self, pending: Any, now: float) -> list[str]:
        """A thumbs-down. pending: GET /operations/pending, or None if it could not be read.
        now: this computer's monotonic clock. Returns the operation ids now cancelled."""
        if not isinstance(pending, list):
            # Whatever was waiting is unknown: no thumbs-up approvals until all of it is too old.
            self.locked_until = max(self.locked_until, now + self.max_age_s + DATE_SLACK_S)
            return []
        ids = [p["operation_id"] for p in pending if isinstance(p, dict) and isinstance(p.get("operation_id"), str)]
        self.cancelled.update(ids)
        return ids

    def check(self, pending: Any, brain_now: float | None, now: float) -> tuple[Approval | None, str]:
        """(the one plan a thumbs-up may approve, "") or (None, why not).

        pending: GET /operations/pending. brain_now: the brain's clock when it answered (from its
        Date header, rounded up), or None if it did not say. now: this computer's monotonic clock.
        """
        if now < self.locked_until:
            return None, (f"thumbs-up approvals are off for {self.locked_until - now:.1f} s more: a thumbs-down"
                          " could not read which plans were waiting")
        if not isinstance(pending, list):
            return None, "the brain's list of waiting plans is not a list"
        if len(pending) != 1:
            return None, ("no plan is waiting for approval" if not pending else
                          f"{len(pending)} plans are waiting; a thumbs-up approves only when exactly one is"
                          " (use the iPad)")
        plan = pending[0]
        if not isinstance(plan, dict):
            return None, "the waiting plan is not a JSON object"
        op_id, plan_digest, proposed_at = plan.get("operation_id"), plan.get("digest"), plan.get("proposed_at")
        if not isinstance(op_id, str) or not _OPERATION_ID.fullmatch(op_id):
            return None, "the waiting plan has no usable operation id"
        if not isinstance(plan_digest, str) or not _DIGEST.fullmatch(plan_digest):
            return None, f"plan {op_id} has no valid digest"
        if op_id in self.cancelled:
            return None, f"plan {op_id} was cancelled with a thumbs-down"
        if isinstance(proposed_at, bool) or not isinstance(proposed_at, (int, float)) or not math.isfinite(proposed_at):
            return None, f"plan {op_id} does not say when it was proposed"
        if brain_now is None or not math.isfinite(brain_now):
            return None, "the brain did not send its clock (no Date header), so the plan's age is unknown"
        age = brain_now - proposed_at
        if age < -DATE_SLACK_S:
            return None, f"plan {op_id} is dated {-age:.0f} s ahead of the brain's clock; not approving"
        if age >= self.max_age_s:
            return None, (f"plan {op_id} is {age:.0f} s old; a thumbs-up approves only plans less than"
                          f" {self.max_age_s:.0f} s old")
        return Approval(op_id, plan_digest, str(plan.get("request") or ""), round(max(age, 0.0), 1)), ""


# ---------------------------------------------------------------------------
# MediaPipe hands -> the same signal names
# ---------------------------------------------------------------------------
MEDIAPIPE_SIGNALS = {"Open_Palm": "open_palm", "Thumb_Up": "thumbs_up", "Thumb_Down": "thumbs_down",
                     "Closed_Fist": "fist", "Victory": "peace", "Pointing_Up": "pointing"}


def signal_from_hands(hands: Iterable[Any]) -> dict[str, Any]:
    """gestures.HandTracker hands -> an answer shaped like the intent model's classifier.

    A hand's signal is MediaPipe's own gesture label and score, where it has one for our signals
    (it has none for waving). Without one, gestures.classify's open-palm and pointing shapes count
    with probability 1: they are yes/no geometry, the shapes overhead_camera.py --gestures already
    stops on. A thumb seen from above can't be told up from down, so it is no signal. An open palm
    on any hand wins (STOP first); otherwise hands that disagree are no signal.
    """
    seen: list[tuple[str, float]] = []
    for hand in hands:
        signal = MEDIAPIPE_SIGNALS.get(getattr(hand, "builtin", ""))
        p = float(getattr(hand, "score", 0.0) or 0.0)
        if signal is None and getattr(hand, "label", "") in ("open_palm", "pointing"):
            signal, p = hand.label, 1.0
        if signal is not None:
            seen.append((signal, p))
    palms = [p for s, p in seen if s == "open_palm"]
    if palms:
        signal, p = "open_palm", max(palms)
    elif seen and len({s for s, _ in seen}) == 1:
        signal, p = seen[0][0], max(p for _, p in seen)
    else:
        signal, p = NO_SIGNAL, 0.0
    return {"signal": signal, "probability": p, "confidence": p, "probs": {signal: p}, "latency_ms": 0}
