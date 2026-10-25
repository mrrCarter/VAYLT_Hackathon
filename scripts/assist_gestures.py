"""ENGRAM Assist: the robot answers a patient's hand signals.

    python scripts/assist_gestures.py --robot http://star-pi.local:8200 --brain http://star-pi.local:8001
    python scripts/assist_gestures.py --webcam 0 --classifier mediapipe --dry-run
    python scripts/assist_gestures.py --images rehearsal/ --dry-run

Every 1/--rate seconds it takes a frame (the robot's camera, a webcam on this computer, or the
next .jpg in a folder), asks the classifier which hand signal the frame shows, and lets
engram/vision/assist_signals.py decide what to do. Which signal does what is the map
engram/vision/assist_signals.default.json. The safety rules are in assist_signals.py and
docs/ASSIST.md.

    --classifier intent      the intent model (engram/vision/intent_vision.py); needs
                             TYPESAFE_API_KEY and TYPESAFE_INTENT_MODEL in the environment
    --classifier mediapipe   MediaPipe hand landmarks on this computer (engram/vision/gestures.py)

Prints one JSON object per line: start, brain, signal (a signal was recognised), action (what was
done about it, or why not), say, error, exit; with --verbose also seen (every classification).
A bad frame, a failed classification or a brain that does not answer is logged and counts as
no signal; the loop goes on. Ctrl+C quits.
"""

from __future__ import annotations

import argparse
import contextlib
import email.utils
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engram.vision.assist_signals import (  # noqa: E402
    APPROVE_MAX_AGE_S, DATE_SLACK_S, DEFAULT_MAP, ApproveGuard, Decider, Decision, MapError, Reading,
    SignalMap, load_map, no_signal, reading_from, signal_from_hands,
)

WHO = "patient (hand signal)"   # approved_by and who, in the brain's records
INTENT_ENV = ("TYPESAFE_API_KEY", "TYPESAFE_INTENT_MODEL")
FRAME_TIMEOUT_S = 3.0
STOP_TIMEOUT_S = 3.0
READ_TIMEOUT_S = 3.0
COMMAND_TIMEOUT_S = 90.0        # the brain's model planner may take a minute
APPROVE_TIMEOUT_S = 300.0       # the brain answers an approve only when the plan has finished running
NO_REJECT_NOTE = ("the brain has no reject endpoint: the plan stays on the iPad, but no thumbs-up will"
                  " approve it")


class SetupError(Exception):
    """Something needed to start is missing. Reported once; the loop does not start."""


class FrameError(Exception):
    """This frame can't be used. It counts as no signal and the loop goes on."""


class EndOfFrames(Exception):
    """--images: every file has been played."""


class BrainError(Exception):
    """The brain did not answer, or answered with an error."""


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
class Log:
    """One JSON object per line. Thread safe; ASCII only, so any console can print it."""

    def __init__(self, out=None):
        self.out = out if out is not None else sys.stdout
        self._lock = threading.Lock()

    def __call__(self, event: str, **fields: Any) -> None:
        record = {"at": round(time.time(), 3), "event": event}
        record.update((k, v) for k, v in fields.items() if v is not None)
        line = json.dumps(record, default=str)
        with self._lock:
            print(line, file=self.out, flush=True)


def system_voice() -> Callable[[str], bool] | None:
    """Something that says a short sentence on this computer without waiting for it to finish,
    or None when this computer has no voice we know how to use."""
    if sys.platform == "win32":
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if not shell:
            return None
        script = ("Add-Type -AssemblyName System.Speech; "
                  "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($env:ENGRAM_ASSIST_SAY)")
        # The text goes in through the environment, never through the command line.
        return lambda text: _spawn([shell, "-NoProfile", "-NonInteractive", "-Command", script],
                                   env={**os.environ, "ENGRAM_ASSIST_SAY": text})
    choices = [["say"]] if sys.platform == "darwin" else [["espeak-ng", "--stdin"], ["espeak", "--stdin"],
                                                          ["spd-say", "-e"]]
    for cmd in choices:
        exe = shutil.which(cmd[0])
        if exe:
            argv = [exe, *cmd[1:]]
            return lambda text: _spawn(argv, stdin_text=text)  # the text goes in on stdin
    return None


def _spawn(cmd: list[str], env: dict | None = None, stdin_text: str | None = None) -> bool:
    p = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if stdin_text is not None:
        p.stdin.write(stdin_text.encode("utf-8"))
        p.stdin.close()
    return True


class Voice:
    """Says a sentence (if this computer can) and logs it either way."""

    def __init__(self, log: Log, speak: Callable[[str], bool] | None):
        self.log, self.speak = log, speak

    def __call__(self, text: str) -> None:
        spoken = False
        if self.speak is not None:
            try:
                spoken = bool(self.speak(text))
            except Exception:  # noqa: BLE001 - no voice is not a reason to stop
                spoken = False
        self.log("say", text=text, spoken=spoken)


# ---------------------------------------------------------------------------
# The brain (engram/api.py)
# ---------------------------------------------------------------------------
def brain_clock(response: httpx.Response) -> float | None:
    """The latest the brain's clock could have read when it answered, from its Date header.

    The header is whole seconds and uvicorn refreshes it about once a second, so the brain's
    clock may be up to DATE_SLACK_S ahead of it. Adding that makes a plan look, if anything,
    older than it is. None when there is no usable Date header.
    """
    try:
        when = email.utils.parsedate_to_datetime(response.headers["date"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        return None
    return when.timestamp() + DATE_SLACK_S


def _detail(r: httpx.Response) -> str:
    try:
        detail = r.json().get("detail", r.text)
    except (ValueError, AttributeError):
        detail = r.text
    return str(detail)[:200]


class Brain:
    """The calls this loop makes to the ENGRAM brain. STOP has its own connection, so it never
    queues behind a long request."""

    def __init__(self, url: str, client: httpx.Client | None = None):
        self.url = url.rstrip("/")
        self.http = client or httpx.Client(base_url=self.url, trust_env=False)
        self.stop_http = client or httpx.Client(base_url=self.url, trust_env=False)

    def _send(self, method: str, path: str, body: dict | None = None, timeout: float = READ_TIMEOUT_S,
              http: httpx.Client | None = None) -> httpx.Response:
        try:
            r = (http or self.http).request(method, path, json=body, timeout=timeout)
        except httpx.HTTPError as e:
            raise BrainError(f"{method} {path}: {type(e).__name__}: {e}") from None
        if r.status_code >= 400:
            raise BrainError(f"{method} {path}: HTTP {r.status_code}: {_detail(r)}")
        return r

    @staticmethod
    def _json(r: httpx.Response, kind: type, what: str) -> Any:
        try:
            data = r.json()
        except ValueError:
            raise BrainError(f"{what}: the answer is not JSON") from None
        if not isinstance(data, kind):
            raise BrainError(f"{what}: expected a JSON {'list' if kind is list else 'object'}")
        return data

    def health(self) -> dict:
        return self._json(self._send("GET", "/health"), dict, "GET /health")

    def stop(self) -> dict:
        return self._json(self._send("POST", "/stop", timeout=STOP_TIMEOUT_S, http=self.stop_http), dict, "POST /stop")

    def pending(self) -> tuple[list, float | None]:
        """The plans waiting for approval, and the brain's clock when it listed them."""
        r = self._send("GET", "/operations/pending")
        return self._json(r, list, "GET /operations/pending"), brain_clock(r)

    def approve(self, operation_id: str, digest: str) -> dict:
        path = f"/operations/{operation_id}/approve"
        r = self._send("POST", path, {"digest": digest, "approved_by": WHO}, timeout=APPROVE_TIMEOUT_S)
        return self._json(r, dict, f"POST {path}")

    def command(self, text: str) -> dict:
        r = self._send("POST", "/commands", {"text": text, "who": WHO}, timeout=COMMAND_TIMEOUT_S)
        return self._json(r, dict, "POST /commands")


# ---------------------------------------------------------------------------
# Carrying out decisions
# ---------------------------------------------------------------------------
class Assist:
    """STOP runs at once on the caller's thread. Every other action runs on one worker thread,
    so a slow brain (an approved plan runs inside its approve request) never delays a STOP.
    While the worker runs, busy is True and the Decider ignores other signals.
    brain=None is a dry run: actions are logged, the brain is never called."""

    def __init__(self, brain: Brain | None, log: Log, say: Callable[[str], None],
                 clock: Callable[[], float] = time.monotonic, background: bool = True):
        self.brain, self.log, self.say, self.clock, self.background = brain, log, say, clock, background
        self.guard = ApproveGuard()
        self.running = ""  # the action on the worker thread, if any
        self._worker: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def do(self, d: Decision) -> None:
        if d.action.kind == "stop" or not self.background:
            self._run(d)
            return
        self.running = d.action.kind
        self._worker = threading.Thread(target=self._run, args=(d,), name="assist-action", daemon=True)
        self._worker.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the worker thread. True when nothing is running any more."""
        if self._worker is not None:
            self._worker.join(timeout)
        return not self.busy

    def _run(self, d: Decision) -> None:
        handlers = {"stop": self._stop, "approve": self._approve, "cancel": self._cancel, "command": self._command}
        try:
            fields = handlers[d.action.kind](d)
        except Exception as e:  # noqa: BLE001 - one failed action never ends the loop
            fields = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        self.log("action", action=d.action.kind, signal=d.signal, **fields)

    def _stop(self, d: Decision) -> dict:
        if self.brain is None:
            result = {"status": "dry_run"}
        else:
            try:
                state = self.brain.stop()
            except BrainError as e:
                result = {"status": "error", "error": str(e)}
            else:
                result = {"status": "sent", "stopped": state.get("stopped"), "robot_connected": state.get("connected")}
                # An unreachable robot comes back as connected: false with stopped: true, so check both.
                if state.get("connected") is False or state.get("stopped") is not True:
                    result["warning"] = ("the brain did not confirm the robot stopped: use the iPad STOP or the"
                                         " motor power")
        if d.action.say and not d.repeat:
            self.say(d.action.say)
        return result

    def _approve(self, d: Decision) -> dict:
        if self.brain is None:
            if d.action.say:
                self.say(d.action.say)
            return {"status": "dry_run",
                    "note": f"would approve the one waiting plan if less than {APPROVE_MAX_AGE_S:.0f} s old"}
        try:
            pending, brain_now = self.brain.pending()
        except BrainError as e:
            return {"status": "error", "error": str(e)}
        choice, why = self.guard.check(pending, brain_now, self.clock())
        if choice is None:
            return {"status": "refused", "reason": why}
        if d.action.say:
            self.say(d.action.say)
        try:
            r = self.brain.approve(choice.operation_id, choice.digest)
        except BrainError as e:
            return {"status": "error", "operation_id": choice.operation_id, "error": str(e)}
        return {"status": "sent", "operation_id": choice.operation_id, "request": choice.request,
                "plan_age_s": choice.age_s, "outcome": r.get("outcome"), "failure": r.get("failure")}

    def _cancel(self, d: Decision) -> dict:
        if self.brain is None:
            result = {"status": "dry_run", "note": "would remember the waiting plans so no thumbs-up approves them"}
        else:
            error = None
            try:
                pending, _ = self.brain.pending()
            except BrainError as e:
                pending, error = None, str(e)
            ids = self.guard.cancel(pending, self.clock())
            if isinstance(pending, list):
                result = {"status": "done", "cancelled": ids, "note": NO_REJECT_NOTE}
            else:
                result = {"status": "done", "cancelled": [], "error": error,
                          "warning": "could not read the waiting plans, so thumbs-up approvals are off for"
                                     f" {self.guard.locked_until - self.clock():.0f} s"}
        if d.action.say:
            self.say(d.action.say)
        return result

    def _command(self, d: Decision) -> dict:
        if d.action.say:
            self.say(d.action.say)
        if self.brain is None:
            return {"status": "dry_run", "text": d.action.text}
        try:
            r = self.brain.command(d.action.text)
        except BrainError as e:
            return {"status": "error", "text": d.action.text, "error": str(e)}
        return {"status": "sent", "text": d.action.text, "kind": r.get("kind"), "result": r.get("status"),
                "operation_id": r.get("operation_id"), "reason": r.get("reason")}


# ---------------------------------------------------------------------------
# Frame sources: each read() returns JPEG bytes, raises FrameError for a bad frame
# ---------------------------------------------------------------------------
class RobotCamera:
    """The robot's own camera: GET <robot>/robot/frame?annotate=false (Brian's server, port 8200)."""

    def __init__(self, url: str, client: httpx.Client | None = None):
        self.http = client or httpx.Client(base_url=url.rstrip("/"), trust_env=False)
        self.current = None

    def read(self) -> bytes:
        try:
            r = self.http.get("/robot/frame", params={"annotate": "false"}, timeout=FRAME_TIMEOUT_S)
        except httpx.HTTPError as e:
            raise FrameError(f"robot camera: {type(e).__name__}: {e}") from None
        if r.status_code != 200:
            raise FrameError(f"robot camera: HTTP {r.status_code}: {r.text[:100]}")
        if not r.content.startswith(b"\xff\xd8"):
            raise FrameError("robot camera: the answer is not a JPEG")
        return r.content

    def close(self) -> None:
        self.http.close()


class Webcam:
    """A camera on this computer (OpenCV)."""

    def __init__(self, index: int):
        try:
            import cv2
        except ImportError:
            raise SetupError("--webcam needs OpenCV: pip install opencv-python") from None
        self.cv2, self.current = cv2, None
        self.cam = cv2.VideoCapture(index)
        if not self.cam.isOpened():
            raise SetupError(f"could not open webcam {index}; try --webcam 1")

    def read(self) -> bytes:
        ok, frame = self.cam.read()
        if not ok or frame is None:
            raise FrameError("the webcam gave no frame")
        ok, jpg = self.cv2.imencode(".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise FrameError("could not encode the webcam frame")
        return jpg.tobytes()

    def close(self) -> None:
        self.cam.release()


class ImageFolder:
    """Plays the .jpg files in a folder once, in name order (for rehearsal and tests)."""

    def __init__(self, folder: str):
        path = Path(folder)
        if not path.is_dir():
            raise SetupError(f"--images: {folder} is not a folder")
        self.files = sorted(f for f in path.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg"))
        if not self.files:
            raise SetupError(f"--images: no .jpg files in {folder}")
        self._next = 0
        self.current = None  # the file read last

    def read(self) -> bytes:
        if self._next >= len(self.files):
            raise EndOfFrames
        f = self.files[self._next]
        self._next += 1
        self.current = f.name
        try:
            return f.read_bytes()
        except OSError as e:
            raise FrameError(f"{f.name}: {e.strerror or e}") from None

    def close(self) -> None:
        pass


def open_source(args: argparse.Namespace):
    if args.robot:
        return RobotCamera(args.robot)
    if args.webcam is not None:
        return Webcam(args.webcam)
    return ImageFolder(args.images)


# ---------------------------------------------------------------------------
# Classifiers: jpeg bytes -> {"signal", "probability", ...}; any exception means no signal
# ---------------------------------------------------------------------------
def intent_classifier() -> Callable[[bytes], dict]:
    missing = [name for name in INTENT_ENV if not os.environ.get(name, "").strip()]
    if missing:
        raise SetupError(f"--classifier intent needs {' and '.join(missing)} set in the environment"
                         " (see docs/ASSIST.md)")
    try:
        from engram.vision.intent_vision import classify_hand_signal
    except ImportError as e:
        raise SetupError(f"--classifier intent needs engram/vision/intent_vision.py: {e}") from None
    return classify_hand_signal


def mediapipe_classifier() -> Callable[[bytes], dict]:
    import importlib.util

    if importlib.util.find_spec("mediapipe") is None:
        raise SetupError("--classifier mediapipe needs mediapipe: pip install mediapipe (Python 3.9 to 3.12)")
    try:
        import cv2
        import numpy as np

        from engram.vision.gestures import HandTracker

        with contextlib.redirect_stdout(sys.stderr):  # the one-time model download prints a line
            tracker = HandTracker()
    except (Exception, SystemExit) as e:  # noqa: BLE001
        raise SetupError(f"could not start MediaPipe: {e}") from None

    def classify(jpeg: bytes) -> dict:
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise FrameError("the frame is not an image")
        t0 = time.perf_counter()
        answer = signal_from_hands(tracker.hands(frame))
        answer["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        return answer

    return classify


def make_classifier(name: str) -> Callable[[bytes], dict]:
    return intent_classifier() if name == "intent" else mediapipe_classifier()


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def classify_frame(classify: Callable[[bytes], Any], jpeg: bytes, log: Log, verbose: bool, where: dict) -> Reading:
    try:
        result = classify(jpeg)
    except Exception as e:  # noqa: BLE001 - the classifier's VisionError or anything else: no signal
        log("error", where="classifier", error=f"{type(e).__name__}: {e}", **where)
        return no_signal("the classifier failed")
    reading = reading_from(result)
    if reading.problem:
        log("error", where="classifier", error=reading.problem, **where)
    elif verbose:
        log("seen", signal=reading.signal, probability=round(reading.probability, 3),
            latency_ms=result.get("latency_ms"), **where)
    return reading


def step(decider: Decider, assist: Assist, reading: Reading, now: float, log: Log, where: dict) -> None:
    d = decider.decide(reading, now, busy=assist.busy)
    if d is None:
        return
    log("signal", signal=d.signal, probability=round(d.probability, 3), action=d.action.kind, **where)
    if d.action.kind == "none":
        return
    if not d.go:
        log("action", action=d.action.kind, signal=d.signal, status="skipped", reason=d.why)
        return
    assist.do(d)


def run(args: argparse.Namespace, signal_map: SignalMap, source, classify: Callable[[bytes], Any],
        brain: Brain | None, log: Log, speak: Callable[[str], bool] | None,
        clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
        background: bool = True) -> int:
    decider = Decider(signal_map)
    assist = Assist(brain, log, Voice(log, speak), clock, background)
    period = 1.0 / args.rate
    frames, reason = 0, "ctrl+c"
    try:
        while True:
            started = clock()
            try:
                try:
                    jpeg = source.read()
                except EndOfFrames:
                    reason = "no more images"
                    break
                except Exception as e:  # noqa: BLE001 - FrameError or anything else: no signal
                    log("error", where="frame", error=str(e) or type(e).__name__)
                    reading, where = no_signal("no frame"), {}
                else:
                    frames += 1
                    where = {"frame": frames, "file": source.current}
                    reading = classify_frame(classify, jpeg, log, args.verbose, where)
                step(decider, assist, reading, started, log, where)
            except Exception as e:  # noqa: BLE001 - last resort: one bad frame never ends the loop
                log("error", where="loop", error=f"{type(e).__name__}: {e}")
            sleep(max(0.0, started + period - clock()))
        if assist.busy:
            log("info", message="no more images; waiting for the last request to finish (Ctrl+C quits now)")
            while not assist.wait(0.2):
                pass
    except KeyboardInterrupt:
        reason = "ctrl+c"
    log("exit", reason=reason, frames=frames, still_running=assist.running if assist.busy else None)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ENGRAM Assist: hand signals to the robot (docs/ASSIST.md).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--robot", metavar="URL", help="the robot's camera (Brian's server), e.g. http://star-pi.local:8200")
    src.add_argument("--webcam", type=int, metavar="N", help="a camera on this computer (OpenCV index, usually 0)")
    src.add_argument("--images", metavar="DIR", help="play the .jpg files in DIR once, in name order")
    ap.add_argument("--classifier", choices=("intent", "mediapipe"), default="intent",
                    help="intent: the intent model (needs TYPESAFE_API_KEY and TYPESAFE_INTENT_MODEL);"
                         " mediapipe: hand landmarks on this computer")
    ap.add_argument("--brain", default="http://127.0.0.1:8001", metavar="URL", help="the ENGRAM brain")
    ap.add_argument("--rate", type=float, default=3.0, help="frames per second (default 3)")
    ap.add_argument("--map", default=str(DEFAULT_MAP), help="the signal map (default: assist_signals.default.json)")
    ap.add_argument("--dry-run", action="store_true", help="log the actions without calling the brain")
    ap.add_argument("--no-speech", action="store_true", help="do not say anything out loud")
    ap.add_argument("--verbose", action="store_true", help="also print every classification")
    args = ap.parse_args(argv)
    if not (math.isfinite(args.rate) and 0 < args.rate <= 30):
        ap.error("--rate must be more than 0 and at most 30")
    for flag in ("robot", "brain"):
        url = getattr(args, flag)
        if url and not url.startswith(("http://", "https://")):
            ap.error(f"--{flag} must start with http:// or https://")
    return args


def main(argv: list[str] | None = None, *, classifier: Callable[[bytes], Any] | None = None,
         brain: Brain | None = None, speaker: Callable[[str], bool] | None = None, out=None,
         clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
         background: bool = True) -> int:
    """The keyword arguments let tests swap in a fake classifier, brain, voice and clock."""
    args = parse_args(argv)
    log = Log(out)
    try:
        signal_map = load_map(args.map)
        classify = classifier or make_classifier(args.classifier)
        source = open_source(args)
    except (MapError, SetupError) as e:
        log("error", where="setup", error=str(e))
        return 2
    brain = None if args.dry_run else (brain or Brain(args.brain))
    speak = None if args.no_speech else (speaker or system_voice())
    log("start", source=args.robot or (f"webcam {args.webcam}" if args.webcam is not None else args.images),
        classifier=args.classifier, brain=None if brain is None else args.brain, dry_run=args.dry_run,
        rate=args.rate, map=args.map, speech=speak is not None,
        signals={name: a.kind for name, a in signal_map.actions.items()})
    try:
        if brain is not None:
            try:
                health = brain.health()
                log("brain", url=args.brain, ok=health.get("ok"), robot=health.get("robot"))
            except BrainError as e:
                log("error", where="brain", error=f"{e} (going on; each action asks again)")
        return run(args, signal_map, source, classify, brain, log, speak, clock, sleep, background)
    finally:
        # The camera is released. The brain's connections are left to the process exit: after a
        # Ctrl+C, an approve may still be waiting on one in the worker thread.
        with contextlib.suppress(Exception):
            source.close()


if __name__ == "__main__":
    raise SystemExit(main())
