# ENGRAM Assist

ENGRAM Assist is the healthcare demo: a patient who cannot speak asks the robot for help.

## Hand signals

The patient makes a hand signal at the robot's camera. A loop on the laptop
(`scripts/assist_gestures.py`) looks at the camera a few times a second, asks a classifier which
signal it sees, and tells the brain what to do. The rules live in
`engram/vision/assist_signals.py`; which signal does what lives in a file the team edits,
`engram/vision/assist_signals.default.json`.

### What each signal does

| Signal | What happens |
|---|---|
| Open palm (flat hand, fingers up, palm to the camera) | **STOP, at once.** One frame is enough, and STOP is sent again on every frame while the palm stays up. Afterwards a person re-arms the robot: on the iPad, or `curl -s -X POST http://star-pi.local:8001/robot/clear_stop`. Re-arming is never a hand signal. |
| Thumbs up | Approves the plan that is waiting, but only under the approval rules below. |
| Thumbs down | Says "Okay, cancelled" out loud and approves nothing. The brain has no way to reject a plan yet, so the plan is still waiting in the brain (and on the iPad, where a person can still approve it); but no thumbs-up will ever approve it. |
| Peace (V sign) | Sends "Bring me my pills" to the brain, as if the patient had said it. When the brain can plan it, the plan shows up on the iPad and waits for approval. |
| Fist | Sends "Pick it up" to the brain, the same way. |
| Pointing, waving | Nothing (for now). |

The brain only plans for objects it knows (`fixtures/lab/arena.json`: Sample A, Sample B, Safety
goggles, Beaker), and "Pick it up" names no object at all. Until the brain handles them, it answers
both default commands with "I don't know which object you mean", the loop prints that answer, and
nothing moves. To rehearse before then, change the command text in the map, for example to
"Bring me the beaker".

### The safety rules, in plain words

1. **Stop always wins.** An open palm stops the robot straight away. It does not wait for a hold,
   for the pause after an action, or for a request that is still running.
2. **Every other signal has to be held**: seen in at least 2 checks in a row, spanning at least
   0.6 seconds.
   If one check sees no signal, fails, or is unsure (probability under 0.6), the hold starts over.
   One hold does one thing. Holding a fist for ten seconds sends "Pick it up" once; to send it
   again, lower the hand and show it again.
3. **After any action there is a 3 second pause** (a stop starts it too) in which other signals are
   ignored, and they are also ignored while the last request to the brain is still running (for
   example while an approved plan is moving the robot). An ignored signal is dropped, not saved:
   show it again afterwards. The open palm works during the pause.
4. **A thumbs-up approves only when all of these are true**: exactly one plan is waiting (zero or
   two or more: nothing happens; use the iPad); it was proposed less than 60 seconds ago, by the
   brain's own clock; nobody cancelled it with a thumbs-down. It approves that plan's exact
   digest, so nothing else can be run by mistake.
5. **A thumbs-down remembers** the plans waiting at that moment, and no later thumbs-up approves
   them. If the loop cannot read the waiting plans just then, thumbs-up approval is switched off
   for about a minute, until anything that was waiting is too old to approve anyway.
6. **When in doubt, nothing happens.** A failed check, a broken camera frame or a brain that does
   not answer counts as "no signal". The loop prints the problem and carries on; it never stops
   because of a bad frame or the network.
7. **The open palm is an extra stop, never the only one.** Keep the iPad STOP and a hand on the motor
   power, as in `docs/CDS_RUNBOOK.md` section 1. The open palm stops the robot only as fast as the
   classifier answers: run with `--verbose` and look at `latency_ms`.

Every approval the loop makes is recorded in the brain as approved by "patient (hand signal)".

### Run it on the laptop, against the robot's camera

The robot server (port 8200) and the brain (port 8001) run on star-pi as in `docs/CDS_RUNBOOK.md`.
The loop runs on the laptop, from the repo folder, with the repo's Python packages installed
(`pip install -r requirements.txt`). If `star-pi.local` is not found, use the robot's address.

With the intent model (needs its key and model id; get them in person or by direct message, never
through the room, and never put them in a file in the repo):

```bash
# macOS, Linux or Git Bash
read -rsp 'intent model key: ' TYPESAFE_API_KEY; echo; export TYPESAFE_API_KEY
export TYPESAFE_INTENT_MODEL=<model id>
python scripts/assist_gestures.py --robot http://star-pi.local:8200 --brain http://star-pi.local:8001 --dry-run
```

```powershell
# Windows PowerShell
$env:TYPESAFE_API_KEY = [System.Net.NetworkCredential]::new("", (Read-Host "intent model key" -AsSecureString)).Password
$env:TYPESAFE_INTENT_MODEL = "<model id>"
python scripts/assist_gestures.py --robot http://star-pi.local:8200 --brain http://star-pi.local:8001 --dry-run
```

`--dry-run` prints what it would do and never calls the brain. When that looks right, run the same
command without `--dry-run`. Press **Ctrl+C** to quit.

Without the intent model, use MediaPipe hand landmarks on the laptop instead (no key; needs
`pip install mediapipe opencv-python`, Python 3.9 to 3.12; it downloads an 8 MB model once):

```bash
python scripts/assist_gestures.py --robot http://star-pi.local:8200 --brain http://star-pi.local:8001 --classifier mediapipe
```

Other options:

- `--webcam 0` uses a camera on the laptop instead of the robot's (`--webcam 1` for a second one;
  needs OpenCV: `pip install -r requirements-vision.txt`).
- `--rate 3` is how many frames per second it looks at (the default).
- `--map my_map.json` uses another signal map.
- `--no-speech` keeps it quiet ("Okay, cancelled" is said on the laptop's speaker).
- `--verbose` also prints every single check, with the probability, so you can see what it sees.

**Rehearse without the robot moving.** Save a few frames while someone shows each signal, in a
folder outside the repo (camera pictures do not belong in git), then play them back:

```bash
mkdir ../rehearsal
curl -s "http://star-pi.local:8200/robot/frame?annotate=false" -o ../rehearsal/001.jpg
# ... 002.jpg, 003.jpg, one per frame; they play in name order, --rate per second
python scripts/assist_gestures.py --images ../rehearsal --dry-run --verbose
```

In Windows PowerShell, type `curl.exe` instead of `curl`.

### Reading what it prints

One line per event, each a JSON object with an `event` field:

- `start`: the settings and the map in use; `brain`: whether the brain answered at start-up.
- `signal`: a signal was recognised (held long enough, or an open palm), and which action it maps to.
- `action`: what was done about it. `status` is `sent` (the brain answered; see `result`,
  `outcome`, `reason`), `done`, `dry_run`, `skipped` (pause or busy; `reason` says which),
  `refused` (an approval rule said no; `reason` says which), or `error` (the brain did not answer,
  or answered with an error).
- `say`: a sentence said out loud (`spoken: false` when the laptop has no voice).
- `error`: a bad frame or a failed check (counted as no signal), or the brain unreachable.
- `exit`: the loop ended (Ctrl+C, or the last image played).

### Changing the map

Edit `engram/vision/assist_signals.default.json`, or copy it and pass the copy with `--map`. Each
signal gets an action:

```json
"peace": {"action": "command", "text": "Bring me my pills"},
"thumbs_down": {"action": "cancel", "say": "Okay, cancelled"},
"waving": {"action": "none"}
```

- Signals: `open_palm`, `thumbs_up`, `thumbs_down`, `pointing`, `fist`, `peace`, `waving`. A signal
  left out does nothing.
- Actions: `stop`, `approve`, `cancel` (the thumbs-down behaviour), `command` (needs `text`: the words
  sent to the brain, at most 500 characters), `none`.
- `say` (optional, on any action but `none`): a short sentence said on the laptop when the action
  happens. On a stop it is said once per palm, not on every frame.
- At least one signal must stop the robot. Any signal mapped to `stop` stops on a single frame.
- Tuning: `hold_frames` (2 to 30), `hold_seconds` (0 to 10), `min_probability` (0.5 to 1),
  `cooldown_s` (0 to 60).

The loop checks the map when it starts. If anything is wrong (a misspelt signal or action, a
command with no text, a signal written twice, nothing that stops), it prints what and does not
start. Restart the loop after each change. `python -m pytest tests/test_assist_signals.py -q`
checks the default map: when you change the mapping on purpose, update
`test_default_map_is_the_agreed_mapping` to match.

The map chooses which signal does what. It cannot change the safety rules above.

### Known gaps in the brain (for the brain team)

- There is no endpoint to reject or cancel a waiting plan. A thumbs-down can only skip it here.
  The plan stays waiting (a person can still approve it on the iPad) and stays on the brain's
  waiting list for 5 minutes; meanwhile a thumbs-up for a newer plan sees two plans and does
  nothing. A `POST /operations/{id}/reject` would let a thumbs-down remove it.
- The waiting list does not say what time the brain thinks it is, so the loop reads the brain's
  clock from the HTTP `Date` header, which is only accurate to a second or two. It allows for that
  by counting plans as up to 2 seconds older than they look.
- `POST /operations/{id}/approve` answers only when the plan has finished moving. The loop sends it
  from a separate thread so the open palm keeps working meanwhile.
- `POST /stop` answers "stopped" even when the brain could not reach the robot (it then also says
  `connected: false`). Whenever the brain does not confirm a connected, stopped robot, the loop's
  stop line carries a warning: use the iPad STOP or the motor power.
