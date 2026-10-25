# ENGRAM Assist

A helper robot for a patient who cannot get up. The patient says a command (or it is typed). The brain
turns it into a short plan, a person approves that exact plan on the iPad (hold Approve), and only then
does the robot move. "Stop" needs no approval: it stops the robot at once.

Built on ENGRAM: the console's speech in and out (whisper and piper on the Pi), the brain's memory and
safety rules ("Stop" always stops; "don't..." and "later..." never move anything; every move waits for a
person), voice presets the intent model gets ready while the patient is still talking, hand signals read
by the intent model, and the Cortex memory inspector.

## Start here (step by step)

**You need:** the robot with Brian's robot server running on the Pi (port 8200, with a non-default
`ENGRAM_ROBOT_SECRET` in `~/engram_robot/robot.env`, see `docs/CDS_RUNBOOK.md` 9b), the iPad, a laptop on the
same private hotspot, and the tags printed from `docs/tags-40mm.pdf` at **100% / actual size** (the black
square must measure 40 mm). Tape **tag 5 on the pill bottle** and put **tag 15 at the patient's place** (on
a box or post by the sofa, facing the room).

1. **Put ENGRAM Assist on the Pi (once).** In an SSH window on the Pi, paste this as one line:

   ```bash
   cd ~/engram_hub && git fetch https://github.com/mrrCarter/VAYLT_Hackathon.git engram-assist && git show FETCH_HEAD:scripts/cds_update.sh > /tmp/cds_update.sh && bash /tmp/cds_update.sh engram-assist
   ```

   It saves anything changed on the Pi first (a backup branch, a patch, a copy of new files and the Python
   packages) and its last line is the exact undo command: keep it. Brian's `~/engram_robot` is never touched.

2. **Start on the pretend robot (nothing can move):** `bash ~/engram_hub/scripts/cds_up.sh --restart`.
   On the iPad open `https://star-pi.local:8443` (first time: install the certificate, `docs/CDS_RUNBOOK.md`
   section 7). Try the commands in the table below; approving is **press and hold**. Check:
   "Where are my pills?" answers without a plan, and "Stop." stops.

3. **Switch to the real robot:** wheels lifted for the first tries, one person's hand on the motor power.
   `bash ~/engram_hub/scripts/cds_up.sh --real`. The status must say `plans + memory (real robot)`.
   Say "Stop." first and check the robot reports stopped; clear it on the iPad. Then "Move forward",
   "Turn left", "Pick it up", "Put it down", each approved by holding Approve.

4. **Hand signals (laptop next to the robot):** see "Hand signals" below. It reads the robot's camera at
   `http://star-pi.local:8200` and sends to the brain at `http://star-pi.local:8001`. The intent model needs
   `TYPESAFE_API_KEY` (Carter gives it privately; never post it) and `TYPESAFE_INTENT_MODEL`.

5. **Status any time:** `bash ~/engram_hub/scripts/cds_status.sh`. **Stop everything:**
   `bash ~/engram_hub/scripts/cds_down.sh` (Brian's server keeps running).

**Safety, always:** "Stop", the iPad's STOP button, an open palm to the camera, or the motor power switch.
Nothing moves without an approval, but the robot cannot see behind or beside itself: keep the floor clear.

## Commands

| Say | Plan the brain sends to Brian's robot server | What the robot does |
|---|---|---|
| "Move forward" | `drive_bounded` 0.20 m | drives straight ahead 20 cm |
| "Move back", "Back up" | `drive_bounded` -0.20 m | backs up 20 cm, slowly (nothing on the robot can see behind it) |
| "Turn left" | `turn_bounded` +45 degrees | turns in place 45 degrees to the left (counter-clockwise) |
| "Turn right" | `turn_bounded` -45 degrees | turns in place 45 degrees to the right |
| "Pick it up" | `arm_pose` reach, `gripper_close`, `arm_pose` carry | lowers the arm to what is in front of the claw, closes the claw, lifts it |
| "Put it down" | `arm_pose` reach, `gripper_open`, `arm_pose` carry | lowers the arm, opens the claw, lifts the arm clear |
| "Bring me my pills", "Get my pills", "Bring me the pill bottle" | `locate_tag`, `pick_tagged_object` pill_bottle, `place_at_zone` patient, `speak` | finds the bottle (tag 5), picks it up, carries it to the patient's marker (tag 15), lets go |
| "Where are my pills?" | none: answered from memory | nothing moves |
| "Stop" | none: stops now | stays stopped until a person clears it on the iPad |

Other wordings work too: "Go forward", "Move backward", "Turn to the left", "Pick that up", "Set it down",
with or without "please" or "Robot,". The arm and the wheels have no sensors on this robot, so their
results say "not sensor checked": watch the robot.

**Distances and angles.** "Move forward 30 centimeters", "Back up 10 cm", "Turn left 90 degrees",
"Turn 30 degrees to the right", "Move forward one foot" (digits or words). The limits are 50 cm forward,
20 cm back (the robot server allows no more blind reversing) and 180 degrees. Past a limit, or with no
unit for a distance, nothing is planned and the brain says why, for example "I can move forward at most
50 cm at a time, and you asked for 80 cm."

**Never a move:** "Don't move forward", "Turn left later" and questions ("Should I move forward?") are
refused or answered. "When the nurse leaves, move forward" is saved and proposed when it happens; it
still needs approval.

**Saying no to a plan** (a thumbs-down): `POST /operations/{operation_id}/reject` with
`{"rejected_by": "patient", "reason": "thumbs down"}`. The plan ends `cancelled`, leaves
`GET /operations/pending` and can never be approved; rejecting it again gives the same answer. An
unknown id is 404, a plan that is no longer waiting is 409. Each pending plan has `created_at` (the
brain's clock, ISO 8601 UTC) and `age_s`, so another machine can tell how old it is without its own clock.

## The pill bottle and the patient's place

| | Tag (ArUco DICT_4X4_50) | Name in the brain | Other names it understands |
|---|---|---|---|
| Pill bottle | **5** | object `pill_bottle` | pills, pill, medicine, medication, meds |
| Patient's place | **15** | zone `patient` | me, my, sofa, couch |

"Me" and "my" mean the patient's place unless the request names a zone: "Bring my pills to the bench"
goes to the bench. Print tags 5 and 15 with the others, at 40 mm like the rest: `python scripts/make_tags.py
--mm 40`. Put tag 15 on a box or post next to the patient, facing where the robot will be after the pick.

**How the robot finds them** (no overhead camera needed): the first step only looks, with the robot's
own camera, and never moves. If tag 5 is not in view, the pick turns the robot in place, 30 degrees at a
time and at most one full turn, until it sees the tag, then picks; the place does the same for tag 15.
If it cannot find a tag after a full turn, that step fails ("not seen in 360 deg sweep") and nothing more
happens. "Stop" while it turns stops it at once, and nothing is picked.

## Changing them

- **Tags and names:** `fixtures/lab/arena.json` (or your own copy, named by `ENGRAM_ARENA`). The pill
  bottle's `tag` and `aliases` are in `objects`, the patient's marker is `"15": "patient"` in `zone_tags`,
  and the patient's other names are in `zone_aliases`. Restart the brain afterwards
  (`bash scripts/cds_up.sh --restart`, plus `--real` on the real robot). If you change the file in the
  repo, update its hash in `engram/package/engram_seed.json` too (`tests/test_engram_package.py` checks it).
- **Arm poses:** "Pick it up" and "Put it down" use the robot's saved poses `reach` and `carry` (Brian's
  names, saved with `robot/engram_robot/tools/arm_pose.py`). If the robot has them under other names, set
  `ENGRAM_REACH_POSE` and `ENGRAM_CARRY_POSE` for the brain.
- **Step sizes and limits:** `STEP_M`, `TURN_DEG` and `MAX_REVERSE_M` in `engram/assist.py`;
  `MAX_DRIVE_M` and `MAX_TURN_DEG` in `engram/capabilities.py`.

The six fixed commands are also voice presets (`engram/spark/presets.py`), so the intent model can get
them ready while the patient is still speaking. A preset always plans its default step, so check the
plan card before approving. Code: `engram/assist.py`. Tests: `tests/test_assist.py`, and
`tests/test_assist_e2e.py` (the brain driving Brian's robot server on its simulator).

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
| Thumbs down | Rejects the plan that is waiting, under the same rule as a thumbs-up (exactly one waiting, proposed less than 60 seconds ago), and then says "Okay, cancelled" out loud. The plan ends cancelled in the brain and can never be approved, on the iPad either. If the reject fails, nothing is said and the line says why; the loop still never approves that plan by hand signal, and another thumbs-down tries again. |
| Peace (V sign) | Sends "Bring me my pills" to the brain, as if the patient had said it. The plan shows up on the iPad and waits for approval (a thumbs-up, or Approve on the iPad). |
| Fist | Sends "Pick it up" to the brain, the same way (lower the arm, close the claw, lift). |
| Pointing, waving | Nothing (for now). |

What the brain does with these two commands is in Commands and The pill bottle above. In
particular, "Bring me my pills" needs a recent sighting of the pill bottle (tag 5) before the
robot can move.

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
   brain's own clock (the brain says how old each waiting plan is; if it ever does not, its HTTP
   Date header stands in; with neither, nothing is approved); no thumbs-down turned it down. It
   approves that plan's exact digest, so nothing else can be run by mistake.
5. **A thumbs-down rejects the plan in the brain** under the same rule (exactly one waiting, less
   than 60 seconds old), and says "Okay, cancelled" only once the brain confirms it. It also
   remembers every plan waiting at that moment, so no later thumbs-up approves them even if the
   reject failed. If the loop cannot read the waiting plans just then, thumbs-up approval is
   switched off for about a minute, until anything that was waiting is too old to approve anyway.
6. **When in doubt, nothing happens.** A failed check, a broken camera frame or a brain that does
   not answer counts as "no signal". The loop prints the problem and carries on; it never stops
   because of a bad frame or the network.
7. **The open palm is an extra stop, never the only one.** Keep the iPad STOP and a hand on the motor
   power, as in `docs/CDS_RUNBOOK.md` section 1. The open palm stops the robot only as fast as the
   classifier answers: run with `--verbose` and look at `latency_ms`.

The brain records every approval the loop makes as approved by "patient (hand signal)", and every
reject as rejected by "hand-signal" with the reason "patient signalled no".

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
  `outcome`, `reason`), `dry_run`, `skipped` (pause or busy; `reason` says which), `refused` (a
  rule said no, for example two plans waiting; `reason` says which), or `error` (the brain did not
  answer, or answered with an error). A thumbs-down's line also lists, as `remembered`, the plans
  no thumbs-up will approve.
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
  happens. On a stop it is said once per palm, not on every frame; on a cancel, only after the
  brain confirms the reject.
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

- Closed: the brain has `POST /operations/{id}/reject` now, and a thumbs-down uses it. After a
  reject, the plan's card can stay on the iPad screen until the page next refreshes; approving it
  there is refused by the brain (the plan is cancelled), so nothing moves.
- Closed: each waiting plan carries `age_s` (and `created_at`) from the brain's own clock, and the
  loop uses it. Only for a plan without `age_s` does it read the brain's clock from the HTTP `Date`
  header (accurate to a second or two, so plans count as up to 2 seconds older than they look).
- `POST /operations/{id}/approve` answers only when the plan has finished moving. The loop sends it
  from a separate thread so the open palm keeps working meanwhile.
- `POST /stop` answers "stopped" even when the brain could not reach the robot (it then also says
  `connected: false`). Whenever the brain does not confirm a connected, stopped robot, the loop's
  stop line carries a warning: use the iPad STOP or the motor power.
