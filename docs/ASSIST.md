# ENGRAM Assist

A helper robot for a patient who cannot get up. The patient says a command (or it is typed). The brain
turns it into a short plan, a person approves that exact plan on the iPad (hold Approve), and only then
does the robot move. "Stop" needs no approval: it stops the robot at once.

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

## The pill bottle and the patient's place

| | Tag (ArUco DICT_4X4_50) | Name in the brain | Other names it understands |
|---|---|---|---|
| Pill bottle | **5** | object `pill_bottle` | pills, pill, medicine, medication, meds |
| Patient's place | **15** | zone `patient` | me, my, sofa, couch |

"Me" and "my" mean the patient's place unless the request names a zone: "Bring my pills to the bench"
goes to the bench. Print tags 5 and 15 with the others, at 40 mm like the rest: `python scripts/make_tags.py
--mm 40`. Put tag 15 on a box or post next to the patient, facing where the robot will be after the pick.

**Before "Bring me my pills" on the real robot:** its first step looks up a sighting of tag 5 from the
last 10 minutes in the brain's memory (the overhead camera's job). With none, it stops at that step with
"overhead camera has not seen pill_bottle recently" and the robot does not move. Without an overhead
camera, post the sighting yourself just before:

```bash
curl -s -X POST http://star-pi.local:8001/observations/tag -H 'Content-Type: application/json' \
  -d '{"tag_id": 5, "zone": "shelf_B"}'
```

The pick itself then needs the robot's camera to see tag 5.

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
