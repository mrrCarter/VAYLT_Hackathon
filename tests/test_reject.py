"""POST /operations/{id}/reject: a person declines a plan waiting for approval, and it can never run."""

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import engram.engine as engine_module
from engram.api import create_app
from engram.engine import EngineError

OBSERVED_ONLY = [("observe", {})]   # the fixture's overhead scan: nothing else reached the robot


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine))


def propose(client):
    """A plan the iPad did not ask for, like a gesture's "return to rest"."""
    p = client.post("/plan/rest", json={}).json()
    assert p["status"] == "proposed"
    return p


def reject(client, operation_id, by="patient", reason="thumbs down"):
    return client.post(f"/operations/{operation_id}/reject", json={"rejected_by": by, "reason": reason})


def approve(client, p):
    return client.post(f"/operations/{p['operation_id']}/approve", json={"digest": p["digest"], "approved_by": "nurse"})


def test_a_rejected_plan_can_never_be_approved(client, engine):
    p = propose(client)
    r = reject(client, p["operation_id"])
    assert r.status_code == 200
    assert r.json() == {"operation_id": p["operation_id"], "outcome": "cancelled",
                        "failure": "rejected by patient: thumbs down", "already_rejected": False}
    late = approve(client, p)
    assert late.status_code == 409 and "cancelled" in late.json()["detail"]
    assert engine.robot.calls == OBSERVED_ONLY
    assert engine.conn.execute("SELECT COUNT(*) FROM approval").fetchone()[0] == 0


def test_rejecting_twice_answers_the_same_and_records_once(client):
    p = propose(client)
    first = reject(client, p["operation_id"]).json()
    again = reject(client, p["operation_id"], by="nurse", reason="")
    assert again.status_code == 200
    assert again.json() == {**first, "already_rejected": True}          # still the first person's rejection
    rows = [e for e in client.get("/events").json() if e["op"] == "reject"]
    assert len(rows) == 1


def test_a_rejected_plan_leaves_the_pending_list(client):
    p = propose(client)
    assert [x["operation_id"] for x in client.get("/operations/pending").json()] == [p["operation_id"]]
    reject(client, p["operation_id"])
    assert client.get("/operations/pending").json() == []


def test_rejecting_an_unknown_operation_is_404(client):
    r = reject(client, "ep_nothere")
    assert r.status_code == 404 and r.json()["detail"] == "no operation ep_nothere"


def test_rejecting_a_plan_that_is_not_waiting_is_409(client, engine):
    done = propose(client)
    assert approve(client, done).json()["outcome"] == "success"
    refused = client.post("/commands", json={"text": "Don't open the gripper", "who": "patient"}).json()
    unchecked = client.post("/plan/rest", json={"zone": "moon"}).json()          # the checker said no
    assert (refused["status"], unchecked["status"]) == ("refused", "rejected")
    for op_id, outcome in ((done["operation_id"], "success"), (refused["operation_id"], "refused"),
                           (unchecked["operation_id"], "rejected")):
        r = reject(client, op_id)
        assert r.status_code == 409 and r.json()["detail"] == f"operation is {outcome}, not waiting for approval"
        row = engine.conn.execute("SELECT outcome FROM episode WHERE id = ?", (op_id,)).fetchone()
        assert row["outcome"] == outcome                                           # left as it was


def test_a_rejection_is_recorded_in_the_operation_journal(client):
    p = propose(client)
    reject(client, p["operation_id"])
    (row,) = [e for e in client.get("/events").json() if e["op"] == "reject"]
    assert (row["episode_id"], row["status"], row["detail"]) == (p["operation_id"], "rejected",
                                                                 "rejected by patient: thumbs down")
    assert json.loads(row["args_json"]) == {"rejected_by": "patient", "digest": p["digest"]}


@pytest.mark.parametrize("body", [{}, {"rejected_by": ""}, {"reason": "no name"}])
def test_a_rejection_needs_a_name(client, body):
    p = propose(client)
    assert client.post(f"/operations/{p['operation_id']}/reject", json=body).status_code == 422
    assert [x["operation_id"] for x in client.get("/operations/pending").json()] == [p["operation_id"]]


def _between_read_and_take(monkeypatch, act):
    """Run act() once, after approve_and_run has read the plan as waiting but before it takes it."""
    real, fired = engine_module.digest, []

    def digest(value):
        if not fired:
            fired.append(True)
            act()
        return real(value)

    monkeypatch.setattr(engine_module, "digest", digest)


def test_an_approval_and_a_rejection_cannot_both_win(engine, monkeypatch):
    p = engine.propose_rest()
    _between_read_and_take(monkeypatch, lambda: engine.reject(p["operation_id"], "patient"))
    with pytest.raises(EngineError, match="no longer waiting"):
        engine.approve_and_run(p["operation_id"], p["digest"], "nurse")
    assert engine.robot.calls == OBSERVED_ONLY
    assert engine.conn.execute("SELECT outcome FROM episode WHERE id = ?", (p["operation_id"],)).fetchone()[0] == "cancelled"
    assert engine.conn.execute("SELECT COUNT(*) FROM approval").fetchone()[0] == 0


def test_two_approvals_run_a_plan_once(engine, monkeypatch):
    p = engine.propose_rest()
    _between_read_and_take(monkeypatch, lambda: engine.approve_and_run(p["operation_id"], p["digest"], "nurse"))
    with pytest.raises(EngineError, match="no longer waiting"):
        engine.approve_and_run(p["operation_id"], p["digest"], "carer")
    assert engine.robot.calls == OBSERVED_ONLY + [("go_to_zone", {"zone": "home"})]
    assert engine.conn.execute("SELECT COUNT(*) FROM approval").fetchone()[0] == 1


def test_pending_plans_carry_the_servers_time(client):
    p = propose(client)
    (item,) = client.get("/operations/pending").json()
    assert item["operation_id"] == p["operation_id"] and item["created_at"].endswith("Z")
    created = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
    assert created.utcoffset().total_seconds() == 0
    assert abs(created.timestamp() - item["proposed_at"]) < 0.001
    assert 0 <= item["age_s"] < 60
