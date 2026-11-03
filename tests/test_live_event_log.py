"""Background research runs and the live event log (SSE) endpoint."""
import json
import time

from fastapi.testclient import TestClient

from app.main import app
from app.core.session_events import SessionEventBus


def _parse_sse(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        name, data = "message", None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
        if data is not None:
            events.append((name, data))
    return events


def test_background_run_streams_events(monkeypatch, offline_llm_router):
    monkeypatch.setattr("agents.nodes.default_model_router", offline_llm_router)
    with TestClient(app) as client:
        res = client.post("/api/v1/research/sessions", json={"query": "analysis", "run_in_background": True})
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "IN_PROGRESS"
        sid = body["session_id"]

        status = "IN_PROGRESS"
        for _ in range(200):
            status = client.get(f"/api/v1/research/sessions/{sid}").json()["status"]
            if status != "IN_PROGRESS":
                break
            time.sleep(0.05)
        assert status == "USER_CLARIFICATION_REQUIRED"

        stream = client.get(f"/api/v1/research/sessions/{sid}/events")
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(stream.text)
        assert events[-1][0] == "end"
        logs = [data for name, data in events if name == "log"]
        assert any(e["kind"] == "node" and e["source"] == "receive_request" for e in logs)
        assert any(e["kind"] == "status" for e in logs)
        seqs = [e["seq"] for e in logs]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

        # Resuming from a cursor only returns newer events.
        tail = _parse_sse(client.get(f"/api/v1/research/sessions/{sid}/events?after={seqs[-2]}").text)
        assert [d["seq"] for n, d in tail if n == "log"] == [seqs[-1]]


def test_unknown_session_stream_ends_immediately():
    with TestClient(app) as client:
        events = _parse_sse(client.get("/api/v1/research/sessions/session_missing/events").text)
        assert events == [("end", {})]


def test_event_bus_is_bounded():
    bus = SessionEventBus()
    bus.start("s")
    for i in range(2100):
        bus.publish("s", f"m{i}")
    events = bus.events_since("s")
    assert len(events) == 2000
    assert events[-1]["seq"] == 2100
    bus.finish("s")
    assert not bus.is_running("s")
