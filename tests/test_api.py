from fastapi.testclient import TestClient

from api.main import create_app


def _client():
    return TestClient(create_app("sqlite:///:memory:"))


def _event(**kw):
    base = {"camera": "cam1", "defect_class": "scratches", "track_id": 7, "frame": 123.5, "confidence": 0.8}
    base.update(kw)
    return base


def test_health():
    assert _client().get("/health").json() == {"status": "ok"}


def test_post_then_list_roundtrip_and_server_timestamp():
    c = _client()
    r = c.post("/events", json=_event())
    assert r.status_code == 201
    body = r.json()
    assert body["id"] == 1 and body["camera"] == "cam1" and body["frame"] == 123.5
    assert body["ts"].endswith("Z") or "+00:00" in body["ts"]       # timezone-aware UTC
    assert c.get("/events").json()[0]["id"] == 1


def test_list_is_newest_first_and_filterable():
    c = _client()
    c.post("/events", json=_event(camera="cam1", defect_class="crazing"))
    c.post("/events", json=_event(camera="cam2", defect_class="scratches"))
    c.post("/events", json=_event(camera="cam2", defect_class="crazing"))
    ids = [e["id"] for e in c.get("/events").json()]
    assert ids == [3, 2, 1]
    assert [e["id"] for e in c.get("/events", params={"camera": "cam2"}).json()] == [3, 2]
    assert [e["id"] for e in c.get("/events", params={"defect_class": "crazing"}).json()] == [3, 1]
    assert len(c.get("/events", params={"limit": 2}).json()) == 2


def test_stats_counts_by_class_and_camera():
    c = _client()
    for cam, cls in [("cam1", "crazing"), ("cam1", "patches"), ("cam2", "crazing")]:
        c.post("/events", json=_event(camera=cam, defect_class=cls))
    s = c.get("/stats").json()
    assert s == {"total": 3, "by_class": {"crazing": 2, "patches": 1}, "by_camera": {"cam1": 2, "cam2": 1}}


def test_validation_rejects_bad_input():
    c = _client()
    assert c.post("/events", json={"camera": "cam1"}).status_code == 422                 # missing fields
    assert c.post("/events", json=_event(confidence=1.5)).status_code == 422             # out of range
    assert c.post("/events", json=_event(frame=-1)).status_code == 422                   # negative frame
    assert c.get("/events", params={"limit": 0}).status_code == 422
    assert c.get("/stats").json()["total"] == 0                                          # nothing was stored