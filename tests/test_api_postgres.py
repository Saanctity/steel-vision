"""The same API against a real PostgreSQL server (skipped unless TEST_POSTGRES_URL is set).

    $env:TEST_POSTGRES_URL = "postgresql+psycopg://steel:steel@localhost:5432/steelvision"
    python -m pytest tests/test_api_postgres.py -q

WARNING: these tests DROP and recreate the `events` table. Use a throwaway database.
"""
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from api.main import Base, create_app

URL = os.environ.get("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TEST_POSTGRES_URL to run the PostgreSQL tests")


@pytest.fixture
def client():
    engine = create_engine(URL)
    Base.metadata.drop_all(engine)       # every test starts from an empty table
    engine.dispose()
    return TestClient(create_app(URL))


def _event(**kw):
    base = {"camera": "cam1", "defect_class": "scratches", "track_id": 7, "frame": 123.5, "confidence": 0.8}
    base.update(kw)
    return base


def test_roundtrip_with_utc_timestamp_and_null_confidence(client):
    r = client.post("/events", json=_event(confidence=None))
    assert r.status_code == 201
    stored = client.get("/events").json()[0]
    assert stored["confidence"] is None and stored["frame"] == 123.5
    assert stored["ts"].endswith("Z") or "+00:00" in stored["ts"]


def test_client_supplied_timestamp_is_kept_and_normalised_to_utc(client):
    client.post("/events", json=_event(ts="2026-01-02T03:04:05+05:30"))
    ts = client.get("/events").json()[0]["ts"]
    assert ts.startswith("2026-01-01T21:34:05")           # 03:04:05 at +05:30 is 21:34:05 UTC the day before


def test_newest_first_filters_and_stats(client):
    for cam, cls in [("cam1", "crazing"), ("cam2", "scratches"), ("cam2", "crazing")]:
        client.post("/events", json=_event(camera=cam, defect_class=cls))
    assert [e["id"] for e in client.get("/events").json()] == [3, 2, 1]
    assert [e["id"] for e in client.get("/events", params={"camera": "cam2"}).json()] == [3, 2]
    assert [e["id"] for e in client.get("/events", params={"defect_class": "crazing"}).json()] == [3, 1]
    assert client.get("/stats").json() == {"total": 3, "by_class": {"crazing": 2, "scratches": 1},
                                           "by_camera": {"cam1": 1, "cam2": 2}}