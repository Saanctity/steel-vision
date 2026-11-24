import httpx
from fastapi.testclient import TestClient

from api.main import create_app
from integration.sinks import AlarmSink, ApiSink, DefectEvent, EventDispatcher


def _ev(**kw):
    base = dict(camera="cam1", defect_class="scratches", track_id=1, frame=10.0, confidence=0.7)
    base.update(kw)
    return DefectEvent(**base)


class FakeAlarm:
    def __init__(self):
        self.calls = []

    def trigger(self, camera):
        self.calls.append(camera)


def test_api_sink_delivers_events_with_detection_time():
    client = TestClient(create_app("sqlite:///:memory:"))
    sink = ApiSink(client=client)
    sink(_ev(track_id=5))
    assert sink.sent == 1 and sink.pending == 0
    stored = client.get("/events").json()[0]
    assert stored["track_id"] == 5 and stored["ts"]


def test_api_sink_buffers_while_api_is_down_then_delivers_in_order():
    state = {"up": False}
    app = TestClient(create_app("sqlite:///:memory:"))
    now = {"t": 0.0}

    def handler(request: httpx.Request) -> httpx.Response:
        if not state["up"]:
            raise httpx.ConnectError("api down")
        r = app.post(str(request.url.path), content=request.content, headers={"content-type": "application/json"})
        return httpx.Response(r.status_code, content=r.content)

    client = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    sink = ApiSink(client=client, retry_after_s=5.0, clock=lambda: now["t"])
    sink(_ev(track_id=1))
    sink(_ev(track_id=2))                      # inside the retry back-off: not even attempted
    assert sink.pending == 2 and sink.failures == 1
    state["up"], now["t"] = True, 6.0          # API recovers, back-off has passed
    sink(_ev(track_id=3))
    assert sink.pending == 0 and sink.sent == 3
    ids_in_db_order = sorted(app.get("/events").json(), key=lambda e: e["id"])
    assert [e["track_id"] for e in ids_in_db_order] == [1, 2, 3]   # delivered oldest first


def test_api_sink_drops_events_the_api_rejects_instead_of_retrying_forever():
    client = TestClient(create_app("sqlite:///:memory:"))
    sink = ApiSink(client=client)
    sink(_ev(confidence=5.0))                  # invalid: confidence > 1 -> HTTP 422
    assert sink.pending == 0 and sink.sent == 0


def test_alarm_sink_respects_class_filter():
    a = FakeAlarm()
    sink = AlarmSink(a, alarm_classes={"crazing"})
    sink(_ev(defect_class="scratches"))
    sink(_ev(defect_class="crazing", camera="cam3"))
    assert a.calls == ["cam3"]
    all_sink = AlarmSink(a)
    all_sink(_ev(camera="cam2"))
    assert a.calls == ["cam3", "cam2"]


def test_dispatcher_isolates_a_failing_sink():
    got = []

    def broken(ev):
        raise RuntimeError("boom")

    EventDispatcher(broken, got.append).emit(_ev())
    assert len(got) == 1