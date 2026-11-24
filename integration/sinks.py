"""Where a counted defect goes: the event-log API and the PLC alarm.

The vision loop builds one DefectEvent per counted defect and hands it to an
EventDispatcher. Each sink is isolated: if one fails (API down, PLC unplugged)
the others still run and the vision loop is never interrupted.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from plc.modbus_alarm import ModbusAlarm

log = logging.getLogger("integration.sinks")


@dataclass
class DefectEvent:
    camera: str
    defect_class: str
    track_id: int
    frame: float
    confidence: float | None = None
    ts: datetime | None = None


class ApiSink:
    """POST events to the API. Keeps events in memory while the API is down.

    Timestamps are stamped HERE (detection time), so a buffered event keeps its
    real time instead of the time it finally got delivered.
    After a failure it waits `retry_after_s` before trying again, so a dead API
    does not slow the vision loop with a timeout on every defect.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", client: httpx.Client | None = None,
                 timeout: float = 1.0, max_buffer: int = 1000, retry_after_s: float = 5.0, clock=time.monotonic):
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)
        self._buffer: deque[dict] = deque(maxlen=max_buffer)
        self._retry_after = retry_after_s
        self._clock = clock
        self._next_try = 0.0
        self.sent = 0
        self.failures = 0

    @property
    def pending(self) -> int:
        return len(self._buffer)

    def __call__(self, ev: DefectEvent) -> None:
        ts = (ev.ts or datetime.now(timezone.utc)).isoformat()
        self._buffer.append({"camera": ev.camera, "defect_class": ev.defect_class, "track_id": ev.track_id,
                             "frame": ev.frame, "confidence": ev.confidence, "ts": ts})
        self.flush()

    def flush(self) -> None:
        while self._buffer and self._clock() >= self._next_try:
            try:
                r = self._client.post("/events", json=self._buffer[0])
                if r.status_code == 422:        # permanently bad event: drop it, do not retry forever
                    log.warning("API rejected an event, dropping it: %s", r.text[:200])
                    self._buffer.popleft()
                    continue
                r.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                self.failures += 1
                self._next_try = self._clock() + self._retry_after
                log.warning("API unreachable (%s); %d event(s) buffered", exc, len(self._buffer))
                return
            self._buffer.popleft()
            self.sent += 1


class AlarmSink:
    """Close the PLC relay for the event's camera. alarm_classes=None means every defect."""

    def __init__(self, alarm: ModbusAlarm, alarm_classes: set[str] | None = None):
        self.alarm = alarm
        self.alarm_classes = alarm_classes

    def __call__(self, ev: DefectEvent) -> None:
        if self.alarm_classes is None or ev.defect_class in self.alarm_classes:
            self.alarm.trigger(ev.camera)


class EventDispatcher:
    def __init__(self, *sinks):
        self.sinks = sinks

    def emit(self, ev: DefectEvent) -> None:
        for sink in self.sinks:
            try:
                sink(ev)
            except Exception:  # noqa: BLE001 - one broken sink must not stop the others
                log.exception("sink %s failed", type(sink).__name__)