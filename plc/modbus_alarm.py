"""Alarm output over Modbus TCP: close a relay when a defect is counted.

Design rules for something that sits next to a production line:
  * trigger() NEVER blocks the vision loop: it only queues; a worker thread talks to the PLC.
  * A PLC that is off, unplugged or slow must not crash anything: failures are
    logged and counted, and the next trigger tries to reconnect.
  * The relay is a pulse: it closes, then opens again after `hold_s` seconds.
  * A late alarm points at the wrong piece of steel, so an alarm that waited longer
    than `max_age_s` (PLC down, backlog) is DROPPED and counted, never sent late.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time

from pymodbus.client import ModbusTcpClient

log = logging.getLogger("plc.alarm")


def channel_for(camera: str) -> int:
    """'cam3' -> 2 (zero-based coil / register address)."""
    m = re.search(r"(\d+)$", camera)
    if not m or int(m.group(1)) < 1:
        raise ValueError(f"cannot derive a channel from camera name {camera!r}")
    return int(m.group(1)) - 1


class ModbusAlarm:
    def __init__(self, host: str = "127.0.0.1", port: int = 5020, device_id: int = 1,
                 hold_s: float = 2.0, timeout: float = 1.0, max_age_s: float = 2.0):
        self.host, self.port, self.device_id, self.hold_s = host, port, device_id, hold_s
        self.max_age_s = max_age_s
        self._client = ModbusTcpClient(host, port=port, timeout=timeout)
        self._q: queue.Queue[tuple[str, float]] = queue.Queue()
        self._off_at: dict[int, float] = {}
        self._stop = threading.Event()
        self.errors = 0
        self.dropped = 0           # alarms discarded because they were too old to be useful
        self.triggers_sent = 0
        self._thread = threading.Thread(target=self._worker, name="modbus-alarm", daemon=True)
        self._thread.start()

    # -- public API ------------------------------------------------------
    def trigger(self, camera: str) -> None:
        """Queue an alarm pulse for this camera. Returns immediately."""
        self._q.put((camera, time.monotonic()))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        for ch in list(self._off_at):           # leave relays open on shutdown
            self._safe(lambda c=ch: self._client.write_coil(c, False, device_id=self.device_id))
        self._client.close()

    # -- worker ----------------------------------------------------------
    def _safe(self, op) -> bool:
        """Run one Modbus operation; any failure is logged and counted, never raised."""
        try:
            if not self._client.connected and not self._client.connect():
                raise ConnectionError(f"cannot connect to PLC at {self.host}:{self.port}")
            result = op()
            if result is None or result.isError():
                raise IOError(f"PLC returned an error: {result}")
            return True
        except Exception as exc:  # noqa: BLE001 - the vision loop must survive anything here
            self.errors += 1
            log.warning("Modbus operation failed (%s); will retry on the next alarm", exc)
            self._client.close()
            return False

    def _raise(self, camera: str, now: float) -> None:
        try:
            ch = channel_for(camera)
        except ValueError as exc:
            log.warning("%s", exc)
            return
        ok = self._safe(lambda: self._client.write_coil(ch, True, device_id=self.device_id))
        if not ok:
            return
        self.triggers_sent += 1
        self._off_at[ch] = now + self.hold_s
        # alarm counter register (+1, wraps at 16 bits)
        def bump():
            r = self._client.read_holding_registers(ch, count=1, device_id=self.device_id)
            if r.isError():
                return r
            return self._client.write_register(ch, (r.registers[0] + 1) & 0xFFFF, device_id=self.device_id)
        self._safe(bump)

    def _release_due(self, now: float) -> None:
        for ch, t in list(self._off_at.items()):
            if now >= t and self._safe(lambda c=ch: self._client.write_coil(c, False, device_id=self.device_id)):
                del self._off_at[ch]

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                cam, queued_at = self._q.get(timeout=0.05)
            except queue.Empty:
                cam, queued_at = None, 0.0
            now = time.monotonic()
            if cam is not None and now - queued_at > self.max_age_s:
                self.dropped += 1
                log.warning("dropping stale alarm for %s (waited %.1fs, limit %.1fs)",
                            cam, now - queued_at, self.max_age_s)
                cam = None
            if cam is not None:
                self._raise(cam, now)
            if self._off_at:
                self._release_due(now)