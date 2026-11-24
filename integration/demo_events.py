"""Demo: send synthetic defects to the API and the simulated PLC (no cameras, no YOLO).

Start these first, each in its own PowerShell window:
    python -m plc.sim_plc
    python -m uvicorn api.main:app
Then:
    python -m integration.demo_events
"""
from __future__ import annotations

import argparse
import logging
import random
import time

from integration.sinks import AlarmSink, ApiSink, DefectEvent, EventDispatcher
from plc.modbus_alarm import ModbusAlarm

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=12)
    ap.add_argument("--cameras", type=int, default=4)
    ap.add_argument("--delay", type=float, default=0.7, help="seconds between events")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--plc-port", type=int, default=5020)
    ap.add_argument("--alarm-classes", nargs="*", default=None,
                    help="only these classes close the relay (default: all)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)   # hide one log line per request

    api = ApiSink(args.api)
    alarm = ModbusAlarm(port=args.plc_port, hold_s=1.5)
    dispatcher = EventDispatcher(api, AlarmSink(alarm, set(args.alarm_classes) if args.alarm_classes else None))
    rng = random.Random(0)
    try:
        for i in range(args.events):
            ev = DefectEvent(camera=f"cam{rng.randint(1, args.cameras)}", defect_class=rng.choice(CLASSES),
                             track_id=i + 1, frame=100.0 + 37 * i, confidence=round(rng.uniform(0.3, 0.95), 2))
            print(f"event {i + 1}: {ev.camera} {ev.defect_class}")
            dispatcher.emit(ev)
            time.sleep(args.delay)
        time.sleep(2.0)   # let the last relay pulse finish
    finally:
        alarm.close()
    print(f"\nAPI: {api.sent} sent, {api.pending} pending, {api.failures} failures | "
        f"PLC: {alarm.triggers_sent} relay pulses, {alarm.errors} errors, {alarm.dropped} stale alarms dropped")


if __name__ == "__main__":
    main()