"""Live system: RTSP cameras -> detection -> speed tracking -> line counting -> API + PLC alarm.

Start first (each in its own window):
    python -m plc.sim_plc                      # simulated PLC
    python -m uvicorn api.main:app             # event-log API
    MediaMTX + python -m ingest.stream_videos  # the simulated cameras
Then:
    python -m live.run_live --cameras 4 --seconds 60

--detector classical uses the milestone-1 image-processing pipeline instead of YOLO
(handy for debugging the plumbing without a model).
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace

import cv2
import httpx
import numpy as np

from classical.pipeline import ClassicalConfig
from classical.pipeline import detect as classical_detect
from detect.prepare_data import CLASSES
from ingest.run_multicam import wait_until_connected
from ingest.stream_reader import StreamReader
from integration.sinks import AlarmSink, ApiSink, EventDispatcher
from live.pipeline import CameraPipeline
from plc.modbus_alarm import ModbusAlarm

log = logging.getLogger("live.run_live")


def make_yolo_detector(weights: str, imgsz: int, conf: float):
    from ultralytics import YOLO
    model = YOLO(weights, task="detect")
    model.predict(np.zeros((200, 200, 3), np.uint8), imgsz=imgsz, conf=conf, device="cpu", verbose=False)  # warm-up

    def detector(frame):
        r = model.predict(frame, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
        out = []
        for box, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            c = int(c)
            out.append((tuple(box), CLASSES[c] if 0 <= c < len(CLASSES) else str(c), float(p)))
        return out
    return detector


def make_classical_detector():
    cfg = replace(ClassicalConfig(), morph_ksize=15)

    def detector(frame):
        boxes, _ = classical_detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cfg)
        return [((float(x1), float(y1), float(x2), float(y2)), "defect", None) for x1, y1, x2, y2 in boxes]
    return detector


def run_loop(readers, pipelines, seconds: float) -> None:
    """Round-robin over cameras; always take each camera's newest frame."""
    last = {r.cam_name: 0 for r in readers}
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        idle = True
        for r, p in zip(readers, pipelines):
            item = r.latest(last[r.cam_name])
            if item is None:
                continue
            idle = False
            frame_id, _ts, frame = item
            last[r.cam_name] = frame_id
            p.process(frame_id, frame, reconnects=r.stats.reconnects)
        if idle:
            time.sleep(0.002)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--host", default="localhost", help="RTSP server host")
    ap.add_argument("--port", type=int, default=8554, help="RTSP server port")
    ap.add_argument("--detector", choices=["yolo", "classical"], default="yolo")
    ap.add_argument("--weights", default="models/neu_v8s/weights/best_320.onnx")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--speed", type=float, default=4.0, help="belt speed in px per source frame")
    ap.add_argument("--line-y", type=float, default=100.0)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--plc-port", type=int, default=5020)
    ap.add_argument("--alarm-classes", nargs="*", default=None, help="only these classes close the relay")
    ap.add_argument("--alarm-hold", type=float, default=2.0, help="seconds the relay stays closed per alarm")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:   # wake the API up first: a cold first request is slow and would time out
        httpx.get(f"{args.api}/health", timeout=10).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        log.warning("API not reachable yet (%s); events will be buffered until it is", exc)

    detector = make_classical_detector() if args.detector == "classical" else \
        make_yolo_detector(args.weights, args.imgsz, args.conf)
    api = ApiSink(args.api, timeout=3.0)
    alarm = ModbusAlarm(port=args.plc_port, hold_s=args.alarm_hold)
    dispatcher = EventDispatcher(api, AlarmSink(alarm, set(args.alarm_classes) if args.alarm_classes else None))

    names = [f"cam{i}" for i in range(1, args.cameras + 1)]
    readers = [StreamReader(n, f"rtsp://{args.host}:{args.port}/{n}") for n in names]
    pipelines = [CameraPipeline(n, detector, args.speed, args.line_y, dispatcher.emit) for n in names]
    for r in readers:
        r.start()
    try:
        if not wait_until_connected(readers):
            log.warning("not all cameras connected after 30s - running anyway")
        run_loop(readers, pipelines, args.seconds)
    finally:
        for r in readers:
            r.stop()
        for r in readers:
            r.join(timeout=8)
        api.flush()
        time.sleep(0.3)
        alarm.close()

    print(f"\n{'camera':<8}{'processed':>10}{'counted':>9}{'reconnects':>11}")
    for r, p in zip(readers, pipelines):
        print(f"{r.cam_name:<8}{p.processed:>10}{p.counted:>9}{r.stats.reconnects:>11}")
    print(f"\nAPI: {api.sent} sent, {api.pending} pending, {api.failures} failures | "
          f"PLC: {alarm.triggers_sent} relay pulses, {alarm.errors} errors, {alarm.dropped} stale alarms dropped")


if __name__ == "__main__":
    main()