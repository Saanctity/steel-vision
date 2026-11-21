"""Read several RTSP cameras and run YOLO on the newest frame of each.

Reports per camera: processed FPS, frames dropped (skipped because inference was
busy), reconnects, and capture-to-result latency (median / p95).

Usage (from repo root):
    python -m ingest.run_multicam --cameras 4 --seconds 30
    python -m ingest.run_multicam --cameras 4 --seconds 30 --no-detect   # ingest only
"""
from __future__ import annotations

import argparse
import csv
import logging
import statistics
import time
from pathlib import Path

import numpy as np

from ingest.stream_reader import StreamReader

log = logging.getLogger("ingest.run_multicam")
OUT = Path("outputs")
TILE = 200  # camera frame size used only for the warm-up frame


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def process_loop(readers, detector, seconds: float):
    """Round-robin over cameras; always take each camera's newest frame."""
    last = {r.cam_name: 0 for r in readers}
    acc = {r.cam_name: {"processed": 0, "dropped": 0, "detections": 0, "lat": []} for r in readers}
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        idle = True
        for r in readers:
            item = r.latest(last[r.cam_name])
            if item is None:
                continue
            idle = False
            fid, ts, frame = item
            a = acc[r.cam_name]
            if last[r.cam_name]:
                a["dropped"] += fid - last[r.cam_name] - 1
            last[r.cam_name] = fid
            dets = detector(frame)
            a["processed"] += 1
            a["detections"] += len(dets)
            a["lat"].append((time.monotonic() - ts) * 1000)
        if idle:
            time.sleep(0.002)
    elapsed = time.monotonic() - t0
    rows = []
    for r in readers:
        a = acc[r.cam_name]
        rows.append({
            "camera": r.cam_name,
            "proc_fps": a["processed"] / elapsed,
            "received": r.stats.frames,
            "dropped": a["dropped"],
            "reconnects": r.stats.reconnects,
            "detections": a["detections"],
            "lat_med_ms": statistics.median(a["lat"]) if a["lat"] else 0.0,
            "lat_p95_ms": percentile(a["lat"], 0.95),
        })
    return rows


def make_yolo_detector(weights: str, imgsz: int, conf: float):
    from ultralytics import YOLO
    model = YOLO(weights, task="detect")
    blank = np.zeros((TILE, TILE, 3), np.uint8)
    model.predict(blank, imgsz=imgsz, conf=conf, device="cpu", verbose=False)  # warm-up

    def detector(frame):
        r = model.predict(frame, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
        return r.boxes.xyxy.tolist()
    return detector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--weights", default="models/neu_v8s/weights/best_320.onnx")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-detect", action="store_true", help="measure ingest only, no model")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    detector = (lambda f: []) if args.no_detect else make_yolo_detector(args.weights, args.imgsz, args.conf)
    readers = [StreamReader(f"cam{i}", f"rtsp://{args.host}:{args.port}/cam{i}")
               for i in range(1, args.cameras + 1)]
    for r in readers:
        r.start()
    try:
        rows = process_loop(readers, detector, args.seconds)
    finally:
        for r in readers:
            r.stop()
        for r in readers:
            r.join(timeout=8)  # let threads leave OpenCV cleanly, avoids abort at exit

    OUT.mkdir(exist_ok=True)
    mode = "ingest_only" if args.no_detect else f"yolo_{args.imgsz}"
    csv_path = OUT / f"multicam_{args.cameras}cams_{mode}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.1f}" if isinstance(v, float) else v) for k, v in row.items()} for row in rows)
    print(f"\n{'camera':<8}{'proc FPS':>9}{'received':>9}{'dropped':>8}{'reconn':>7}{'dets':>6}{'lat med':>9}{'lat p95':>9}")
    for r in rows:
        print(f"{r['camera']:<8}{r['proc_fps']:>9.1f}{r['received']:>9}{r['dropped']:>8}"
              f"{r['reconnects']:>7}{r['detections']:>6}{r['lat_med_ms']:>8.0f}ms{r['lat_p95_ms']:>8.0f}ms")
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()