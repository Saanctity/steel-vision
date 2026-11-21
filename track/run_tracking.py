"""Track defects with YOLO + ByteTrack, count line crossings, score against truth.

For each simulated camera video the true crossings are known exactly (from the
JSON sidecar), so we report event precision / recall instead of just totals.
Scores are shown at several matching tolerances (a loose tolerance flatters the
result), plus per-class recall to show WHICH defect types are missed.

Usage (from repo root):
    python -m track.run_tracking                 # every 4th frame
    python -m track.run_tracking --stride 11     # ~2.2 FPS, like the 4-camera CPU run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2

from detect.eval_compare import prf
from track.counter import LineCounter, score_cameras, truth_events_cls

log = logging.getLogger("track.run_tracking")
OUT = Path("outputs")
TOLERANCES = [5, 8, 12, 30]  # frames; 1 frame = speed px of travel (4 px by default)


def process_video(video: Path, track_fn, stride: int, line_y: float):
    """Run track_fn(frame, frame_idx) on every `stride`-th frame. Returns (events, n_frames, n_processed)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    counter = LineCounter(line_y)
    idx = processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            counter.update(idx, track_fn(frame, idx))
            processed += 1
        idx += 1
    cap.release()
    return counter.events, idx, processed


def make_yolo_tracker(weights: str, imgsz: int, conf: float):
    """Fresh model per camera, so tracker state is never shared between cameras."""
    from ultralytics import YOLO
    model = YOLO(weights, task="detect")

    def track_fn(frame, _idx):
        r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                        imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
        if r.boxes.id is None:
            return []
        ids = r.boxes.id.int().tolist()
        return [(i, b[0], b[1]) for i, b in zip(ids, r.boxes.xywh.tolist())]
    return track_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=4)
    ap.add_argument("--videos", type=Path, default=Path("videos"))
    ap.add_argument("--weights", default="models/neu_v8s/weights/best_320.onnx")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=4, help="process every Nth frame")
    ap.add_argument("--line-y", type=float, default=100.0)
    ap.add_argument("--tol", type=float, default=8.0, help="main matching tolerance in frames")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)

    cam_data = {}
    for cam in range(1, args.cameras + 1):
        video = args.videos / f"cam{cam}.mp4"
        meta = json.loads((args.videos / f"cam{cam}.json").read_text())
        track_fn = make_yolo_tracker(args.weights, args.imgsz, args.conf)
        events, n_frames, processed = process_video(video, track_fn, args.stride, args.line_y)
        truth = truth_events_cls(meta["defects"], args.line_y, meta["speed_px_per_frame"], n_frames)
        cam_data[f"cam{cam}"] = {"truth": truth, "events": events, "n_frames": n_frames}
        print(f"cam{cam}: {processed}/{n_frames} frames processed, true {len(truth)}, counted {len(events)}", flush=True)

    # keep the raw events so results can be re-scored without re-running YOLO
    raw = OUT / f"tracking_events_stride{args.stride}.json"
    raw.write_text(json.dumps({c: {"events": d["events"], "truth": d["truth"], "n_frames": d["n_frames"]}
                               for c, d in cam_data.items()}))

    tols = sorted(set(TOLERANCES) | {args.tol})
    res = score_cameras(cam_data, tols, args.tol)

    csv_path = OUT / f"tracking_eval_stride{args.stride}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tolerance_frames", "tp", "fp", "fn", "precision", "recall", "f1"])
        for t in tols:
            tp, fp, fn = res["by_tol"][t]
            pr, rc, f1 = prf(tp, fp, fn)
            w.writerow([t, tp, fp, fn, f"{pr:.3f}", f"{rc:.3f}", f"{f1:.3f}"])

    print(f"\n=== stride={args.stride}  line_y={args.line_y} ===")
    print("\nSensitivity to matching tolerance (all cameras):")
    print(f"{'tol(frames)':>11}{'TP':>6}{'FP':>5}{'FN':>5}{'prec':>7}{'recall':>8}{'F1':>6}")
    for t in tols:
        tp, fp, fn = res["by_tol"][t]
        pr, rc, f1 = prf(tp, fp, fn)
        mark = "  <- main" if t == args.tol else ""
        print(f"{t:>11}{tp:>6}{fp:>5}{fn:>5}{pr:>7.2f}{rc:>8.2f}{f1:>6.2f}{mark}")

    print(f"\nPer camera (tolerance {args.tol:g} frames):")
    print(f"{'camera':<8}{'true':>6}{'count':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'prec':>7}{'recall':>8}{'F1':>6}")
    for cam, (n_true, n_found, tp, fp, fn) in res["per_camera"].items():
        pr, rc, f1 = prf(tp, fp, fn)
        print(f"{cam:<8}{n_true:>6}{n_found:>7}{tp:>5}{fp:>5}{fn:>5}{pr:>7.2f}{rc:>8.2f}{f1:>6.2f}")

    print(f"\nRecall per defect class (tolerance {args.tol:g} frames):")
    print(f"{'class':<17}{'true':>6}{'found':>7}{'recall':>8}")
    for cls, (n, m) in sorted(res["per_class"].items()):
        print(f"{cls:<17}{n:>6}{m:>7}{(m / n if n else 0):>8.2f}")
    print(f"\nSaved: {csv_path}\nSaved: {raw}")


if __name__ == "__main__":
    main()