"""CPU benchmark: PyTorch vs ONNX Runtime across input sizes, speed AND accuracy.

For every (backend, imgsz) it reports end-to-end latency per image
(preprocess + inference + postprocess), FPS and class-agnostic F1 on the
NEU-DET validation split, so the speed/accuracy trade-off is visible.

Needs:  pip install onnx onnxruntime
Usage (from repo root):
    python -m detect.benchmark
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from pathlib import Path

from classical.pipeline import load_gt_boxes
from detect.eval_compare import match_counts, prf

DATA = Path("data/neu-det")
OUT = Path("outputs")
SIZES = (640, 416, 320, 256)  # multiples of 32 (YOLO stride)
IOU_THR = 0.5
WARMUP = 5  # first images are excluded from timing (model/session warm-up)


def summarize(times: list[float]) -> tuple[float, float]:
    """Return (median, p95) of a list of timings."""
    s = sorted(times)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return statistics.median(s), p95


def run(model, paths, gts, imgsz: int, conf: float):
    totals = []
    tp = fp = fn = 0
    for i, (p, gt) in enumerate(zip(paths, gts)):
        r = model.predict(str(p), imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
        if i >= WARMUP:
            totals.append(sum(r.speed.values()))  # preprocess + inference + postprocess, ms
        pred = [tuple(int(round(v)) for v in b) for b in r.boxes.xyxy.tolist()]
        a, b, c = match_counts(gt, pred, IOU_THR)
        tp, fp, fn = tp + a, fp + b, fn + c
    med, p95 = summarize(totals)
    prec, rec, f1 = prf(tp, fp, fn)
    return med, p95, prec, rec, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/neu_v8s/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO

    img_dir, ann_dir = DATA / "valid_images", DATA / "valid_annotations"
    if not img_dir.exists():
        raise SystemExit(f"Not found: {img_dir}. Run from the repo root.")
    OUT.mkdir(exist_ok=True)
    paths = sorted(img_dir.glob("*.jpg"))
    gts = [load_gt_boxes(ann_dir / f"{p.stem}.xml") for p in paths]

    pt_model = YOLO(args.weights)
    rows = []
    for sz in SIZES:
        # PyTorch
        rows.append(("pytorch", sz, None, *run(pt_model, paths, gts, sz, args.conf)))
        # ONNX: export at this fixed size, keep one file per size
        exported = Path(pt_model.export(format="onnx", imgsz=sz))
        target = exported.with_name(f"best_{sz}.onnx")
        exported.replace(target)
        onnx_model = YOLO(str(target), task="detect")
        rows.append(("onnxruntime", sz, target.stat().st_size / 1e6,
                     *run(onnx_model, paths, gts, sz, args.conf)))

    csv_path = OUT / "benchmark_cpu.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "imgsz", "model_mb", "median_ms", "p95_ms", "fps", "precision", "recall", "f1_iou50"])
        for be, sz, mb, med, p95, pr, rc, f1 in rows:
            w.writerow([be, sz, "" if mb is None else f"{mb:.1f}", f"{med:.1f}", f"{p95:.1f}",
                        f"{1000 / med:.1f}", f"{pr:.3f}", f"{rc:.3f}", f"{f1:.3f}"])

    print(f"\nCPU: {os.cpu_count()} logical cores | {len(paths)} val images | conf={args.conf} | IoU>={IOU_THR}")
    print(f"{'backend':<13}{'imgsz':>6}{'med ms':>9}{'p95 ms':>9}{'FPS':>7}{'prec':>7}{'recall':>8}{'F1':>6}")
    for be, sz, mb, med, p95, pr, rc, f1 in rows:
        print(f"{be:<13}{sz:>6}{med:>9.1f}{p95:>9.1f}{1000 / med:>7.1f}{pr:>7.2f}{rc:>8.2f}{f1:>6.2f}")
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()