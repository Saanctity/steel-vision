"""Fair comparison: classical baseline vs YOLO on the NEU-DET validation split.

Both methods are scored by the same code: strict one-to-one matching of
predicted boxes to ground-truth boxes at a given IoU threshold.

Rows reported:
  classical      localisation only (it cannot say which defect type)
  yolo           localisation only (class ignored, same rule as classical)
  yolo+class     a match also requires the correct defect type

Usage (from repo root):
    python -m detect.eval_compare
    python -m detect.eval_compare --conf 0.25 --weights models/neu_v8s/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2

from classical.pipeline import ClassicalConfig, class_from_name, detect, iou, load_gt_boxes

log = logging.getLogger("detect.eval_compare")

DATA = Path("data/neu-det")
OUT = Path("outputs")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
IOU_THRESHOLDS = (0.3, 0.5)
CLASSICAL_KSIZE = 15  # best overall F1 from the sweep (picked on this same val set)


def match_counts(gt, pred, thr, scores=None):
    """One-to-one greedy matching. Returns (tp, fp, fn).

    Candidate pairs are taken in order of decreasing IoU; each ground-truth
    box and each prediction can be used only once.
    """
    pairs = sorted(
        ((iou(g, p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)),
        reverse=True,
    )
    used_g, used_p, tp = set(), set(), 0
    for v, gi, pi in pairs:
        if v < thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/neu_v8s/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from ultralytics import YOLO  # imported here so the pure functions test without it

    img_dir, ann_dir = DATA / "valid_images", DATA / "valid_annotations"
    if not img_dir.exists():
        raise SystemExit(f"Not found: {img_dir}. Run from the repo root.")
    OUT.mkdir(exist_ok=True)

    model = YOLO(args.weights)
    cfg = replace(ClassicalConfig(), morph_ksize=CLASSICAL_KSIZE)

    # counts[(method, thr, cls)] = [tp, fp, fn]
    counts: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])
    yolo_ms = []
    paths = sorted(img_dir.glob("*.jpg"))
    for p in paths:
        cls = class_from_name(p)
        cid = CLASSES.index(cls)
        gt = load_gt_boxes(ann_dir / f"{p.stem}.xml")
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

        cl_boxes, _ = detect(gray, cfg)

        res = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, device="cpu", verbose=False)[0]
        yolo_ms.append(res.speed["inference"])
        yb = [tuple(int(round(v)) for v in b) for b in res.boxes.xyxy.tolist()]
        yc = [int(c) for c in res.boxes.cls.tolist()]
        y_same = [b for b, c in zip(yb, yc) if c == cid]

        for thr in IOU_THRESHOLDS:
            for method, pred in (("classical", cl_boxes), ("yolo", yb), ("yolo+class", y_same)):
                tp, fp, fn = match_counts(gt, pred, thr)
                if method == "yolo+class":
                    # wrong-class boxes are false positives too
                    fp += len(yb) - len(y_same)
                for key in ((method, thr, cls), (method, thr, "ALL")):
                    c = counts[key]
                    c[0] += tp
                    c[1] += fp
                    c[2] += fn

    csv_path = OUT / "compare_classical_vs_yolo.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iou", "method", "class", "precision", "recall", "f1", "tp", "fp", "fn"])
        for (method, thr, cls), (tp, fp, fn) in sorted(counts.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
            pr, rc, f1 = prf(tp, fp, fn)
            w.writerow([thr, method, cls, f"{pr:.3f}", f"{rc:.3f}", f"{f1:.3f}", tp, fp, fn])

    print(f"\nValidation images: {len(paths)}   YOLO conf={args.conf}   classical ksize={CLASSICAL_KSIZE}")
    for thr in IOU_THRESHOLDS:
        print(f"\n=== IoU >= {thr} ===")
        print(f"{'class':<17}{'method':<12}{'prec':>6}{'recall':>8}{'F1':>6}")
        for cls in CLASSES + ["ALL"]:
            for method in ("classical", "yolo", "yolo+class"):
                tp, fp, fn = counts[(method, thr, cls)]
                pr, rc, f1 = prf(tp, fp, fn)
                print(f"{cls:<17}{method:<12}{pr:>6.2f}{rc:>8.2f}{f1:>6.2f}")
            print()
    print(f"YOLO CPU inference: {sum(yolo_ms) / len(yolo_ms):.1f} ms/image (mean, imgsz={args.imgsz})")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()