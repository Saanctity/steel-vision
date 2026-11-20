"""Parameter sweep: how does the morphology kernel size trade precision vs recall?

Usage (from repo root):
    python -m classical.sweep
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2

from classical.pipeline import ClassicalConfig, class_from_name, detect, iou, load_gt_boxes

DATA = Path("data/neu-det")
OUT = Path("outputs")
IOU_HIT = 0.3
KERNELS = [3, 5, 9, 15, 21]


def evaluate(paths, ann_dir, cfg):
    """Return (gt_total, gt_found, pred_total, pred_correct) for these images."""
    gt_total = gt_found = pred_total = pred_ok = 0
    for p in paths:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        gt = load_gt_boxes(ann_dir / f"{p.stem}.xml")
        pred, _ = detect(gray, cfg)
        gt_total += len(gt)
        pred_total += len(pred)
        gt_found += sum(any(iou(g, q) >= IOU_HIT for q in pred) for g in gt)
        pred_ok += sum(any(iou(g, q) >= IOU_HIT for g in gt) for q in pred)
    return gt_total, gt_found, pred_total, pred_ok


def prf(gt_total, gt_found, pred_total, pred_ok):
    rec = gt_found / gt_total if gt_total else 0.0
    prec = pred_ok / pred_total if pred_total else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main() -> None:
    img_dir, ann_dir = DATA / "valid_images", DATA / "valid_annotations"
    if not img_dir.exists():
        raise SystemExit(f"Not found: {img_dir}. Run from the repo root.")
    OUT.mkdir(exist_ok=True)

    by_class = defaultdict(list)
    for p in sorted(img_dir.glob("*.jpg")):
        by_class[class_from_name(p)].append(p)
    all_paths = [p for ps in by_class.values() for p in ps]

    base = ClassicalConfig()
    rows = []
    for k in KERNELS:
        cfg = replace(base, morph_ksize=k)
        for cls, paths in sorted(by_class.items()):
            rows.append((k, cls, *prf(*evaluate(paths, ann_dir, cfg))))
        rows.append((k, "ALL", *prf(*evaluate(all_paths, ann_dir, cfg))))

    csv_path = OUT / "classical_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["morph_ksize", "class", "precision", "recall", "f1"])
        for k, cls, pr, rc, f1 in rows:
            w.writerow([k, cls, f"{pr:.3f}", f"{rc:.3f}", f"{f1:.3f}"])

    print(f"\n{'ksize':>5}  {'class':<16}{'prec':>6}{'recall':>8}{'F1':>6}")
    for k, cls, pr, rc, f1 in rows:
        print(f"{k:>5}  {cls:<16}{pr:>6.2f}{rc:>8.2f}{f1:>6.2f}")
        if cls == "ALL":
            print()
    best = max((r for r in rows if r[1] == "ALL"), key=lambda r: r[4])
    print(f"Best overall F1: morph_ksize={best[0]} (F1={best[4]:.2f})")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()