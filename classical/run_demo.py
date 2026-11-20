"""Milestone 1 runner: classical baseline on NEU-DET.

Usage (from repo root):
    python -m classical.run_demo
    python -m classical.run_demo --split train --per-class 4
"""
from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from classical.pipeline import ClassicalConfig, class_from_name, detect, iou, load_gt_boxes

log = logging.getLogger("classical.run_demo")
DATA = Path("data/neu-det")
OUT = Path("outputs")
IOU_HIT = 0.3  # classical boxes are loose, so a lenient match threshold


def draw(gray: np.ndarray, gt, pred, scale: int = 2) -> np.ndarray:
    img = cv2.cvtColor(cv2.resize(gray, None, fx=scale, fy=scale), cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in gt:    # green = ground truth
        cv2.rectangle(img, (x1 * scale, y1 * scale), (x2 * scale, y2 * scale), (0, 200, 0), 2)
    for x1, y1, x2, y2 in pred:  # red = classical prediction
        cv2.rectangle(img, (x1 * scale, y1 * scale), (x2 * scale, y2 * scale), (0, 0, 255), 1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["train", "valid"])
    ap.add_argument("--per-class", type=int, default=3, help="images per class in the montage")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    img_dir, ann_dir = DATA / f"{args.split}_images", DATA / f"{args.split}_annotations"
    if not img_dir.exists():
        raise SystemExit(f"Not found: {img_dir}. Run from the repo root.")
    OUT.mkdir(exist_ok=True)

    cfg = ClassicalConfig()
    by_class: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(img_dir.glob("*.jpg")):
        by_class[class_from_name(p)].append(p)

    stats = {}
    rows = []
    for cls, paths in sorted(by_class.items()):
        gt_total = hits = n_pred = 0
        tiles = []
        for i, p in enumerate(paths):
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            gt = load_gt_boxes(ann_dir / f"{p.stem}.xml")
            pred, _ = detect(gray, cfg)
            gt_total += len(gt)
            n_pred += len(pred)
            hits += sum(any(iou(g, q) >= IOU_HIT for q in pred) for g in gt)
            if i < args.per_class:
                tiles.append(draw(gray, gt, pred))
        stats[cls] = (len(paths), gt_total, hits, n_pred)
        rows.append(np.hstack(tiles))

    montage_path = OUT / f"classical_{args.split}.png"
    cv2.imwrite(str(montage_path), np.vstack(rows))

    csv_path = OUT / f"classical_{args.split}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "images", "gt_boxes", "gt_found", "recall", "pred_per_image"])
        print(f"\n{'class':<18}{'imgs':>5}{'recall':>9}{'pred/img':>10}")
        for cls, (n, gt_total, hits, n_pred) in stats.items():
            rec = hits / gt_total if gt_total else 0.0
            ppi = n_pred / n
            w.writerow([cls, n, gt_total, hits, f"{rec:.3f}", f"{ppi:.2f}"])
            print(f"{cls:<18}{n:>5}{rec:>9.2f}{ppi:>10.1f}")
    print(f"\nSaved: {montage_path}\nSaved: {csv_path}")


if __name__ == "__main__":
    main()