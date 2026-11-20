"""Convert NEU-DET (Pascal-VOC XML) into the YOLO dataset layout.

Output:
    datasets/neu-det-yolo/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt      (class cx cy w h, all normalised 0..1)
        data.yaml

NEU-DET's own "valid" split is used as "val". Each image holds a single
defect class, which we take from the filename (e.g. rolled-in_scale_10.jpg).

Usage (from repo root):
    python -m detect.prepare_data
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2

from classical.pipeline import class_from_name, load_gt_boxes

log = logging.getLogger("detect.prepare_data")

SRC = Path("data/neu-det")
DST = Path("datasets/neu-det-yolo")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SPLITS = {"train": "train", "val": "valid"}  # yolo name -> neu-det folder prefix


def convert_split(yolo_split: str, src_prefix: str) -> tuple[int, int]:
    img_src = SRC / f"{src_prefix}_images"
    ann_src = SRC / f"{src_prefix}_annotations"
    img_dst = DST / "images" / yolo_split
    lbl_dst = DST / "labels" / yolo_split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    n_img = n_box = 0
    for img_path in sorted(img_src.glob("*.jpg")):
        cls = class_from_name(img_path)
        if cls not in CLASSES:
            raise ValueError(f"Unknown class '{cls}' from {img_path.name}")
        cid = CLASSES.index(cls)
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        lines = []
        for x1, y1, x2, y2 in load_gt_boxes(ann_src / f"{img_path.stem}.xml"):
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                log.warning("skipping degenerate box in %s", img_path.name)
                continue
            lines.append(f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        shutil.copy2(img_path, img_dst / img_path.name)
        (lbl_dst / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        n_box += len(lines)
    return n_img, n_box


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not SRC.exists():
        raise SystemExit(f"Not found: {SRC}. Run from the repo root.")
    for yolo_split, prefix in SPLITS.items():
        n_img, n_box = convert_split(yolo_split, prefix)
        print(f"{yolo_split}: {n_img} images, {n_box} boxes")
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (DST / "data.yaml").write_text(f"path: .\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print(f"Wrote {DST / 'data.yaml'}")


if __name__ == "__main__":
    main()