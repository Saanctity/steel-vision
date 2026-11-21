"""Simulate cameras watching a steel strip, from NEU-DET validation images.

Images are stacked into a long vertical strip; a 200x200 window slides along
it at a fixed speed, exactly like steel rolling under a camera. Each camera
gets its own random strip, plus a JSON sidecar with the true defect positions
(in strip coordinates) so counts can be checked later.

Only the *validation* split is used: the model never trained on it.

Usage (from repo root):
    python -m ingest.make_strip_video --cameras 4 --seconds 60
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from classical.pipeline import class_from_name, load_gt_boxes

log = logging.getLogger("ingest.make_strip_video")

TILE = 200  # NEU-DET images are 200x200
DATA = Path("data/neu-det")


def build_strip(img_paths: list[Path], ann_dir: Path, n_tiles: int, rng: np.random.Generator):
    """Stack n_tiles random images vertically. Returns (strip, defects).

    defects: list of {"cls": str, "box": [x1, y1, x2, y2]} in strip coordinates.
    """
    chosen = rng.choice(len(img_paths), size=n_tiles, replace=len(img_paths) < n_tiles)
    tiles, defects = [], []
    for k, idx in enumerate(chosen):
        p = img_paths[idx]
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (TILE, TILE):
            raise ValueError(f"{p.name}: expected a {TILE}x{TILE} grayscale image")
        tiles.append(gray)
        for x1, y1, x2, y2 in load_gt_boxes(ann_dir / f"{p.stem}.xml"):
            defects.append({"cls": class_from_name(p), "box": [x1, k * TILE + y1, x2, k * TILE + y2]})
    return np.vstack(tiles), defects


def write_camera(path: Path, strip: np.ndarray, fps: int, seconds: int, speed: int) -> int:
    """Slide a TILE-high window over the strip (wrapping around). Returns frame count."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (TILE, TILE))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    span = strip.shape[0] - TILE
    n_frames = fps * seconds
    for i in range(n_frames):
        y = (i * speed) % span
        writer.write(cv2.cvtColor(strip[y:y + TILE], cv2.COLOR_GRAY2BGR))
    writer.release()
    return n_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--speed", type=int, default=4, help="strip speed in px per frame")
    ap.add_argument("--out", type=Path, default=Path("videos"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    img_dir, ann_dir = DATA / "valid_images", DATA / "valid_annotations"
    paths = sorted(img_dir.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"No images in {img_dir}. Run from the repo root.")
    args.out.mkdir(exist_ok=True)

    travel_px = args.fps * args.seconds * args.speed
    n_tiles = max(2, -(-travel_px // TILE) + 1)  # ceil + 1, so the strip is longer than the travel
    for cam in range(1, args.cameras + 1):
        rng = np.random.default_rng(args.seed + cam)
        strip, defects = build_strip(paths, ann_dir, n_tiles, rng)
        video = args.out / f"cam{cam}.mp4"
        frames = write_camera(video, strip, args.fps, args.seconds, args.speed)
        (args.out / f"cam{cam}.json").write_text(json.dumps(
            {"fps": args.fps, "speed_px_per_frame": args.speed, "tile": TILE, "defects": defects}, indent=1))
        print(f"cam{cam}: {frames} frames, {n_tiles} tiles, {len(defects)} true defects -> {video}")


if __name__ == "__main__":
    main()