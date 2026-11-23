"""Why are defects missed? Break counting recall down by box height and by crowding.

Uses the events saved by track.run_tracking plus the ground-truth sidecars, so
no model run is needed. Uses a generous tolerance on purpose: a miss here means
the defect was never counted at all, not that the count was slightly late.

Usage (from repo root):
    python -m track.diagnose --stride 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from track.counter import match_indices

OUT = Path("outputs")
HEIGHT_BUCKETS = [(0, 50, "<50 px"), (50, 100, "50-99 px"), (100, 150, "100-149 px"), (150, 10_000, ">=150 px")]


def annotate_truth(defects: list[dict], line_y: float, speed: float, n_frames: int, crowd_tol: float = 8.0):
    """True crossings with extra features: box height and whether another
    true crossing happens within `crowd_tol` frames (overlapping/neighbouring labels)."""
    items = []
    for d in defects:
        x1, y1, x2, y2 = d["box"]
        i = ((y1 + y2) / 2 - line_y) / speed
        if 0 <= i < n_frames:
            items.append({"frame": i, "cls": d["cls"], "h": y2 - y1})
    items.sort(key=lambda r: r["frame"])
    for k, r in enumerate(items):
        r["crowded"] = any(j != k and abs(items[j]["frame"] - r["frame"]) <= crowd_tol for j in range(len(items)))
    return items


def height_bucket(h: float) -> str:
    for lo, hi, name in HEIGHT_BUCKETS:
        if lo <= h < hi:
            return name
    return HEIGHT_BUCKETS[-1][2]


def breakdown(items: list[dict], matched: set[int]) -> dict:
    """Recall by height bucket and crowded/isolated. Values are [true, matched]."""
    by_height = {name: [0, 0] for _, _, name in HEIGHT_BUCKETS}
    by_crowd = {"isolated": [0, 0], "crowded": [0, 0]}
    for k, r in enumerate(items):
        hit = int(k in matched)
        for tbl, key in ((by_height, height_bucket(r["h"])), (by_crowd, "crowded" if r["crowded"] else "isolated")):
            tbl[key][0] += 1
            tbl[key][1] += hit
    return {"by_height": by_height, "by_crowd": by_crowd}


def merge(a: dict, b: dict) -> dict:
    for tbl in a:
        for key, (n, m) in b[tbl].items():
            a[tbl][key][0] += n
            a[tbl][key][1] += m
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--suffix", default="", help='"" for bytetrack runs, "_speed" for --tracker speed')
    ap.add_argument("--videos", type=Path, default=Path("videos"))
    ap.add_argument("--line-y", type=float, default=100.0)
    ap.add_argument("--tol", type=float, default=30.0)
    args = ap.parse_args()

    raw = json.loads((OUT / f"tracking_events_stride{args.stride}{args.suffix}.json").read_text())
    total = None
    for cam, d in raw.items():
        meta = json.loads((args.videos / f"{cam}.json").read_text())
        items = annotate_truth(meta["defects"], args.line_y, meta["speed_px_per_frame"], d["n_frames"])
        matched, _ = match_indices([r["frame"] for r in items], d["events"], args.tol)
        part = breakdown(items, matched)
        total = part if total is None else merge(total, part)

    print(f"Recall by what kind of defect (stride {args.stride}, tolerance {args.tol:g} frames):")
    for title, tbl in (("Box height", total["by_height"]), ("Neighbouring labels", total["by_crowd"])):
        print(f"\n{title}:")
        print(f"{'group':<14}{'true':>6}{'counted':>9}{'recall':>8}")
        for key, (n, m) in tbl.items():
            if n:
                print(f"{key:<14}{n:>6}{m:>9}{m / n:>8.2f}")


if __name__ == "__main__":
    main()