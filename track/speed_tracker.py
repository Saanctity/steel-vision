"""Tracker that uses the KNOWN belt speed instead of relying on box overlap.

On a real line the belt speed comes from an encoder / the PLC. Content moves up
the image by `speed` px per frame, so a track last seen at frame f0 should be
at (box shifted up by speed * (f - f0)) at frame f. Detections are matched to
those predicted boxes, which keeps IDs stable even when the camera is sampled
slowly and a small defect moves further than its own height between samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from classical.pipeline import iou


@dataclass
class _Track:
    id: int
    box: tuple
    last_idx: int
    label: Any = None   # e.g. (class name, confidence) of the latest matched detection


class SpeedTracker:
    def __init__(self, speed: float, iou_thr: float = 0.1, max_gap_frames: int = 80):
        self.speed = speed
        self.iou_thr = iou_thr
        self.max_gap = max_gap_frames
        self._tracks: list[_Track] = []
        self._next_id = 1

    def _predict(self, t: _Track, idx: int) -> tuple:
        dy = -self.speed * (idx - t.last_idx)
        x1, y1, x2, y2 = t.box
        return (x1, y1 + dy, x2, y2 + dy)

    def label_of(self, track_id: int):
        """Label attached to a live track by the latest update (None if unknown / forgotten)."""
        return next((t.label for t in self._tracks if t.id == track_id), None)

    def update(self, idx: int, boxes, labels=None) -> list[tuple[int, float, float]]:
        """boxes: xyxy detections at frame idx; labels: optional list aligned with boxes.
        Returns [(track_id, cx, cy)] for tracks seen now."""
        preds = [self._predict(t, idx) for t in self._tracks]
        pairs = sorted(((iou(p, b), ti, bi) for ti, p in enumerate(preds) for bi, b in enumerate(boxes)),
                       reverse=True)
        used_t, used_b = set(), set()
        for v, ti, bi in pairs:
            if v < self.iou_thr:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            self._tracks[ti].box = tuple(boxes[bi])
            self._tracks[ti].last_idx = idx
            if labels is not None:
                self._tracks[ti].label = labels[bi]
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                self._tracks.append(_Track(self._next_id, tuple(b), idx, labels[bi] if labels is not None else None))
                self._next_id += 1
        # forget tracks not seen for too long, or whose predicted box has left the image
        self._tracks = [t for t in self._tracks
                        if idx - t.last_idx <= self.max_gap and self._predict(t, idx)[3] > 0]
        return [(t.id, (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2)
                for t in self._tracks if t.last_idx == idx]