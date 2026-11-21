"""Line-crossing counter and ground-truth scoring for the simulated steel strip.

Geometry of the simulation: the camera window slides DOWN the strip at
`speed` px per frame, so content moves UP in the image. A defect with strip
centre y_c is at image row  y_c - speed * frame_index.
"""
from __future__ import annotations


class LineCounter:
    """Counts each track ID once, when its centre crosses line_y moving upward."""

    def __init__(self, line_y: float):
        self.line_y = line_y
        self._last_y: dict[int, float] = {}
        self._counted: set[int] = set()
        self.events: list[int] = []  # frame index at which each count happened

    def update(self, frame_idx: int, tracks) -> None:
        """tracks: iterable of (track_id, cx, cy) for this frame."""
        for tid, _cx, cy in tracks:
            prev = self._last_y.get(tid)
            self._last_y[tid] = cy
            if prev is not None and tid not in self._counted and prev > self.line_y >= cy:
                self._counted.add(tid)
                self.events.append(frame_idx)


def truth_events_cls(defects: list[dict], line_y: float, speed: float, n_frames: int) -> list[tuple[float, str]]:
    """(frame index, class) at which each true defect's centre reaches the line."""
    out = []
    for d in defects:
        cy = (d["box"][1] + d["box"][3]) / 2
        i = (cy - line_y) / speed
        if 0 <= i < n_frames:
            out.append((i, d["cls"]))
    return sorted(out)


def truth_events(defects: list[dict], line_y: float, speed: float, n_frames: int) -> list[float]:
    return [t for t, _ in truth_events_cls(defects, line_y, speed, n_frames)]


def match_indices(truth: list[float], found: list[float], tol: float) -> tuple[set[int], set[int]]:
    """One-to-one matching within `tol` frames, closest pairs first.

    Returns (indices of matched truth events, indices of matched found events).
    """
    pairs = sorted((abs(t - f), ti, fi) for ti, t in enumerate(truth) for fi, f in enumerate(found))
    used_t: set[int] = set()
    used_f: set[int] = set()
    for dist, ti, fi in pairs:
        if dist > tol:
            break
        if ti in used_t or fi in used_f:
            continue
        used_t.add(ti)
        used_f.add(fi)
    return used_t, used_f


def match_events(truth: list[float], found: list[float], tol: float) -> tuple[int, int, int]:
    """Returns (tp, fp, fn)."""
    used_t, _ = match_indices(truth, found, tol)
    tp = len(used_t)
    return tp, len(found) - tp, len(truth) - tp


def evaluate(truth: list[tuple[float, str]], found: list[float], tol: float) -> dict:
    """tp/fp/fn plus per-class [true, matched] counts."""
    used_t, _ = match_indices([t for t, _ in truth], found, tol)
    per_class: dict[str, list[int]] = {}
    for ti, (_, cls) in enumerate(truth):
        c = per_class.setdefault(cls, [0, 0])
        c[0] += 1
        c[1] += ti in used_t
    tp = len(used_t)
    return {"tp": tp, "fp": len(found) - tp, "fn": len(truth) - tp, "per_class": per_class}


def score_cameras(cam_data: dict, tols: list[float], main_tol: float) -> dict:
    """cam_data: {cam: {"truth": [(frame, cls)], "events": [frame]}}.

    Returns {"by_tol": {tol: (tp, fp, fn)}, "per_camera": {cam: (n_true, n_found, tp, fp, fn)},
             "per_class": {cls: [true, matched]}} (per_camera and per_class at main_tol).
    """
    by_tol = {t: [0, 0, 0] for t in tols}
    per_camera, per_class = {}, {}
    for cam, d in cam_data.items():
        for t in tols:
            r = evaluate(d["truth"], d["events"], t)
            for k, key in enumerate(("tp", "fp", "fn")):
                by_tol[t][k] += r[key]
        r = evaluate(d["truth"], d["events"], main_tol)
        per_camera[cam] = (len(d["truth"]), len(d["events"]), r["tp"], r["fp"], r["fn"])
        for cls, (n, m) in r["per_class"].items():
            c = per_class.setdefault(cls, [0, 0])
            c[0] += n
            c[1] += m
    return {"by_tol": {t: tuple(v) for t, v in by_tol.items()}, "per_camera": per_camera, "per_class": per_class}