import json

import cv2
import numpy as np

from ingest.make_strip_video import TILE, build_strip, write_camera
from track.counter import LineCounter, evaluate, match_events, score_cameras, truth_events
from track.run_tracking import process_video


def test_counts_each_id_once_when_moving_up_through_line():
    c = LineCounter(line_y=100)
    c.update(0, [(1, 50, 150)])
    c.update(1, [(1, 50, 120)])
    c.update(2, [(1, 50, 90)])   # crosses here
    c.update(3, [(1, 50, 70)])
    c.update(4, [(1, 50, 120)])  # moves back down...
    c.update(5, [(1, 50, 90)])   # ...and crosses again: must NOT recount
    assert len(c.events) == 1
    assert abs(c.events[0] - (1 + 20 / 30)) < 1e-9   # interpolated between frames 1 and 2


def test_ignores_downward_motion_and_new_ids_without_history():
    c = LineCounter(line_y=100)
    c.update(0, [(7, 0, 50)])
    c.update(1, [(7, 0, 150)])   # downward crossing
    c.update(2, [(8, 0, 90)])    # first sighting already above the line
    assert c.events == []


def test_truth_events_geometry():
    defects = [{"cls": "a", "box": [0, 200, 10, 240]},    # centre y=220 -> frame (220-100)/4 = 30
               {"cls": "a", "box": [0, 40, 10, 60]},      # centre 50 -> already above the line: excluded
               {"cls": "a", "box": [0, 5000, 10, 5040]}]  # centre 5020 -> frame 1230 > n_frames: excluded
    assert truth_events(defects, line_y=100, speed=4, n_frames=1000) == [30.0]


def test_match_events_is_one_to_one_and_respects_tolerance():
    assert match_events([10, 50], [12, 49], tol=5) == (2, 0, 0)
    assert match_events([10], [10, 11], tol=5) == (1, 1, 0)   # extra count is a false positive
    assert match_events([10, 80], [12], tol=5) == (1, 0, 1)   # missed one
    assert match_events([10], [40], tol=5) == (0, 1, 1)       # too far apart


def test_pipeline_with_oracle_tracker_recovers_truth(tmp_path):
    """Oracle tracker = perfect detections from the sidecar. Checks video reading,
    striding, line crossing and the truth formula all agree with each other."""
    img_dir, ann_dir = tmp_path / "imgs", tmp_path / "anns"
    img_dir.mkdir()
    ann_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(img_dir / f"scratches_{i}.jpg"), np.full((TILE, TILE), 100 + 20 * i, np.uint8))
        (ann_dir / f"scratches_{i}.xml").write_text(
            "<annotation><object><bndbox><xmin>10</xmin><ymin>20</ymin>"
            "<xmax>60</xmax><ymax>60</ymax></bndbox></object></annotation>")
    paths = sorted(img_dir.glob("*.jpg"))
    speed, fps, seconds = 4, 10, 20            # 800 px of travel
    strip, defects = build_strip(paths, ann_dir, 6, np.random.default_rng(0))  # 1200 px strip
    video = tmp_path / "cam.mp4"
    write_camera(video, strip, fps=fps, seconds=seconds, speed=speed)

    def oracle(_frame, idx):
        window_y = idx * speed
        out = []
        for j, d in enumerate(defects):
            cy = (d["box"][1] + d["box"][3]) / 2 - window_y
            if 0 <= cy < TILE:
                out.append((j, (d["box"][0] + d["box"][2]) / 2, cy))
        return out

    events, n_frames, processed = process_video(video, oracle, stride=2, line_y=100)
    truth = truth_events(defects, 100, speed, n_frames)
    tp, fp, fn = match_events(truth, events, tol=3)
    assert n_frames == fps * seconds and processed == n_frames // 2
    assert len(truth) > 0
    assert (fp, fn) == (0, 0) and tp == len(truth)


def test_evaluate_reports_per_class_recall():
    truth = [(10.0, "crazing"), (50.0, "scratches"), (90.0, "crazing")]
    r = evaluate(truth, found=[11.0, 51.0], tol=3)   # crazing@90 is missed
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 1)
    assert r["per_class"] == {"crazing": [2, 1], "scratches": [1, 1]}


def test_tolerance_changes_the_verdict():
    truth = [(10.0, "a"), (20.0, "a")]
    found = [14.0]                                   # 4 frames from the first truth event
    assert evaluate(truth, found, tol=3)["tp"] == 0  # tight: a false positive
    assert evaluate(truth, found, tol=5)["tp"] == 1  # loose: counted as correct


def test_score_cameras_aggregates_over_cameras_and_tolerances():
    cam_data = {"cam1": {"truth": [(10.0, "a")], "events": [10.0]},
                "cam2": {"truth": [(10.0, "b"), (40.0, "b")], "events": [11.0, 90.0]}}
    res = score_cameras(cam_data, tols=[2, 100], main_tol=2)
    assert res["by_tol"][2] == (2, 1, 1)      # tp, fp, fn
    assert res["by_tol"][100] == (3, 0, 0)    # loose enough to match the stray count
    assert res["per_camera"]["cam2"] == (2, 2, 1, 1, 1)
    assert res["per_class"] == {"a": [1, 1], "b": [2, 1]}


def test_crossing_time_is_interpolated_not_delayed_by_stride():
    c = LineCounter(line_y=100)
    c.update(0, [(1, 0, 150)])
    c.update(10, [(1, 0, 50)])       # seen again 10 frames later, well past the line
    assert c.events == [5.0]         # it crossed halfway between the two sightings