from track.diagnose import annotate_truth, breakdown, height_bucket, merge


def test_annotate_truth_flags_crowded_and_height():
    defects = [
        {"cls": "a", "box": [0, 200, 10, 240]},   # centre 220, h 40  -> frame 30
        {"cls": "a", "box": [0, 210, 10, 250]},   # centre 230, h 40  -> frame 32.5 (crowded with the first)
        {"cls": "b", "box": [0, 600, 10, 800]},   # centre 700, h 200 -> frame 150 (isolated)
    ]
    items = annotate_truth(defects, line_y=100, speed=4, n_frames=1000)
    assert [round(r["frame"], 1) for r in items] == [30.0, 32.5, 150.0]
    assert [r["crowded"] for r in items] == [True, True, False]
    assert [r["h"] for r in items] == [40, 40, 200]


def test_breakdown_splits_recall_by_height_and_crowding():
    items = [{"frame": 1, "cls": "a", "h": 30, "crowded": False},
             {"frame": 2, "cls": "a", "h": 30, "crowded": False},
             {"frame": 3, "cls": "a", "h": 180, "crowded": True},
             {"frame": 4, "cls": "a", "h": 180, "crowded": True}]
    r = breakdown(items, matched={0, 1})           # found both small isolated ones, missed both big crowded ones
    assert r["by_height"]["<50 px"] == [2, 2]
    assert r["by_height"][">=150 px"] == [2, 0]
    assert r["by_crowd"] == {"isolated": [2, 2], "crowded": [2, 0]}


def test_height_bucket_edges_and_merge():
    assert [height_bucket(h) for h in (49, 50, 99, 100, 149, 150, 400)] == \
        ["<50 px", "50-99 px", "50-99 px", "100-149 px", "100-149 px", ">=150 px", ">=150 px"]
    a = breakdown([{"frame": 1, "cls": "a", "h": 10, "crowded": False}], {0})
    b = breakdown([{"frame": 1, "cls": "a", "h": 10, "crowded": False}], set())
    assert merge(a, b)["by_height"]["<50 px"] == [2, 1]