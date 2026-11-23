from classical.pipeline import iou
from track.counter import LineCounter
from track.speed_tracker import SpeedTracker


def _box_at(frame, y_strip=420.0, h=30, x=50, w=40, speed=4):
    y = y_strip - speed * frame        # content moves up by `speed` px per frame
    return (x, y, x + w, y + h)


def test_keeps_one_id_when_overlap_matching_would_fail():
    # stride 11 at 4 px/frame = 44 px jump, bigger than the 30 px box: zero overlap with the old position
    assert iou(_box_at(0), _box_at(11)) == 0.0
    tr = SpeedTracker(speed=4)
    ids = []
    for f in (0, 11, 22, 33):
        out = tr.update(f, [_box_at(f)])
        ids.append(out[0][0])
    assert ids == [1, 1, 1, 1]


def test_counts_a_fast_sampled_defect_exactly_once():
    tr, counter = SpeedTracker(speed=4), LineCounter(line_y=100)
    for f in range(0, 120, 11):        # defect centre crosses y=100 between processed frames
        counter.update(f, tr.update(f, [_box_at(f)]))
    assert len(counter.events) == 1


def test_two_defects_keep_separate_ids_and_new_ones_get_new_ids():
    tr = SpeedTracker(speed=4)
    a0 = tr.update(0, [_box_at(0, y_strip=400, x=10), _box_at(0, y_strip=520, x=120)])
    a1 = tr.update(8, [_box_at(8, y_strip=400, x=10), _box_at(8, y_strip=520, x=120)])
    assert sorted(i for i, _, _ in a0) == sorted(i for i, _, _ in a1) == [1, 2]
    a2 = tr.update(16, [_box_at(16, y_strip=400, x=10), _box_at(16, y_strip=520, x=120), _box_at(16, y_strip=470, x=70)])
    assert sorted(i for i, _, _ in a2) == [1, 2, 3]


def test_tolerates_detection_jitter():
    tr = SpeedTracker(speed=4)
    tr.update(0, [_box_at(0)])
    x1, y1, x2, y2 = _box_at(11)
    out = tr.update(11, [(x1 + 4, y1 - 5, x2 + 4, y2 - 5)])   # a few px of detector noise
    assert out[0][0] == 1


def test_track_is_dropped_after_leaving_the_image_or_a_long_gap():
    tr = SpeedTracker(speed=4, max_gap_frames=30)
    tr.update(0, [_box_at(0, y_strip=300)])
    tr.update(40, [])                  # not seen for 40 frames > max gap
    out = tr.update(41, [_box_at(41, y_strip=300)])
    assert out[0][0] == 2              # it is a new track, not the old one