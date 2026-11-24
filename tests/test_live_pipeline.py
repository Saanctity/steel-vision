from live.pipeline import CameraPipeline
from track.counter import LineCounter
from track.speed_tracker import SpeedTracker


def _moving_detector(speed=4, y0=300.0, h=30):
    """A fake detector: one defect whose top edge is at y0 at frame 0, moving up `speed` px/frame."""
    state = {"frame": 0}

    def det(_frame):
        y = y0 - speed * state["frame"]
        return [((50.0, y, 90.0, y + h), "crazing", 0.9)]
    return det, state


def test_speed_tracker_remembers_labels():
    tr = SpeedTracker(speed=4)
    out = tr.update(0, [(10, 100, 40, 130)], [("scratches", 0.7)])
    tid = out[0][0]
    assert tr.label_of(tid) == ("scratches", 0.7)
    tr.update(5, [(10, 80, 40, 110)], [("patches", 0.5)])   # same defect, updated label
    assert tr.label_of(tid) == ("patches", 0.5)
    assert tr.label_of(999) is None
    assert tr.update(6, [(10, 70, 40, 100)])[0][0] == tid    # labels stay optional


def test_line_counter_announces_each_count_once():
    got = []
    c = LineCounter(100, on_count=lambda tid, frame: got.append((tid, frame)))
    c.update(0, [(1, 0, 150)])
    c.update(10, [(1, 0, 50)])
    c.update(20, [(1, 0, 20)])
    assert got == [(1, 5.0)]


def test_pipeline_emits_one_event_with_class_and_confidence():
    det, state = _moving_detector()
    events = []
    p = CameraPipeline("cam2", det, speed=4, line_y=100, emit=events.append)
    for f in range(0, 200, 11):          # sampled every 11th frame
        state["frame"] = f
        p.process(f, frame=None)
    assert len(events) == 1 and p.counted == 1 and p.processed == len(range(0, 200, 11))
    ev = events[0]
    assert (ev.camera, ev.defect_class, ev.confidence) == ("cam2", "crazing", 0.9)
    # centre starts at 315 and reaches y=100 after (315-100)/4 = 53.75 frames
    assert abs(ev.frame - 53.75) < 1.0


def test_pipeline_does_not_link_sightings_across_a_reconnect():
    # centre is at y=139 at frame 44 and y=95 at frame 55: it crosses the line y=100 in between
    det, state = _moving_detector()
    p = CameraPipeline("cam1", det, speed=4, line_y=100, emit=lambda ev: None)
    state["frame"] = 44
    p.process(44, None)
    first_tracker = p.tracker
    state["frame"] = 55
    p.process(55, None, reconnects=1)          # the stream dropped and came back in between
    assert p.tracker is not first_tracker      # state was reset
    assert p.counted == 0                      # no count built from a pre-gap sighting

    # control: the same two sightings WITHOUT a reconnect do produce the count
    det2, state2 = _moving_detector()
    q = CameraPipeline("cam1", det2, speed=4, line_y=100, emit=lambda ev: None)
    state2["frame"] = 44
    q.process(44, None)
    state2["frame"] = 55
    q.process(55, None, reconnects=0)
    assert q.counted == 1