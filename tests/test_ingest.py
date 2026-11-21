import time
from pathlib import Path

import cv2
import numpy as np

from ingest.make_strip_video import TILE, write_camera
from ingest.run_multicam import percentile, process_loop, wait_until_connected
from ingest.stream_reader import StreamReader
from ingest.stream_videos import ffmpeg_cmd


class FakeReader:
    """Pretends a camera that delivers 3 new frames every time it is polled."""
    def __init__(self, name):
        self.cam_name = name
        self.stats = type("S", (), {"frames": 0, "reconnects": 0})()

    def latest(self, after_id=0):
        return after_id + 3, time.monotonic(), "frame"


def test_ffmpeg_cmd_publishes_h264_over_rtsp_tcp():
    cmd = ffmpeg_cmd("ffmpeg", Path("videos/cam1.mp4"), "rtsp://localhost:8554/cam1")
    assert "libx264" in cmd and "-stream_loop" in cmd and "-re" in cmd
    assert cmd[-1] == "rtsp://localhost:8554/cam1"
    assert cmd[cmd.index("-rtsp_transport") + 1] == "tcp"


def test_percentile():
    assert percentile([], 0.95) == 0.0
    assert percentile([5, 1, 3, 2, 4], 0.95) == 5


def test_process_loop_counts_dropped_frames():
    rows = process_loop([FakeReader("camA")], lambda f: [1, 2], seconds=0.1)
    r = rows[0]
    processed = r["detections"] // 2          # the detector returns 2 boxes per frame
    assert processed > 1
    assert r["dropped"] == 2 * (processed - 1)  # fake camera skips 2 frames between polls


def test_reader_reads_frames_from_a_file(tmp_path):
    strip = np.random.default_rng(0).integers(0, 255, (4 * TILE, TILE), dtype=np.uint8)
    video = tmp_path / "cam.mp4"
    write_camera(video, strip, fps=10, seconds=2, speed=4)
    reader = StreamReader("t", str(video))
    reader.start()
    deadline = time.monotonic() + 10
    while reader.stats.frames < 5 and time.monotonic() < deadline:
        time.sleep(0.05)
    reader.stop()
    reader.join(timeout=8)
    assert reader.stats.frames >= 5
    item = reader.latest(0)
    assert item is not None and item[2].shape[:2] == (TILE, TILE)
    assert reader.latest(item[0]) is None  # nothing newer than what we already took


def test_reader_survives_unreachable_source_and_stops():
    reader = StreamReader("bad", "rtsp://127.0.0.1:1/nothing", open_timeout_ms=500, read_timeout_ms=500)
    reader.start()
    time.sleep(1.5)
    assert reader.stats.connected is False and reader.stats.frames == 0
    reader.stop()
    reader.join(timeout=8)
    assert not reader.is_alive()


def test_wait_until_connected():
    class R:
        def __init__(self, connected):
            self.stats = type("S", (), {"connected": connected})            ()

    assert wait_until_connected([R(True), R(True)], timeout=0.5)
    assert not wait_until_connected([R(True), R(False)],            timeout=0.3)