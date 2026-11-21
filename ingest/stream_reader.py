"""Threaded camera reader: keeps only the NEWEST frame, reconnects when a stream drops.

Why "newest frame only": if inference is slower than the camera, queuing frames
makes latency grow without bound. Dropping old frames keeps the system looking
at what is happening NOW, and counts the drops so they are visible.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

# Must be set before OpenCV opens its first stream: RTSP over TCP avoids UDP packet loss.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import cv2  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class StreamStats:
    frames: int = 0           # frames received from the camera
    reconnects: int = 0       # times the stream was re-established after a loss
    connected: bool = False
    last_frame_time: float = 0.0


class StreamReader(threading.Thread):
    def __init__(self, name: str, url: str, open_timeout_ms: int = 5000,
                 read_timeout_ms: int = 5000, backoff_max: float = 5.0):
        super().__init__(name=f"reader-{name}", daemon=True)
        self.cam_name = name
        self.url = url
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.backoff_max = backoff_max
        self.stats = StreamStats()
        self._lock = threading.Lock()
        self._latest: tuple[int, float, object] | None = None
        self._frame_id = 0
        self._stop_evt = threading.Event()
        self._ever_connected = False

    # -- consumer side ---------------------------------------------------
    def latest(self, after_id: int = 0):
        """Return (frame_id, capture_time, frame) if newer than after_id, else None."""
        with self._lock:
            if self._latest is None or self._latest[0] <= after_id:
                return None
            return self._latest

    def stop(self) -> None:
        self._stop_evt.set()

    # -- worker side -----------------------------------------------------
    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms,
        ])
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def run(self) -> None:
        delay = 0.5
        while not self._stop_evt.is_set():
            cap = self._open()
            if cap is None:
                self.stats.connected = False
                log.warning("[%s] cannot open stream, retrying in %.1fs", self.cam_name, delay)
                if self._stop_evt.wait(delay):
                    break
                delay = min(delay * 2, self.backoff_max)  # exponential backoff
                continue
            if self._ever_connected:
                self.stats.reconnects += 1
            self._ever_connected = True
            self.stats.connected = True
            delay = 0.5
            log.info("[%s] connected", self.cam_name)
            try:
                while not self._stop_evt.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        log.warning("[%s] stream lost", self.cam_name)
                        break
                    now = time.monotonic()
                    with self._lock:
                        self._frame_id += 1
                        self._latest = (self._frame_id, now, frame)
                    self.stats.frames += 1
                    self.stats.last_frame_time = now
            finally:
                cap.release()
                self.stats.connected = False