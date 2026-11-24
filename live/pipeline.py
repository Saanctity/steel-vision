"""One camera's path from frame to counted-defect event.

frame -> detector -> SpeedTracker (known belt speed) -> LineCounter -> emit(DefectEvent)

`detector(frame)` returns [(xyxy box, class name, confidence), ...], so a real
YOLO model and a test stand-in plug in the same way.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from integration.sinks import DefectEvent
from track.counter import LineCounter
from track.speed_tracker import SpeedTracker

log = logging.getLogger("live.pipeline")


class CameraPipeline:
    def __init__(self, camera: str, detector: Callable, speed: float, line_y: float,
                 emit: Callable[[DefectEvent], None]):
        self.camera, self.detector, self.speed, self.line_y, self.emit = camera, detector, speed, line_y, emit
        self.processed = 0
        self.counted = 0
        self._reconnects_seen = 0
        self._reset()

    def _reset(self) -> None:
        self.tracker = SpeedTracker(self.speed)
        self.counter = LineCounter(self.line_y, on_count=self._on_count)

    def process(self, frame_id: int, frame, reconnects: int = 0) -> None:
        """frame_id is the SOURCE frame index (the tracker uses its differences as elapsed time)."""
        if reconnects != self._reconnects_seen:
            # frames were lost while the stream was down, so old tracks no longer line up
            log.info("[%s] stream re-established: resetting tracks", self.camera)
            self._reconnects_seen = reconnects
            self._reset()
        dets = self.detector(frame)
        tracks = self.tracker.update(frame_id, [d[0] for d in dets], [(d[1], d[2]) for d in dets])
        self.counter.update(frame_id, tracks)
        self.processed += 1

    def _on_count(self, track_id: int, crossing_frame: float) -> None:
        cls, conf = self.tracker.label_of(track_id) or ("unknown", None)
        self.counted += 1
        self.emit(DefectEvent(camera=self.camera, defect_class=cls, track_id=track_id,
                              frame=crossing_frame, confidence=conf))