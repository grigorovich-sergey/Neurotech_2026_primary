"""Live MindLink input ordering and contextual feedback helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
import math
import queue
import time
from typing import Any, Callable

from experiment_learning.contracts import FeedbackResolution
from experiment_learning.state_machine import ExperimentController
from foundations.contracts import GazeSample, SceneFrame
from foundations.events import Event, JsonlEventLogger


@dataclass(order=True)
class _BufferedInput:
    timestamp: float
    stream_priority: int
    sequence: int
    received_monotonic: float = field(compare=False)
    stream: str = field(compare=False)
    sample: SceneFrame | GazeSample = field(compare=False)


class LiveInputMerger:
    """Merge asynchronous scene/gaze callbacks into bounded timestamp order."""

    def __init__(
        self,
        *,
        scene_queue_size: int,
        gaze_queue_size: int,
        reorder_hold_seconds: float,
        event_logger: JsonlEventLogger,
        failure_callback: Callable[[str, BaseException], None],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in (
            ("scene_queue_size", scene_queue_size),
            ("gaze_queue_size", gaze_queue_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(reorder_hold_seconds, bool)
            or not isinstance(reorder_hold_seconds, (int, float))
            or not math.isfinite(float(reorder_hold_seconds))
            or reorder_hold_seconds < 0
        ):
            raise ValueError("reorder_hold_seconds must be finite and non-negative")
        if not callable(failure_callback) or not callable(monotonic):
            raise TypeError("failure_callback and monotonic must be callable")
        self.scene_queue: queue.Queue[tuple[SceneFrame, float]] = queue.Queue(
            maxsize=scene_queue_size
        )
        self.scene_queue_size = scene_queue_size
        self.gaze_queue: queue.Queue[tuple[GazeSample, float]] = queue.Queue(
            maxsize=gaze_queue_size
        )
        self.reorder_hold_seconds = float(reorder_hold_seconds)
        self.event_logger = event_logger
        self.failure_callback = failure_callback
        self.monotonic = monotonic
        self._heap: list[_BufferedInput] = []
        self._sequence = 0
        self._latest_seen: dict[str, float | None] = {"scene": None, "gaze": None}
        self.scene_queue_drop_count = 0

    def on_scene(self, sample: SceneFrame) -> None:
        if not isinstance(sample, SceneFrame):
            self.failure_callback("mindlink_scene_type", TypeError("expected SceneFrame"))
            return
        item = (sample, self.monotonic())
        try:
            self.scene_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                dropped, _ = self.scene_queue.get_nowait()
            except queue.Empty:  # pragma: no cover - defensive queue race
                dropped = sample
            self._record_scene_drop(dropped)
            self.scene_queue.put_nowait(item)

    def _record_scene_drop(self, sample: SceneFrame) -> None:
        self.scene_queue_drop_count += 1
        self.event_logger.log(
            Event(
                float(sample.timestamp),
                "experiment_live_scene_queue_drop",
                {"count": self.scene_queue_drop_count},
            )
        )

    def _push_buffered(self, item: _BufferedInput) -> None:
        if item.stream == "scene":
            pending_scenes = [
                (index, pending)
                for index, pending in enumerate(self._heap)
                if pending.stream == "scene"
            ]
            if len(pending_scenes) >= self.scene_queue_size:
                dropped_index, dropped = min(
                    pending_scenes,
                    key=lambda value: (
                        value[1].timestamp,
                        value[1].sequence,
                    ),
                )
                del self._heap[dropped_index]
                heapq.heapify(self._heap)
                assert isinstance(dropped.sample, SceneFrame)
                self._record_scene_drop(dropped.sample)
        heapq.heappush(self._heap, item)

    def on_gaze(self, sample: GazeSample) -> None:
        if not isinstance(sample, GazeSample):
            self.failure_callback("mindlink_gaze_type", TypeError("expected GazeSample"))
            return
        try:
            self.gaze_queue.put_nowait((sample, self.monotonic()))
        except queue.Full:
            self.failure_callback(
                "mindlink_gaze_queue_overflow",
                RuntimeError("live gaze queue overflowed; no gaze sample was dropped"),
            )

    def pop_ready(
        self, *, force: bool = False, cutoff_timestamp: float | None = None
    ) -> tuple[str, SceneFrame | GazeSample] | None:
        self._drain_callbacks()
        if not self._heap:
            return None
        first = self._heap[0]
        if cutoff_timestamp is not None and first.timestamp > cutoff_timestamp + 1e-12:
            return None
        if not force and not self._ready(first):
            return None
        item = heapq.heappop(self._heap)
        return item.stream, item.sample

    def has_pending_through(self, timestamp: float) -> bool:
        self._drain_callbacks()
        return bool(self._heap and self._heap[0].timestamp <= timestamp + 1e-12)

    def has_pending(self) -> bool:
        self._drain_callbacks()
        return bool(self._heap)

    def discard_after(self, timestamp: float) -> int:
        self._drain_callbacks()
        kept: list[_BufferedInput] = []
        discarded = 0
        while self._heap:
            item = heapq.heappop(self._heap)
            if item.timestamp > timestamp + 1e-12:
                discarded += 1
            else:
                kept.append(item)
        self._heap = kept
        heapq.heapify(self._heap)
        return discarded

    def _drain_callbacks(self) -> None:
        for stream, source, priority in (
            ("scene", self.scene_queue, 0),
            ("gaze", self.gaze_queue, 1),
        ):
            while True:
                try:
                    sample, received = source.get_nowait()
                except queue.Empty:
                    break
                timestamp = float(sample.timestamp)
                latest = self._latest_seen[stream]
                if latest is not None and timestamp < latest - 1e-12:
                    error = ValueError(f"{stream} callback timestamps moved backwards")
                    if stream == "gaze":
                        self.failure_callback("mindlink_late_gaze", error)
                    else:
                        self.event_logger.log(
                            Event(
                                timestamp,
                                "experiment_live_late_scene_drop",
                                {"latest_scene_timestamp": latest},
                            )
                        )
                    continue
                self._latest_seen[stream] = timestamp
                self._sequence += 1
                self._push_buffered(
                    _BufferedInput(
                        timestamp,
                        priority,
                        self._sequence,
                        received,
                        stream,
                        sample,
                    ),
                )

    def _ready(self, item: _BufferedInput) -> bool:
        if self.monotonic() - item.received_monotonic >= self.reorder_hold_seconds:
            return True
        scene_high = self._latest_seen["scene"]
        gaze_high = self._latest_seen["gaze"]
        return (
            scene_high is not None
            and gaze_high is not None
            and scene_high >= item.timestamp
            and gaze_high >= item.timestamp
        )


class ContextualFeedbackDriver:
    """Resolve operator button presses against whichever feedback target is open."""

    def __init__(self, *, event_logger: JsonlEventLogger, session_id: str) -> None:
        self.event_logger = event_logger
        self.session_id = session_id
        self._presses: deque[float] = deque()

    def submit_press(self, timestamp: float) -> None:
        value = float(timestamp)
        if not math.isfinite(value) or value < 0:
            raise ValueError("feedback press timestamp must be finite and non-negative")
        if self._presses and value < self._presses[-1]:
            raise ValueError("feedback press timestamps must be non-decreasing")
        self._presses.append(value)

    def before_time(
        self, timestamp: float, controller: ExperimentController
    ) -> tuple[FeedbackResolution, ...]:
        value = float(timestamp)
        resolutions: list[FeedbackResolution] = []
        while self._presses and self._presses[0] <= value + 1e-12:
            press = self._presses.popleft()
            episode_id = controller.pending_feedback_episode_id
            result = controller.accept_feedback(press)
            if result is None:
                self.event_logger.log(
                    Event(
                        press,
                        "integration_feedback_press_ignored",
                        {"session_id": self.session_id, "reason": "no_open_window"},
                    )
                )
            else:
                resolutions.append(result)
                self.event_logger.log(
                    Event(
                        press,
                        "integration_feedback_press",
                        {"session_id": self.session_id, "episode_id": episode_id},
                    )
                )
        timeout = controller.advance_time(value)
        if timeout is not None:
            resolutions.append(timeout)
        return tuple(resolutions)

    def feedback_opened(
        self,
        *,
        episode_id: int,
        outcome_timestamp: float,
        controller: ExperimentController,
    ) -> None:
        del episode_id, outcome_timestamp, controller


if __name__ == "__main__":
    print("LiveInputMerger orders MindLink callbacks by scientific timestamp.")
