"""AdHawk MindLink-197 adapter for canonical scene and gaze samples."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import importlib
import math
from numbers import Real
import queue
import threading
from typing import Any

import cv2
import numpy as np

from foundations.contracts import GazeSample, SceneFrame


@dataclass(frozen=True)
class FrameMetadata:
    """Diagnostic timing information for one emitted scene frame."""

    timestamp: float
    host_receipt_timestamp: float
    vendor_frame_timestamp: datetime | None
    tracker_timestamp: float | None
    dropped_frame_count: int


@dataclass(frozen=True)
class GazeMetadata:
    """Diagnostic information for one emitted gaze observation."""

    timestamp: float
    host_receipt_timestamp: float
    vendor_timestamp: float | None
    gaze_in_image: tuple[float, ...] | None


class _NumericTimestampMapper:
    def __init__(self) -> None:
        self._anchor_vendor: float | None = None
        self._anchor_host: float | None = None
        self._last: float | None = None

    def map(self, vendor_timestamp: float, host_receipt_timestamp: float) -> float:
        vendor = _finite_number("vendor timestamp", vendor_timestamp)
        host = _host_timestamp(host_receipt_timestamp)
        if self._anchor_vendor is None:
            self._anchor_vendor = vendor
            self._anchor_host = host
        assert self._anchor_host is not None
        relative = self._anchor_host + (vendor - self._anchor_vendor)
        if relative < 0:
            raise ValueError("mapped vendor timestamp precedes the experiment origin")
        if self._last is not None and relative < self._last:
            raise ValueError("vendor timestamps moved backwards")
        self._last = relative
        return relative


class _DatetimeTimestampMapper:
    def __init__(self) -> None:
        self._anchor_vendor: datetime | None = None
        self._anchor_host: float | None = None
        self._last: float | None = None

    def map(self, vendor_timestamp: datetime, host_receipt_timestamp: float) -> float:
        if not isinstance(vendor_timestamp, datetime):
            raise TypeError("frame timestamp must be a datetime")
        host = _host_timestamp(host_receipt_timestamp)
        if self._anchor_vendor is None:
            self._anchor_vendor = vendor_timestamp
            self._anchor_host = host
        assert self._anchor_vendor is not None and self._anchor_host is not None
        relative = self._anchor_host + (vendor_timestamp - self._anchor_vendor).total_seconds()
        if relative < 0:
            raise ValueError("mapped frame timestamp precedes the experiment origin")
        if self._last is not None and relative < self._last:
            raise ValueError("frame timestamps moved backwards")
        self._last = relative
        return relative


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _host_timestamp(value: Any) -> float:
    result = _finite_number("host clock timestamp", value)
    if result < 0:
        raise ValueError("host clock timestamp must be non-negative")
    return result


def _normalize_pixel_gaze(
    x: Any, y: Any, *, width: int, height: int
) -> tuple[float, float] | None:
    if width <= 1 or height <= 1:
        raise ValueError("scene dimensions must both exceed one pixel")
    try:
        x_value = _finite_number("gaze x", x)
        y_value = _finite_number("gaze y", y)
    except ValueError:
        return None
    if not (0.0 <= x_value <= width - 1 and 0.0 <= y_value <= height - 1):
        return None
    return x_value / (width - 1), y_value / (height - 1)


class MindLinkAdapter:
    """Translate the live AdHawk SDK into canonical project scene/gaze values."""

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        frame_queue_size: int = 2,
        on_disconnect: Callable[[Any], None] | None = None,
        sdk: Any | None = None,
        api_factory: Callable[[], Any] | None = None,
        video_receiver_factory: Callable[[], Any] | None = None,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        if isinstance(frame_queue_size, bool) or not isinstance(frame_queue_size, int) or frame_queue_size <= 0:
            raise ValueError("frame_queue_size must be a positive integer")
        if on_disconnect is not None and not callable(on_disconnect):
            raise TypeError("on_disconnect must be callable or None")

        if sdk is None:
            try:
                sdk = importlib.import_module("adhawkapi")
                importlib.import_module("adhawkapi.frontend")
            except ImportError as exc:
                raise RuntimeError(
                    "live MindLink mode requires the vendor adhawkapi installation"
                ) from exc
        frontend = sdk.frontend
        self.sdk = sdk
        self.clock = clock
        self.frame_queue_size = frame_queue_size
        self.on_disconnect = on_disconnect
        self.api = (api_factory or frontend.FrontendApi)()
        self._video_receiver_factory = video_receiver_factory or frontend.VideoReceiver

        self._connected = threading.Event()
        self._tracker_ready = threading.Event()
        self._calibration_done = threading.Event()
        self._calibration_result: tuple[Any, ...] | None = None
        self._connection_error: Any = None
        self._disconnect_error: Any = None

        self._capture_lock = threading.Lock()
        self._capture_active = False
        self._frame_stop = threading.Event()
        self._frame_queue: queue.Queue[tuple[Any, Any, Any, float]] = queue.Queue(maxsize=frame_queue_size)
        self._frame_worker: threading.Thread | None = None
        self._video_receiver: Any | None = None
        self._dropped_frame_count = 0
        self._dropped_frame_timestamp_count = 0
        self._dropped_gaze_timestamp_count = 0
        self._scene_shape: tuple[int, int] | None = None
        self._gaze_mapper = _NumericTimestampMapper()
        self._frame_mapper = _DatetimeTimestampMapper()

        self._on_scene_frame: Callable[[SceneFrame], None] | None = None
        self._on_gaze_sample: Callable[[GazeSample], None] | None = None
        self._on_frame_metadata: Callable[[FrameMetadata], None] | None = None
        self._on_gaze_metadata: Callable[[GazeMetadata], None] | None = None

        self.api.register_stream_handler(self.sdk.PacketType.TRACKER_READY, self._tracker_ready_handler)
        self.api.register_stream_handler(self.sdk.PacketType.EYETRACKING_STREAM, self._et_data_handler)

    @property
    def dropped_frame_count(self) -> int:
        return self._dropped_frame_count

    @property
    def disconnect_error(self) -> Any:
        return self._disconnect_error

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def dropped_frame_timestamp_count(self) -> int:
        return self._dropped_frame_timestamp_count

    @property
    def dropped_gaze_timestamp_count(self) -> int:
        return self._dropped_gaze_timestamp_count

    @property
    def is_capturing(self) -> bool:
        return self._capture_active

    def _on_connect(self, error: Any) -> None:
        if error:
            self._connection_error = error
        else:
            self._connected.set()

    def _on_disconnect(self, error: Any) -> None:
        self._disconnect_error = error
        self._connected.clear()
        self._capture_active = False
        self._frame_stop.set()
        if self.on_disconnect is not None:
            self.on_disconnect(error)

    def _tracker_ready_handler(self, *args: Any) -> None:
        del args
        self._tracker_ready.set()

    def connect(
        self, *, connect_timeout_seconds: float = 5.0, tracker_ready_timeout_seconds: float = 15.0
    ) -> None:
        connect_timeout = _positive_seconds("connect_timeout_seconds", connect_timeout_seconds)
        ready_timeout = _positive_seconds("tracker_ready_timeout_seconds", tracker_ready_timeout_seconds)
        self.api.start(connect_cb=self._on_connect, disconnect_cb=self._on_disconnect)
        if not self._connected.wait(connect_timeout):
            detail = f": {self._connection_error}" if self._connection_error is not None else ""
            raise RuntimeError(f"failed to connect to AdHawk Backend Service{detail}")
        self.api.enable_tracking(True)
        if not self._tracker_ready.wait(ready_timeout):
            raise RuntimeError("MindLink tracker did not become ready")

    def calibrate(
        self,
        *,
        marker_size_mm: int = 35,
        returning_user: bool = False,
        timeout_seconds: float = 120.0,
    ) -> tuple[Any, ...]:
        if isinstance(marker_size_mm, bool) or not isinstance(marker_size_mm, int) or marker_size_mm <= 0:
            raise ValueError("marker_size_mm must be a positive integer")
        if not isinstance(returning_user, bool):
            raise TypeError("returning_user must be a bool")
        timeout = _positive_seconds("timeout_seconds", timeout_seconds)
        self._calibration_done.clear()
        self._calibration_result = None

        def callback(*response: Any) -> None:
            self._calibration_result = tuple(response)
            self._calibration_done.set()

        self.api.quick_start_gui(
            mode=self.sdk.MarkerSequenceMode.FIXED_GAZE,
            marker_size_mm=marker_size_mm,
            returning_user=returning_user,
            callback=callback,
        )
        if not self._calibration_done.wait(timeout):
            raise RuntimeError("MindLink calibration timed out")
        assert self._calibration_result is not None
        return self._calibration_result

    def start_capture(
        self,
        *,
        on_scene_frame: Callable[[SceneFrame], None],
        on_gaze_sample: Callable[[GazeSample], None],
        on_frame_metadata: Callable[[FrameMetadata], None] | None = None,
        on_gaze_metadata: Callable[[GazeMetadata], None] | None = None,
    ) -> None:
        for name, callback in (("on_scene_frame", on_scene_frame), ("on_gaze_sample", on_gaze_sample)):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        for name, callback in (("on_frame_metadata", on_frame_metadata), ("on_gaze_metadata", on_gaze_metadata)):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None")

        with self._capture_lock:
            if self._capture_active:
                raise RuntimeError("MindLink capture is already active")
            self._on_scene_frame = on_scene_frame
            self._on_gaze_sample = on_gaze_sample
            self._on_frame_metadata = on_frame_metadata
            self._on_gaze_metadata = on_gaze_metadata
            self._scene_shape = None
            self._gaze_mapper = _NumericTimestampMapper()
            self._frame_mapper = _DatetimeTimestampMapper()
            self._frame_stop.clear()
            self._drain_frame_queue()

            receiver = self._video_receiver_factory()
            receiver.frame_received_event.add_callback(self._frame_handler)
            receiver.start()
            self._video_receiver = receiver

            streams = (
                self.sdk.EyeTrackingStreamTypes.GAZE,
                self.sdk.EyeTrackingStreamTypes.GAZE_IN_IMAGE,
            )
            try:
                for stream in streams:
                    self.api.set_et_stream_control(stream, True)
                self._capture_active = True
                self._frame_worker = threading.Thread(
                    target=self._frame_worker_loop,
                    name="mindlink_frames",
                    daemon=True,
                )
                self._frame_worker.start()
                self.api.start_video_stream(*receiver.address)
            except Exception:
                self._capture_active = False
                self._frame_stop.set()
                for stream in streams:
                    try:
                        self.api.set_et_stream_control(stream, False)
                    except Exception:
                        pass
                try:
                    receiver.shutdown()
                except Exception:
                    pass
                self._video_receiver = None
                raise

    def _frame_handler(
        self, tracker_timestamp: Any, frame_image_data: Any, frame_timestamp: Any
    ) -> None:
        if not self._capture_active:
            return
        host_receipt = _host_timestamp(self.clock())
        item = (tracker_timestamp, frame_image_data, frame_timestamp, host_receipt)
        try:
            self._frame_queue.put_nowait(item)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._dropped_frame_count += 1
            except queue.Empty:
                pass
            self._frame_queue.put_nowait(item)

    def _frame_worker_loop(self) -> None:
        while not self._frame_stop.is_set() or not self._frame_queue.empty():
            try:
                tracker_timestamp, frame_image_data, frame_timestamp, host_receipt = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            encoded = np.frombuffer(frame_image_data, dtype=np.uint8)
            image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            self._scene_shape = (height, width)

            vendor_frame_timestamp = frame_timestamp if isinstance(frame_timestamp, datetime) else None
            if vendor_frame_timestamp is None:
                timestamp = host_receipt
            else:
                try:
                    timestamp = self._frame_mapper.map(vendor_frame_timestamp, host_receipt)
                except (TypeError, ValueError):
                    self._dropped_frame_timestamp_count += 1
                    continue
            scene = SceneFrame(timestamp=timestamp, image=image_rgb)
            if self._on_scene_frame is not None:
                self._on_scene_frame(scene)
            if self._on_frame_metadata is not None:
                tracker_value: float | None
                try:
                    candidate = _finite_number("tracker timestamp", tracker_timestamp)
                    tracker_value = None if candidate == 0.0 else candidate
                except ValueError:
                    tracker_value = None
                self._on_frame_metadata(
                    FrameMetadata(
                        timestamp=timestamp,
                        host_receipt_timestamp=host_receipt,
                        vendor_frame_timestamp=vendor_frame_timestamp,
                        tracker_timestamp=tracker_value,
                        dropped_frame_count=self._dropped_frame_count,
                    )
                )

    def _et_data_handler(self, et_data: Any) -> None:
        if not self._capture_active or self._on_gaze_sample is None:
            return
        host_receipt = _host_timestamp(self.clock())
        vendor_raw = getattr(et_data, "timestamp", None)
        try:
            vendor_timestamp = _finite_number("gaze vendor timestamp", vendor_raw)
            timestamp = self._gaze_mapper.map(vendor_timestamp, host_receipt)
        except ValueError:
            self._dropped_gaze_timestamp_count += 1
            return

        raw_gaze = getattr(et_data, "gaze_in_image", None)
        gaze_tuple: tuple[float, ...] | None = None
        if raw_gaze is not None:
            try:
                gaze_tuple = tuple(float(value) for value in raw_gaze)
            except (TypeError, ValueError):
                gaze_tuple = None

        normalized: tuple[float, float] | None = None
        if gaze_tuple is not None and len(gaze_tuple) >= 2 and self._scene_shape is not None:
            height, width = self._scene_shape
            normalized = _normalize_pixel_gaze(
                gaze_tuple[0], gaze_tuple[1], width=width, height=height
            )
        if normalized is None:
            sample = GazeSample(timestamp, None, None, False, None)
        else:
            sample = GazeSample(timestamp, normalized[0], normalized[1], True, None)
        self._on_gaze_sample(sample)
        if self._on_gaze_metadata is not None:
            self._on_gaze_metadata(
                GazeMetadata(
                    timestamp=timestamp,
                    host_receipt_timestamp=host_receipt,
                    vendor_timestamp=vendor_timestamp,
                    gaze_in_image=gaze_tuple,
                )
            )

    def stop_capture(self) -> None:
        with self._capture_lock:
            receiver = self._video_receiver
            streams = (
                self.sdk.EyeTrackingStreamTypes.GAZE,
                self.sdk.EyeTrackingStreamTypes.GAZE_IN_IMAGE,
            )
            self._capture_active = False
            self._frame_stop.set()
            if receiver is not None:
                try:
                    self.api.stop_video_stream(*receiver.address)
                except Exception:
                    pass
            for stream in streams:
                try:
                    self.api.set_et_stream_control(stream, False)
                except Exception:
                    pass
            if receiver is not None:
                try:
                    receiver.shutdown()
                except Exception:
                    pass
            self._video_receiver = None
        worker = self._frame_worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)
        self._frame_worker = None
        self._drain_frame_queue()

    def close(self) -> None:
        self.stop_capture()
        try:
            self.api.shutdown()
        finally:
            self._connected.clear()
            self._tracker_ready.clear()

    def _drain_frame_queue(self) -> None:
        while True:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                return


def _positive_seconds(name: str, value: Any) -> float:
    result = _finite_number(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


if __name__ == "__main__":
    print(_normalize_pixel_gaze(639, 359, width=1280, height=720))
