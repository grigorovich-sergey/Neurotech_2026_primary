#!/usr/bin/env python3
"""Troubleshoot IDUN Guardian recording with timestamp-safe raw EEG ordering.

This current playground version deliberately owns its Guardian client lifecycle
instead of importing the production Guardian adapter. It is a hardware
troubleshooting playground: raw 20-sample SDK blocks are buffered for 30
seconds, filtered locally, and displayed as theta, alpha, and beta traces.
Press Esc to stop cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import sys
import time
from typing import Any

# Resolve project modules directly from the checkout. No editable install or
# caller-provided PYTHONPATH is needed for project imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

from eeg_pipeline.buffer import EEGBuffer
from eeg_pipeline.contracts import EEGSample
from eeg_pipeline.credentials import load_guardian_api_token
from eeg_pipeline.processing import (
    EEGFeatureExtractor,
    EEGPreprocessor,
    FEATURE_NAMES,
)


TOKEN_FILE = Path(".secrets/idun_api_token")
TOKEN_ENVIRONMENT_VARIABLE = "IDUN_API_TOKEN"
DEFAULT_RECORDING_SECONDS = 180

SAMPLE_RATE_HZ = 250.0
EXPECTED_RAW_BLOCK_SAMPLES = 20
# IDUN websocket callbacks can arrive more than a second out of order, while
# their sequence field is not guaranteed to be contiguous for raw EEG. Hold a
# Guardian-timestamp horizon, then emit the chronologically oldest block.
REORDER_HOLDBACK_SECONDS = 3.0
WINDOW_SECONDS = 30.0
PREFILTER_LOW_HZ = 1.0
PREFILTER_HIGH_HZ = 40.0
FILTER_ORDER = 4
MINIMUM_DISPLAY_SAMPLES = int(SAMPLE_RATE_HZ)

WINDOW_NAME = "IDUN Guardian troubleshooting monitor - Esc to stop"
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 820
BUILD_ID = "guardian-playground-v5-2026-08-14"


@dataclass(frozen=True, slots=True)
class GuardianPreflight:
    """Measurements gathered before recording, if the selected mode requests them."""

    battery_percent: float | None
    impedance_ohms: float | None


class GuardianQueueOverflowError(RuntimeError):
    """Raised instead of silently dropping raw EEG from the playground queue."""


class GuardianTimestampMapper:
    """Map Guardian Unix timestamps onto the playground run-relative clock."""

    def __init__(
        self, *, anchor_run_timestamp: float, anchor_vendor_timestamp: float
    ) -> None:
        self.anchor_run_timestamp = _non_negative_finite(
            "anchor_run_timestamp", anchor_run_timestamp
        )
        self.anchor_vendor_timestamp = _non_negative_finite(
            "anchor_vendor_timestamp", anchor_vendor_timestamp
        )
        self._last_relative: float | None = None
        self._last_vendor: float | None = None

    @property
    def last_vendor_timestamp(self) -> float | None:
        return self._last_vendor

    def map(self, vendor_timestamp: float) -> float:
        vendor = _non_negative_finite("Guardian timestamp", vendor_timestamp)
        relative = self.anchor_run_timestamp + (
            vendor - self.anchor_vendor_timestamp
        )
        if relative < 0.0:
            raise ValueError("mapped Guardian timestamp precedes the run origin")
        if self._last_vendor is not None and vendor < self._last_vendor:
            raise ValueError(
                "Guardian timestamps moved backwards: "
                f"current={vendor:.6f}, previous={self._last_vendor:.6f}, "
                f"delta={vendor - self._last_vendor:.6f}s"
            )
        if self._last_relative is not None and relative < self._last_relative:
            raise ValueError("mapped Guardian timestamps moved backwards")
        self._last_vendor = vendor
        self._last_relative = relative
        return relative


@dataclass(frozen=True, slots=True)
class _GuardianRawSample:
    vendor_timestamp: float
    value_uv: float


@dataclass(frozen=True, slots=True)
class _GuardianRawBlock:
    sequence: int
    samples: tuple[_GuardianRawSample, ...]
    host_receipt_timestamp: float


class GuardianAdapter:
    """Async troubleshooting adapter that follows the manufacturer's loop model.

    Unlike the production adapter, this class creates no worker thread and no
    second event loop. ``start_recording`` runs as a task on the main asyncio
    loop, matching the working manufacturer example while OpenCV is polled by
    short non-blocking turns on that same thread.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        address: str | None,
        api_token: str,
        debug: bool,
        timestamp_mode: str = "first-block",
        queue_capacity_samples: int = 15_000,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        if address is not None and (not isinstance(address, str) or not address):
            raise ValueError("Guardian address must be a non-empty string or None")
        if not isinstance(api_token, str) or not api_token:
            raise ValueError("Guardian api_token must be a non-empty string")
        if not isinstance(debug, bool):
            raise TypeError("debug must be a bool")
        if timestamp_mode not in {"first-block", "production"}:
            raise ValueError("timestamp_mode must be 'first-block' or 'production'")
        if (
            isinstance(queue_capacity_samples, bool)
            or not isinstance(queue_capacity_samples, int)
            or queue_capacity_samples <= 0
        ):
            raise ValueError("queue_capacity_samples must be a positive integer")
        if client_factory is None:
            try:
                module = importlib.import_module("idun_guardian_sdk")
            except ImportError as exc:
                raise RuntimeError("idun-guardian-sdk is not installed") from exc
            client_factory = module.GuardianClient

        self.clock = clock
        self.timestamp_mode = timestamp_mode
        self.queue_capacity_samples = queue_capacity_samples
        self.client = client_factory(
            address=address,
            api_token=api_token,
            debug=debug,
        )
        self._samples: deque[EEGSample] = deque()
        self._failure: BaseException | None = None
        self._recording_task: asyncio.Task[str | None] | None = None
        self._recording_id: str | None = None
        self._mapper: GuardianTimestampMapper | None = None
        self._pending_raw_blocks: dict[int, _GuardianRawBlock] = {}
        self._seen_raw_sequences: set[int] = set()
        self._last_received_sequence: int | None = None
        self._reorder_window_announced = False
        self._reorder_finalized = False
        self._prepared = False
        self._started = False
        self._closed = False
        self._recording_started_at: float | None = None

        self.sdk_event_count = 0
        self.raw_block_received_count = 0
        self.raw_block_count = 0
        self.last_raw_block_size = 0
        self.unexpected_raw_block_count = 0
        self.out_of_order_raw_block_count = 0
        self.duplicate_raw_block_count = 0
        self.late_raw_block_count = 0
        self.sequence_discontinuity_count = 0
        self.last_sequence: int | None = None
        self.last_emitted_sequence: int | None = None
        self.last_action: str | None = None

    @property
    def recording_active(self) -> bool:
        task = self._recording_task
        return task is not None and not task.done()

    @property
    def recording_done(self) -> bool:
        task = self._recording_task
        return task is not None and task.done()

    @property
    def recording_id(self) -> str | None:
        self._capture_recording_id()
        return self._recording_id

    @property
    def recording_task_state(self) -> str:
        task = self._recording_task
        if task is None:
            return "not-created"
        if task.cancelled():
            return "cancelled"
        if task.done():
            return "done"
        return "running"

    @property
    def recording_elapsed_seconds(self) -> float:
        if self._recording_started_at is None:
            return 0.0
        return max(0.0, float(self.clock()) - self._recording_started_at)

    @property
    def pending_raw_block_count(self) -> int:
        return len(self._pending_raw_blocks)

    def _set_failure(self, exc: BaseException) -> None:
        if self._failure is None:
            self._failure = exc

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _capture_recording_id(self, candidate: Any = None) -> None:
        if isinstance(candidate, str) and candidate:
            self._recording_id = candidate
            return
        getter = getattr(self.client, "get_recording_id", None)
        if not callable(getter):
            return
        try:
            value = getter()
        except (AttributeError, RuntimeError, ValueError):
            return
        if isinstance(value, str) and value:
            self._recording_id = value

    def _enqueue(self, sample: EEGSample) -> None:
        if len(self._samples) >= self.queue_capacity_samples:
            error = GuardianQueueOverflowError(
                "Guardian playground queue overflowed; no samples were dropped silently"
            )
            self._set_failure(error)
            task = self._recording_task
            if task is not None and not task.done():
                task.cancel()
            raise error
        self._samples.append(sample)

    def _initialise_mapper(self, raw_block: _GuardianRawBlock) -> None:
        first_vendor = raw_block.samples[0].vendor_timestamp
        last_vendor = raw_block.samples[-1].vendor_timestamp
        block_duration = last_vendor - first_vendor
        if block_duration < 0.0:
            raise ValueError("Guardian raw EEG block timestamps moved backwards")
        # Align the last sample of the first received block to host receipt. This
        # avoids clock-skew-induced negative run timestamps while preserving the
        # vendor's 250 Hz spacing inside and across blocks.
        first_run_timestamp = max(
            0.0, raw_block.host_receipt_timestamp - block_duration
        )
        self._mapper = GuardianTimestampMapper(
            anchor_run_timestamp=first_run_timestamp,
            anchor_vendor_timestamp=first_vendor,
        )

    def _emit_raw_block(self, raw_block: _GuardianRawBlock) -> None:
        previous_vendor: float | None = None
        for index, raw_sample in enumerate(raw_block.samples):
            if (
                previous_vendor is not None
                and raw_sample.vendor_timestamp < previous_vendor
            ):
                raise ValueError(
                    f"sequence {raw_block.sequence} has backwards timestamps at "
                    f"sample {index}: current={raw_sample.vendor_timestamp:.6f}, "
                    f"previous={previous_vendor:.6f}"
                )
            previous_vendor = raw_sample.vendor_timestamp

        if self._mapper is None:
            self._initialise_mapper(raw_block)
        assert self._mapper is not None
        for raw_sample in raw_block.samples:
            try:
                timestamp = self._mapper.map(raw_sample.vendor_timestamp)
            except ValueError as exc:
                raise ValueError(
                    f"sequence {raw_block.sequence}: {exc}"
                ) from exc
            self._enqueue(
                EEGSample(
                    timestamp=timestamp,
                    value_uv=raw_sample.value_uv,
                    valid=True,
                    vendor_timestamp_unix=raw_sample.vendor_timestamp,
                    host_receipt_timestamp=raw_block.host_receipt_timestamp,
                )
            )

        if (
            self.last_emitted_sequence is not None
            and raw_block.sequence != self.last_emitted_sequence + 1
        ):
            self.sequence_discontinuity_count += 1
            print(
                "[stream diagnostic] timestamp-ordered raw sequence changed "
                f"{self.last_emitted_sequence} -> {raw_block.sequence}"
            )
        self.raw_block_count += 1
        self.last_emitted_sequence = raw_block.sequence
        if self.raw_block_count <= 3 or self.raw_block_count % 50 == 0:
            print(
                f"[stream] timestamp-ordered raw block #{self.raw_block_count}: "
                f"{len(raw_block.samples)} samples, sequence={raw_block.sequence}"
            )

    @staticmethod
    def _raw_block_order_key(raw_block: _GuardianRawBlock) -> tuple[float, int]:
        return raw_block.samples[0].vendor_timestamp, raw_block.sequence

    def _emit_oldest_pending_raw_block(self) -> None:
        sequence, raw_block = min(
            self._pending_raw_blocks.items(),
            key=lambda item: self._raw_block_order_key(item[1]),
        )
        del self._pending_raw_blocks[sequence]
        self._emit_raw_block(raw_block)

    def _flush_ready_raw_blocks(self) -> None:
        while self._pending_raw_blocks:
            oldest = min(
                self._pending_raw_blocks.values(),
                key=self._raw_block_order_key,
            )
            newest_vendor_timestamp = max(
                block.samples[-1].vendor_timestamp
                for block in self._pending_raw_blocks.values()
            )
            if (
                newest_vendor_timestamp - oldest.samples[-1].vendor_timestamp
                < REORDER_HOLDBACK_SECONDS
            ):
                return
            self._emit_oldest_pending_raw_block()

    def _queue_raw_block(self, raw_block: _GuardianRawBlock) -> None:
        """Buffer callback jitter and emit in Guardian timestamp order."""

        sequence = raw_block.sequence
        if sequence in self._seen_raw_sequences:
            self.duplicate_raw_block_count += 1
            print(
                f"[stream warning] duplicate raw sequence {sequence}; "
                "duplicate ignored"
            )
            return

        last_emitted_vendor_timestamp = (
            None if self._mapper is None else self._mapper.last_vendor_timestamp
        )
        if (
            last_emitted_vendor_timestamp is not None
            and raw_block.samples[0].vendor_timestamp
            <= last_emitted_vendor_timestamp
        ):
            self._seen_raw_sequences.add(sequence)
            self.late_raw_block_count += 1
            print(
                "[stream warning] raw block arrived beyond the timestamp "
                f"holdback and was not reinserted: sequence={sequence}, "
                f"block_start={raw_block.samples[0].vendor_timestamp:.6f}, "
                f"last_emitted={last_emitted_vendor_timestamp:.6f}"
            )
            return

        if (
            self._last_received_sequence is not None
            and sequence < self._last_received_sequence
        ):
            self.out_of_order_raw_block_count += 1
            print(
                f"[stream reorder] received sequence {sequence} after "
                f"{self._last_received_sequence}; buffering by Guardian timestamp"
            )
        self._last_received_sequence = sequence
        self._seen_raw_sequences.add(sequence)
        self._pending_raw_blocks[sequence] = raw_block
        if (
            not self._reorder_window_announced
            and self.raw_block_count > 0
        ):
            self._reorder_window_announced = True
            print(
                "[stream reorder] timestamp jitter buffer primed; retaining a "
                f"{REORDER_HOLDBACK_SECONDS:.1f}s Guardian-timestamp holdback"
            )
        self._flush_ready_raw_blocks()

    def _finalize_raw_block_order(self) -> None:
        if self._reorder_finalized:
            return
        while self._pending_raw_blocks:
            self._emit_oldest_pending_raw_block()
        self._reorder_finalized = True

    def _handle_raw_event(self, event: Any) -> None:
        """SDK callback: validate and queue one documented 20-sample block."""

        try:
            message = getattr(event, "message", event)
            if not isinstance(message, Mapping):
                raise ValueError("Guardian event message must be a mapping")
            self.sdk_event_count += 1
            action = message.get("action")
            sequence = message.get("sequence")
            self.last_action = action if isinstance(action, str) else None
            if isinstance(sequence, bool) or not isinstance(sequence, int):
                raise ValueError(
                    "Guardian raw event requires an integer sequence for safe ordering"
                )
            self.last_sequence = sequence

            raw_block = message.get("raw_eeg")
            if not isinstance(raw_block, list) or not raw_block:
                raise ValueError("Guardian live event is missing a raw_eeg block")
            self.raw_block_received_count += 1
            self.last_raw_block_size = len(raw_block)
            if len(raw_block) != EXPECTED_RAW_BLOCK_SAMPLES:
                self.unexpected_raw_block_count += 1
                print(
                    "[stream warning] raw block "
                    f"#{self.raw_block_received_count} contains "
                    f"{len(raw_block)} samples; "
                    f"expected {EXPECTED_RAW_BLOCK_SAMPLES}"
                )

            host_receipt_timestamp = _non_negative_finite(
                "host receipt timestamp", self.clock()
            )
            normalized_samples: list[_GuardianRawSample] = []
            for raw_sample in raw_block:
                if not isinstance(raw_sample, Mapping):
                    raise ValueError("Guardian raw EEG samples must be mappings")
                if "timestamp" not in raw_sample or "ch1" not in raw_sample:
                    raise ValueError(
                        "Guardian raw EEG sample requires timestamp and ch1"
                    )
                normalized_samples.append(
                    _GuardianRawSample(
                        vendor_timestamp=_non_negative_finite(
                            "Guardian timestamp", raw_sample["timestamp"]
                        ),
                        value_uv=_finite_number(
                            "Guardian ch1 sample", raw_sample["ch1"]
                        ),
                    )
                )
            self._queue_raw_block(
                _GuardianRawBlock(
                    sequence=sequence,
                    samples=tuple(normalized_samples),
                    host_receipt_timestamp=host_receipt_timestamp,
                )
            )
        except BaseException as exc:
            error = RuntimeError(
                f"raw Guardian callback failed: {type(exc).__name__}: {exc}"
            )
            self._set_failure(error)
            print(f"[stream error] {error}", file=sys.stderr)
            task = self._recording_task
            if task is not None and not task.done():
                task.cancel()

    async def _measure_impedance(
        self,
        *,
        duration_seconds: float,
        max_impedance_ohms: float,
        mains_frequency_hz: int,
    ) -> float:
        readings: list[float] = []

        def handle_impedance(value: Any) -> None:
            readings.append(_finite_number("Guardian impedance", value))

        print("[preflight] starting impedance stream")
        task = asyncio.create_task(
            self.client.stream_impedance(
                mains_freq_60hz=mains_frequency_hz == 60,
                handler=handle_impedance,
            ),
            name="guardian-playground-impedance",
        )
        try:
            await asyncio.sleep(duration_seconds)
        finally:
            print("[preflight] requesting impedance stop")
            self.client.stop_impedance()
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise RuntimeError(
                    "Guardian impedance task did not stop within 10 seconds"
                )
        if not readings:
            raise RuntimeError("Guardian impedance preflight returned no readings")
        impedance = readings[-1]
        print(f"[preflight] final impedance: {impedance:.0f} ohm")
        if impedance >= max_impedance_ohms:
            raise RuntimeError(
                f"Guardian impedance {impedance:.0f} ohm is not below "
                f"{max_impedance_ohms:.0f} ohm"
            )
        return impedance

    async def prepare(
        self,
        *,
        mode: str,
        impedance_seconds: float,
        max_impedance_ohms: float,
        mains_frequency_hz: int,
    ) -> GuardianPreflight:
        """Run a selectable pre-start path to isolate lifecycle incompatibilities."""

        if self._closed:
            raise RuntimeError("Guardian adapter is closed")
        if self._prepared:
            raise RuntimeError("Guardian prepare() has already been called")
        if mode not in {"manufacturer", "connect", "battery", "impedance"}:
            raise ValueError("unknown Guardian preflight mode")

        if mode == "manufacturer":
            print(
                "[preflight] manufacturer mode: explicit BLE connect, battery, "
                "and impedance are skipped"
            )
            result = GuardianPreflight(None, None)
            self._prepared = True
            return result

        try:
            print("[preflight] awaiting client.connect_device()")
            await self.client.connect_device()
            print("[preflight] BLE connect completed")
            battery: float | None = None
            if mode in {"battery", "impedance"}:
                print("[preflight] awaiting client.check_battery()")
                battery = _battery_percent(await self.client.check_battery())
                print(f"[preflight] battery: {battery:.0f}%")
            impedance: float | None = None
            if mode == "impedance":
                impedance = await self._measure_impedance(
                    duration_seconds=impedance_seconds,
                    max_impedance_ohms=max_impedance_ohms,
                    mains_frequency_hz=mains_frequency_hz,
                )
            result = GuardianPreflight(battery, impedance)
            self._prepared = True
            return result
        except BaseException:
            try:
                await self.client.disconnect_device()
            except BaseException:
                pass
            raise

    async def _record(self, recording_seconds: int) -> str | None:
        print(
            "[recording] entering client.start_recording("
            f"recording_timer={recording_seconds}, led_sleep=False, "
            "calc_latency=False)"
        )
        return await self.client.start_recording(
            recording_timer=recording_seconds,
            led_sleep=False,
            calc_latency=False,
        )

    def _recording_finished(self, task: asyncio.Task[str | None]) -> None:
        if task.cancelled():
            print("[recording] start_recording task cancelled")
            return
        try:
            result = task.result()
        except BaseException as exc:
            self._set_failure(exc)
            print(
                f"[recording error] {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return
        self._capture_recording_id(result)
        try:
            self._finalize_raw_block_order()
        except BaseException as exc:
            self._set_failure(exc)
            print(f"[stream error] {type(exc).__name__}: {exc}", file=sys.stderr)
            return
        print(
            "[recording] start_recording returned normally"
            + (
                ""
                if self._recording_id is None
                else f"; recording_id={self._recording_id}"
            )
        )

    async def start(self, *, recording_seconds: int) -> None:
        """Subscribe to raw EEG, then launch the SDK coroutine on this event loop."""

        if self._closed:
            raise RuntimeError("Guardian adapter is closed")
        if not self._prepared:
            raise RuntimeError("Guardian prepare() must complete before start()")
        if self._started:
            raise RuntimeError("Guardian recording has already been started")
        if recording_seconds <= 0:
            raise ValueError("recording_seconds must be positive")

        self._samples.clear()
        self._failure = None
        self._mapper = None
        self._pending_raw_blocks.clear()
        self._seen_raw_sequences.clear()
        self._last_received_sequence = None
        self._reorder_window_announced = False
        self._reorder_finalized = False
        self.sdk_event_count = 0
        self.raw_block_received_count = 0
        self.raw_block_count = 0
        self.last_raw_block_size = 0
        self.unexpected_raw_block_count = 0
        self.out_of_order_raw_block_count = 0
        self.duplicate_raw_block_count = 0
        self.late_raw_block_count = 0
        self.sequence_discontinuity_count = 0
        self.last_sequence = None
        self.last_emitted_sequence = None
        self.last_action = None
        print("[recording] subscribing to entitlement-safe raw EEG only")
        self.client.subscribe_live_insights(
            raw_eeg=True,
            handler=self._handle_raw_event,
        )
        self._recording_started_at = _non_negative_finite(
            "recording start timestamp", self.clock()
        )
        if self.timestamp_mode == "production":
            self._mapper = GuardianTimestampMapper(
                anchor_run_timestamp=self._recording_started_at,
                anchor_vendor_timestamp=time.time(),
            )
            print("[recording] timestamp mapper: production start/Unix anchor")
        else:
            print("[recording] timestamp mapper: first raw block/host receipt anchor")
        self._started = True
        self._recording_task = asyncio.create_task(
            self._record(recording_seconds),
            name="guardian-playground-recording",
        )
        self._recording_task.add_done_callback(self._recording_finished)
        # Give start_recording one event-loop turn and surface an immediate error.
        await asyncio.sleep(0)
        if self._recording_task.done():
            try:
                result = self._recording_task.result()
            except BaseException as exc:
                self._set_failure(exc)
                raise
            self._capture_recording_id(result)
            raise RuntimeError(
                "Guardian start_recording returned before any raw EEG was received"
            )

    def drain(self, *, raise_on_failure: bool = True) -> tuple[EEGSample, ...]:
        """Return every queued raw sample without silently suppressing failures."""

        if raise_on_failure:
            self.raise_if_failed()
        drained = tuple(self._samples)
        self._samples.clear()
        if raise_on_failure:
            self.raise_if_failed()
        return drained

    async def stop(self) -> None:
        task = self._recording_task
        if task is None:
            return
        if not task.done():
            print("[recording] cancelling start_recording task")
            task.cancel()
        result = await asyncio.gather(task, return_exceptions=True)
        if result and isinstance(result[0], str):
            self._capture_recording_id(result[0])
        if not self._reorder_finalized and self._failure is None:
            self._finalize_raw_block_order()

    async def close(self) -> None:
        if self._closed:
            return
        await self.stop()
        print("[cleanup] awaiting client.disconnect_device()")
        await self.client.disconnect_device()
        self._closed = True
        print("[cleanup] Guardian disconnected")


@dataclass(frozen=True)
class BandSpec:
    name: str
    low_hz: float
    high_hz: float
    feature_name: str
    color_bgr: tuple[int, int, int]


BANDS = (
    BandSpec("THETA", 4.0, 8.0, "theta_power_4_8_hz", (255, 210, 80)),
    BandSpec("ALPHA", 8.0, 13.0, "alpha_power_8_13_hz", (100, 235, 120)),
    BandSpec("BETA", 13.0, 30.0, "beta_power_13_30_hz", (80, 165, 255)),
)


@dataclass(frozen=True)
class BandSnapshot:
    timestamps: np.ndarray
    traces_uv: dict[str, np.ndarray]
    powers_uv2: dict[str, float]
    duration_seconds: float
    effective_rate_hz: float
    gap_count: int


class DeferredClock:
    """Run-relative clock whose origin is set immediately before recording."""

    def __init__(self) -> None:
        self._origin: float | None = None

    def start(self) -> None:
        if self._origin is not None:
            raise RuntimeError("live monitor clock has already started")
        self._origin = time.perf_counter()

    def now(self) -> float:
        if self._origin is None:
            return 0.0
        return time.perf_counter() - self._origin


class RollingBandProcessor:
    """Keep raw samples and derive the three display bands on demand."""

    def __init__(self) -> None:
        self.buffer = EEGBuffer(history_seconds=WINDOW_SECONDS)
        self.preprocessor = EEGPreprocessor(
            sample_rate_hz=SAMPLE_RATE_HZ,
            low_hz=PREFILTER_LOW_HZ,
            high_hz=PREFILTER_HIGH_HZ,
            order=FILTER_ORDER,
        )
        self.feature_extractor = EEGFeatureExtractor(sample_rate_hz=SAMPLE_RATE_HZ)
        self._band_filters = {
            band.name: butter(
                FILTER_ORDER,
                [band.low_hz, band.high_hz],
                btype="bandpass",
                fs=SAMPLE_RATE_HZ,
                output="sos",
            )
            for band in BANDS
        }
        self.total_samples = 0
        self.drain_count = 0
        self.last_drain_size = 0
        self.last_timestamp: float | None = None

    def add_drain(self, samples: tuple[EEGSample, ...]) -> None:
        if not samples:
            return
        for sample in samples:
            self.buffer.add(sample)
        self.total_samples += len(samples)
        self.drain_count += 1
        self.last_drain_size = len(samples)
        self.last_timestamp = float(samples[-1].timestamp)

    def snapshot(self) -> BandSnapshot | None:
        if self.last_timestamp is None or len(self.buffer) < MINIMUM_DISPLAY_SAMPLES:
            return None
        requested_start = max(0.0, self.last_timestamp - WINDOW_SECONDS)
        window = self.buffer.window(requested_start, self.last_timestamp)
        if len(window.samples) < MINIMUM_DISPLAY_SAMPLES:
            return None

        timestamps = np.asarray(
            [sample.timestamp for sample in window.samples], dtype=np.float64
        )
        broadband = self.preprocessor.process(window)
        features = self.feature_extractor.extract(broadband)
        feature_values = dict(zip(FEATURE_NAMES, features, strict=True))
        traces = {
            band.name: np.asarray(
                sosfiltfilt(self._band_filters[band.name], broadband),
                dtype=np.float64,
            )
            for band in BANDS
        }
        powers = {
            band.name: float(feature_values[band.feature_name]) for band in BANDS
        }
        duration = float(timestamps[-1] - timestamps[0])
        effective_rate = (
            float((len(timestamps) - 1) / duration) if duration > 0.0 else 0.0
        )
        gap_count = int(
            np.count_nonzero(np.diff(timestamps) > (1.5 / SAMPLE_RATE_HZ))
        )
        return BandSnapshot(
            timestamps=timestamps,
            traces_uv=traces,
            powers_uv2=powers,
            duration_seconds=duration,
            effective_rate_hz=effective_rate,
            gap_count=gap_count,
        )


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative_finite(name: str, value: Any) -> float:
    result = _finite_number(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _battery_percent(value: Any) -> float:
    result = _finite_number("Guardian battery", value)
    if not 0.0 <= result <= 100.0:
        raise ValueError("Guardian battery must be within 0-100 percent")
    return result


def _text(
    image: np.ndarray,
    value: str,
    position: tuple[int, int],
    *,
    color: tuple[int, int, int] = (225, 230, 238),
    scale: float = 0.58,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _trace_segments(
    timestamps: np.ndarray, values: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    boundaries = np.flatnonzero(np.diff(timestamps) > (1.5 / SAMPLE_RATE_HZ)) + 1
    return tuple(
        (timestamp_run, value_run)
        for timestamp_run, value_run in zip(
            np.split(timestamps, boundaries),
            np.split(values, boundaries),
            strict=True,
        )
        if len(timestamp_run) >= 2
    )


def _draw_band_panel(
    image: np.ndarray,
    *,
    band: BandSpec,
    snapshot: BandSnapshot,
    top: int,
    bottom: int,
) -> None:
    left = 82
    right = image.shape[1] - 30
    plot_top = top + 34
    plot_bottom = bottom - 26
    center = (plot_top + plot_bottom) // 2
    plot_height = plot_bottom - plot_top
    values = snapshot.traces_uv[band.name]

    cv2.rectangle(image, (left, plot_top), (right, plot_bottom), (63, 68, 78), 1)
    cv2.line(image, (left, center), (right, center), (52, 57, 66), 1)
    for seconds_ago in (30, 20, 10, 0):
        fraction = (WINDOW_SECONDS - seconds_ago) / WINDOW_SECONDS
        x = int(round(left + fraction * (right - left)))
        cv2.line(image, (x, plot_top), (x, plot_bottom), (42, 47, 56), 1)
        _text(
            image,
            f"-{seconds_ago}s" if seconds_ago else "now",
            (x - 18, bottom - 6),
            color=(150, 157, 169),
            scale=0.4,
        )

    robust_peak = float(np.percentile(np.abs(values), 99.0))
    scale_uv = max(0.25, robust_peak * 1.15)
    _text(
        image,
        f"{band.name}  {band.low_hz:g}-{band.high_hz:g} Hz",
        (left, top + 22),
        color=band.color_bgr,
        scale=0.62,
        thickness=2,
    )
    _text(
        image,
        f"Welch power: {snapshot.powers_uv2[band.name]:.4g} uV^2",
        (left + 260, top + 22),
        color=(213, 218, 226),
        scale=0.53,
    )
    _text(
        image,
        f"+/-{scale_uv:.2f} uV",
        (right - 145, top + 22),
        color=(150, 157, 169),
        scale=0.45,
    )

    end_timestamp = float(snapshot.timestamps[-1])
    for timestamp_run, value_run in _trace_segments(snapshot.timestamps, values):
        max_points = max(2, 2 * (right - left))
        if len(timestamp_run) > max_points:
            indices = np.linspace(0, len(timestamp_run) - 1, max_points, dtype=int)
            timestamp_run = timestamp_run[indices]
            value_run = value_run[indices]
        x_coordinates = left + (
            (timestamp_run - (end_timestamp - WINDOW_SECONDS)) / WINDOW_SECONDS
        ) * (right - left)
        y_coordinates = center - (value_run / scale_uv) * (plot_height * 0.46)
        points = np.column_stack(
            (
                np.clip(np.rint(x_coordinates), left, right),
                np.clip(np.rint(y_coordinates), plot_top, plot_bottom),
            )
        ).astype(np.int32)
        cv2.polylines(image, [points], False, band.color_bgr, 1, cv2.LINE_AA)


def render_frame(
    *,
    processor: RollingBandProcessor,
    snapshot: BandSnapshot | None,
    preflight: GuardianPreflight | None,
    adapter: GuardianAdapter | None,
    status: str,
) -> np.ndarray:
    image = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), (18, 21, 27), np.uint8)
    _text(
        image,
        "IDUN GUARDIAN - RECORDING TROUBLESHOOTING",
        (30, 34),
        scale=0.72,
        thickness=2,
    )
    _text(
        image,
        "ESC cancels recording and disconnects",
        (900, 34),
        color=(150, 157, 169),
        scale=0.45,
    )

    battery = (
        "--"
        if preflight is None or preflight.battery_percent is None
        else f"{preflight.battery_percent:.0f}%"
    )
    impedance = (
        "--"
        if preflight is None or preflight.impedance_ohms is None
        else f"{preflight.impedance_ohms / 1000.0:.1f} kOhm"
    )
    _text(
        image,
        f"Status: {status}   Battery: {battery}   Impedance: {impedance}",
        (30, 64),
        color=(110, 225, 140) if status == "LIVE" else (180, 190, 205),
        scale=0.56,
    )

    duration = 0.0 if snapshot is None else min(
        WINDOW_SECONDS, snapshot.duration_seconds
    )
    rate = 0.0 if snapshot is None else snapshot.effective_rate_hz
    gap_count = 0 if snapshot is None else snapshot.gap_count
    _text(
        image,
        (
            f"Rolling window: {duration:4.1f}/{WINDOW_SECONDS:.0f}s   "
            f"retained: {len(processor.buffer)}   received: {processor.total_samples}   "
            f"rate: {rate:5.1f} Hz   gaps: {gap_count}"
        ),
        (30, 90),
        color=(190, 197, 209),
        scale=0.5,
    )
    if adapter is None:
        sdk_line = "SDK task: not-created   raw received/emitted: 0/0   pending: 0"
    else:
        sdk_line = (
            f"SDK task: {adapter.recording_task_state}   "
            f"raw received/emitted: {adapter.raw_block_received_count}/"
            f"{adapter.raw_block_count}   pending: {adapter.pending_raw_block_count}   "
            f"out-of-order: {adapter.out_of_order_raw_block_count}   "
            f"late: {adapter.late_raw_block_count}   "
            f"sequence gaps: {adapter.sequence_discontinuity_count}"
        )
    _text(
        image,
        sdk_line,
        (30, 114),
        color=(150, 157, 169),
        scale=0.45,
    )

    if snapshot is None:
        message = (
            "Waiting for the first raw 20-sample SDK block..."
            if status in {"STARTING", "LIVE"}
            else "Press SPACE after the selected preflight stage."
        )
        _text(
            image,
            message,
            (330, 420),
            color=(125, 205, 245),
            scale=0.68,
            thickness=2,
        )
        return image

    panel_top = 132
    panel_height = (CANVAS_HEIGHT - panel_top - 8) // len(BANDS)
    for index, band in enumerate(BANDS):
        top = panel_top + index * panel_height
        _draw_band_panel(
            image,
            band=band,
            snapshot=snapshot,
            top=top,
            bottom=top + panel_height - 4,
        )
    return image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mac-address",
        "--address",
        dest="address",
        metavar="MAC",
        help=(
            "Guardian Bluetooth MAC address; omit to let the SDK discover it "
            "(--address remains an alias)"
        ),
    )
    parser.add_argument(
        "--preflight-mode",
        choices=("manufacturer", "connect", "battery", "impedance"),
        default="manufacturer",
        help=(
            "manufacturer lets start_recording own connection; connect adds only "
            "explicit BLE connection; battery adds battery check; impedance adds "
            "the full production-like preflight"
        ),
    )
    parser.add_argument(
        "--mains-frequency-hz",
        type=int,
        choices=(50, 60),
        default=60,
        help="mains frequency for impedance mode (default: 60)",
    )
    parser.add_argument(
        "--impedance-seconds",
        type=float,
        default=2.0,
        help="impedance preflight duration (default: 2)",
    )
    parser.add_argument(
        "--max-impedance-ohms",
        type=float,
        default=300_000.0,
        help="hard impedance threshold (default: 300000)",
    )
    parser.add_argument(
        "--recording-seconds",
        type=int,
        default=DEFAULT_RECORDING_SECONDS,
        help=(
            "SDK recording timer; Esc normally stops earlier "
            f"(default: {DEFAULT_RECORDING_SECONDS})"
        ),
    )
    parser.add_argument(
        "--timestamp-mode",
        choices=("first-block", "production"),
        default="first-block",
        help=(
            "first-block avoids host/vendor clock skew; production reproduces the "
            "current source adapter timestamp anchor (default: first-block)"
        ),
    )
    parser.add_argument(
        "--first-block-timeout",
        type=float,
        default=45.0,
        help="fail if no raw block arrives within this many seconds (default: 45)",
    )
    parser.add_argument(
        "--update-hz",
        type=float,
        default=5.0,
        help="processing/display refresh frequency (default: 5)",
    )
    parser.add_argument(
        "--sdk-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "enable verbose official SDK diagnostics, including the one-second "
            "recording countdown (default: disabled)"
        ),
    )
    args = parser.parse_args()
    for name in (
        "impedance_seconds",
        "max_impedance_ohms",
        "first_block_timeout",
        "update_hz",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.recording_seconds <= 0:
        parser.error("--recording-seconds must be positive")
    return args


async def run_monitor(args: argparse.Namespace) -> int:
    print(f"Guardian playground build: {BUILD_ID}")
    processor = RollingBandProcessor()
    clock = DeferredClock()
    preflight: GuardianPreflight | None = None
    adapter: GuardianAdapter | None = None
    recording_started = False
    exit_reason = "startup failed"
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, CANVAS_WIDTH, CANVAS_HEIGHT)
        cv2.imshow(
            WINDOW_NAME,
            render_frame(
                processor=processor,
                snapshot=None,
                preflight=None,
                adapter=None,
                status="INITIALISING",
            ),
        )
        cv2.waitKey(1)

        api_token = load_guardian_api_token(
            environment_variable=TOKEN_ENVIRONMENT_VARIABLE,
            token_file=TOKEN_FILE,
            base_directory=PROJECT_ROOT,
        )
        adapter = GuardianAdapter(
            clock=clock.now,
            address=args.address,
            api_token=api_token,
            debug=args.sdk_debug,
            timestamp_mode=args.timestamp_mode,
        )
        print(f"Selected preflight mode: {args.preflight_mode}")
        preflight = await adapter.prepare(
            mode=args.preflight_mode,
            impedance_seconds=args.impedance_seconds,
            max_impedance_ohms=args.max_impedance_ohms,
            mains_frequency_hz=args.mains_frequency_hz,
        )

        cv2.imshow(
            WINDOW_NAME,
            render_frame(
                processor=processor,
                snapshot=None,
                preflight=preflight,
                adapter=adapter,
                status="READY - PRESS SPACE",
            ),
        )
        print("Press Space in the monitor window to launch start_recording, or Esc to stop.")
        start_requested = False
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord(" "):
                start_requested = True
                break
            if key == 27:
                exit_reason = "Esc pressed before recording"
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                exit_reason = "monitor window closed before recording"
                break
            await asyncio.sleep(0)

        if start_requested:
            clock.start()
            await adapter.start(recording_seconds=args.recording_seconds)
            recording_started = True
            exit_reason = "Guardian recording timer completed"
            print(
                "Recording coroutine launched on the main asyncio loop; "
                "waiting for the first raw block."
            )

            refresh_interval = 1.0 / args.update_hz
            next_refresh = 0.0
            latest_snapshot: BandSnapshot | None = None
            while True:
                processor.add_drain(adapter.drain())
                if (
                    adapter.raw_block_received_count == 0
                    and adapter.recording_elapsed_seconds >= args.first_block_timeout
                ):
                    raise RuntimeError(
                        "start_recording task is "
                        f"{adapter.recording_task_state}, but no raw 20-sample block "
                        f"arrived within {args.first_block_timeout:.1f} seconds"
                    )

                now = time.monotonic()
                if now >= next_refresh:
                    latest_snapshot = processor.snapshot()
                    cv2.imshow(
                        WINDOW_NAME,
                        render_frame(
                            processor=processor,
                            snapshot=latest_snapshot,
                            preflight=preflight,
                            adapter=adapter,
                            status=(
                                "LIVE"
                                if adapter.raw_block_received_count > 0
                                else "STARTING"
                            ),
                        ),
                    )
                    next_refresh = now + refresh_interval

                if cv2.waitKey(1) & 0xFF == 27:
                    exit_reason = "Esc pressed"
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = "monitor window closed"
                    break
                if adapter.recording_done:
                    adapter.raise_if_failed()
                    if adapter.raw_block_count == 0:
                        raise RuntimeError(
                            "start_recording finished without any raw EEG blocks"
                        )
                    break
                await asyncio.sleep(0.002)
    except KeyboardInterrupt:
        exit_reason = "keyboard interrupt"
    except BaseException as exc:
        primary_error = exc
    finally:
        if adapter is not None:
            if recording_started:
                try:
                    await adapter.stop()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            try:
                processor.add_drain(adapter.drain(raise_on_failure=False))
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                await adapter.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        cv2.destroyAllWindows()

    if primary_error is not None:
        print(
            f"Guardian playground failed: {type(primary_error).__name__}: "
            f"{primary_error}",
            file=sys.stderr,
        )
    for error in cleanup_errors:
        print(
            f"Guardian cleanup warning: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
    if primary_error is not None:
        return 1

    recording_id = None if adapter is None else adapter.recording_id
    raw_blocks = 0 if adapter is None else adapter.raw_block_count
    print(
        f"Stopped ({exit_reason}). Received {processor.total_samples} samples "
        f"from {raw_blocks} raw SDK blocks."
    )
    if recording_id is not None:
        print(f"IDUN recording ID: {recording_id}")
    return 1 if cleanup_errors else 0


def main() -> int:
    return asyncio.run(run_monitor(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
