"""Persistent official-IDUN-SDK adapter and Guardian timestamp mapping."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import importlib
import math
from numbers import Real
import threading
import time
from typing import Any, Coroutine, TypeVar

from eeg_pipeline.contracts import EEGSample, EEGWindow, WindowCompleteness

_STOP_POLL_SECONDS = 0.05
_WORKER_START_TIMEOUT_SECONDS = 10.0
_WORKER_STOP_TIMEOUT_SECONDS = 15.0
_GUARDIAN_SAMPLE_RATE_HZ = 250.0
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class GuardianPreflight:
    """Guardian setup measurements collected before raw EEG starts."""

    battery_percent: float
    impedance_ohms: float | None


class GuardianQueueOverflowError(RuntimeError):
    """Raised instead of silently dropping raw EEG from the handoff queue."""


class GuardianTimestampMapper:
    """Map Guardian Unix seconds onto an integration-owned run-relative clock.

    Mapping is deliberately order-independent. Guardian callbacks may arrive out
    of timestamp order; ordering and late-data policy belong to GuardianAdapter's
    mutable timestamp grid rather than to this clock transform.
    """

    def __init__(
        self, *, anchor_run_timestamp: float, anchor_unix_timestamp: float
    ) -> None:
        self.anchor_run_timestamp = _host_timestamp(
            "anchor_run_timestamp", anchor_run_timestamp
        )
        self.anchor_unix_timestamp = _host_timestamp(
            "anchor_unix_timestamp", anchor_unix_timestamp
        )

    def map(self, guardian_unix_timestamp: float) -> float:
        if (
            isinstance(guardian_unix_timestamp, bool)
            or not isinstance(guardian_unix_timestamp, Real)
            or not math.isfinite(float(guardian_unix_timestamp))
        ):
            raise ValueError("Guardian timestamp must be finite")
        relative = self.anchor_run_timestamp + (
            float(guardian_unix_timestamp) - self.anchor_unix_timestamp
        )
        if relative < 0:
            raise ValueError("Guardian timestamp precedes the experiment origin")
        return relative


class GuardianLiveParser:
    """Convert one SDK live-insights event into project EEGSample values.

    When no mapper is supplied, the first raw block is anchored by aligning its
    final vendor sample to the nearest 250 Hz point around host receipt time. This
    avoids assuming that the Guardian clock and host Unix clock are synchronized.
    """

    def __init__(
        self,
        mapper: GuardianTimestampMapper | None,
        on_sample: Callable[[EEGSample], None],
        host_clock: Callable[[], float],
    ) -> None:
        if not callable(host_clock):
            raise TypeError("host_clock must be callable")
        self.mapper = mapper
        self.on_sample = on_sample
        self.host_clock = host_clock

    def parse(self, event: Any) -> tuple[EEGSample, ...]:
        message = getattr(event, "message", event)
        if not isinstance(message, Mapping):
            raise ValueError("Guardian live event message must be a mapping")
        raw_eeg = message.get("raw_eeg")
        if not isinstance(raw_eeg, list):
            raise ValueError("Guardian live event is missing raw_eeg samples")
        host_receipt_timestamp = _host_timestamp(
            "host receipt timestamp", self.host_clock()
        )
        parsed_raw: list[tuple[float, Any]] = []
        previous_vendor_timestamp: float | None = None
        for raw_sample in raw_eeg:
            if not isinstance(raw_sample, Mapping):
                raise ValueError("Guardian raw_eeg entries must be mappings")
            if "timestamp" not in raw_sample or "ch1" not in raw_sample:
                raise ValueError("Guardian raw_eeg entries require timestamp and ch1")
            vendor_timestamp_unix = _host_timestamp(
                "Guardian timestamp", raw_sample["timestamp"]
            )
            if (
                previous_vendor_timestamp is not None
                and vendor_timestamp_unix < previous_vendor_timestamp
            ):
                raise ValueError(
                    "Guardian timestamps moved backwards within a raw block"
                )
            previous_vendor_timestamp = vendor_timestamp_unix
            parsed_raw.append((vendor_timestamp_unix, raw_sample["ch1"]))

        if not parsed_raw:
            return ()
        if self.mapper is None:
            block_duration = parsed_raw[-1][0] - parsed_raw[0][0]
            receipt_slot = math.floor(
                host_receipt_timestamp * _GUARDIAN_SAMPLE_RATE_HZ + 0.5
            )
            last_run_timestamp = receipt_slot / _GUARDIAN_SAMPLE_RATE_HZ
            first_run_timestamp = max(0.0, last_run_timestamp - block_duration)
            self.mapper = GuardianTimestampMapper(
                anchor_run_timestamp=first_run_timestamp,
                anchor_unix_timestamp=parsed_raw[0][0],
            )

        return tuple(
            EEGSample(
                timestamp=self.mapper.map(vendor_timestamp_unix),
                value_uv=value_uv,
                valid=True,
                vendor_timestamp_unix=vendor_timestamp_unix,
                host_receipt_timestamp=host_receipt_timestamp,
            )
            for vendor_timestamp_unix, value_uv in parsed_raw
        )

    def __call__(self, event: Any) -> None:
        for sample in self.parse(event):
            self.on_sample(sample)


def _host_timestamp(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _positive_number(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


class GuardianAdapter:
    """Own one Guardian client and asyncio loop across preflight and recording.

    The preferred live path is window(): callbacks populate a mutable 250 Hz grid
    whose missing positions may be replaced by late packets while they remain in
    the moving request horizon. drain() remains as a one-way compatibility path.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        address: str | None = None,
        api_token: str | None = None,
        debug: bool = False,
        queue_capacity_samples: int = 15_000,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if address is not None and (not isinstance(address, str) or not address):
            raise ValueError("Guardian address must be a non-empty string or None")
        if api_token is not None and (not isinstance(api_token, str) or not api_token):
            raise ValueError("Guardian api_token must be a non-empty string or None")
        if not isinstance(debug, bool):
            raise TypeError("debug must be a bool")
        if not callable(clock):
            raise TypeError("clock must be callable")
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
                raise RuntimeError(
                    "live Guardian mode requires installation with the 'guardian' extra"
                ) from exc
            client_factory = module.GuardianClient

        self.clock = clock
        self.mapper: GuardianTimestampMapper | None = None
        self.queue_capacity_samples = queue_capacity_samples
        self._client_factory = client_factory
        self._client_kwargs = {
            "address": address,
            "api_token": api_token,
            "debug": debug,
        }

        self._state_lock = threading.RLock()
        self._failure_lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._samples: deque[EEGSample] = deque()
        self._window_samples: dict[int, EEGSample] = {}
        self._consumer_mode: str | None = None
        self._closed_before_slot = 0
        self._finalize_cursor_slot = 0
        self._latest_window_start: float | None = None
        self._lost_sample_count = 0
        self._lost_block_count = 0
        self._failure: BaseException | None = None
        self._queue_overflowed = False

        self._worker_ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Any | None = None
        self._startup_error: BaseException | None = None

        self._connected = False
        self._prepared = False
        self._preflight: GuardianPreflight | None = None
        self._impedance_task: asyncio.Task[None] | None = None
        self._latest_impedance_ohms: float | None = None
        self._recording_started = False
        self._recording_done = threading.Event()
        self._recording_task: asyncio.Task[str | None] | None = None
        self._live_parser: GuardianLiveParser | None = None
        self._recording_id: str | None = None
        self._closed = False

    @property
    def client(self) -> Any | None:
        return self._client

    @property
    def preflight(self) -> GuardianPreflight | None:
        return self._preflight

    @property
    def connected(self) -> bool:
        with self._state_lock:
            return self._connected

    @property
    def recording_id(self) -> str | None:
        with self._state_lock:
            return self._recording_id

    @property
    def recording_active(self) -> bool:
        with self._state_lock:
            return self._recording_started and not self._recording_done.is_set()

    @property
    def recording_done(self) -> bool:
        with self._state_lock:
            return self._recording_started and self._recording_done.is_set()

    @property
    def queue_overflowed(self) -> bool:
        with self._failure_lock:
            return self._queue_overflowed

    @property
    def lost_sample_count(self) -> int:
        with self._sample_lock:
            return self._lost_sample_count

    @property
    def lost_block_count(self) -> int:
        with self._sample_lock:
            return self._lost_block_count

    def _worker_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            self._client = self._client_factory(**self._client_kwargs)
        except BaseException as exc:
            self._startup_error = exc
            self._worker_ready.set()
            loop.close()
            return
        self._worker_ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()

    def _ensure_worker(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Guardian adapter is closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._worker_main,
                    name="guardian_sdk_loop",
                    daemon=True,
                )
                self._thread.start()
        if not self._worker_ready.wait(_WORKER_START_TIMEOUT_SECONDS):
            raise RuntimeError("Guardian SDK worker did not start within 10 seconds")
        if self._startup_error is not None:
            raise RuntimeError(
                "Guardian SDK client construction failed"
            ) from self._startup_error

    def _submit(
        self,
        coroutine: Coroutine[Any, Any, _T],
        *,
        timeout_seconds: float | None = None,
    ) -> _T:
        loop = self._loop
        if loop is None or not loop.is_running():
            coroutine.close()
            raise RuntimeError("Guardian SDK worker is not running")
        if threading.current_thread() is self._thread:
            coroutine.close()
            raise RuntimeError("Guardian blocking API cannot run on its SDK worker")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise RuntimeError("Guardian SDK operation timed out") from exc

    def _set_failure(self, exc: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = exc

    def _raise_failure(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise failure

    async def _connect_async(self) -> None:
        client = self._client
        if client is None:
            raise RuntimeError("Guardian client is unavailable")
        await client.connect_device()

    def connect(self) -> None:
        """Connect Guardian over BLE without starting impedance or recording."""

        with self._state_lock:
            if self._connected:
                return
        self._ensure_worker()
        try:
            self._submit(self._connect_async(), timeout_seconds=30.0)
        except BaseException:
            try:
                self._submit(self._disconnect_async(), timeout_seconds=15.0)
            except BaseException:
                pass
            raise
        with self._state_lock:
            self._connected = True

    async def _check_battery_async(self) -> float:
        client = self._client
        if client is None:
            raise RuntimeError("Guardian client is unavailable")
        battery_value = await client.check_battery()
        if (
            isinstance(battery_value, bool)
            or not isinstance(battery_value, Real)
            or not math.isfinite(float(battery_value))
            or not 0.0 <= float(battery_value) <= 100.0
        ):
            raise RuntimeError("Guardian battery check returned an invalid value")
        return float(battery_value)

    def check_battery(self) -> float:
        """Return battery percentage for an already-connected Guardian."""

        if not self.connected:
            raise RuntimeError("Guardian connect() must complete before battery check")
        return self._submit(self._check_battery_async(), timeout_seconds=15.0)

    def _impedance_finished(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except BaseException as exc:
            self._set_failure(exc)

    async def _start_impedance_async(self, *, mains_frequency_hz: int) -> None:
        client = self._client
        if client is None:
            raise RuntimeError("Guardian client is unavailable")

        def handle_impedance(value: Any) -> None:
            impedance = _host_timestamp("Guardian impedance reading", value)
            with self._state_lock:
                self._latest_impedance_ohms = impedance

        self._impedance_task = asyncio.create_task(
            client.stream_impedance(
                mains_freq_60hz=mains_frequency_hz == 60,
                handler=handle_impedance,
            )
        )
        self._impedance_task.add_done_callback(self._impedance_finished)
        await asyncio.sleep(0)

    def start_impedance(self, *, mains_frequency_hz: int = 60) -> None:
        """Start continuous impedance measurements without starting raw EEG."""

        if mains_frequency_hz not in {50, 60}:
            raise ValueError("mains_frequency_hz must be 50 or 60")
        if not self.connected:
            raise RuntimeError("Guardian connect() must complete before impedance")
        with self._state_lock:
            if self._recording_started:
                raise RuntimeError("Guardian impedance cannot start during recording")
            if self._impedance_task is not None and not self._impedance_task.done():
                raise RuntimeError("Guardian impedance stream is already active")
            self._latest_impedance_ohms = None
        self._submit(
            self._start_impedance_async(mains_frequency_hz=mains_frequency_hz),
            timeout_seconds=15.0,
        )

    async def _stop_impedance_async(self) -> None:
        task = self._impedance_task
        if task is None:
            return
        client = self._client
        if client is not None and not task.done():
            client.stop_impedance()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._impedance_task = None

    def stop_impedance(self) -> None:
        """Stop and await the active impedance stream."""

        loop = self._loop
        if loop is not None and loop.is_running():
            self._submit(
                self._stop_impedance_async(),
                timeout_seconds=_WORKER_STOP_TIMEOUT_SECONDS,
            )
        self._raise_failure()

    def latest_impedance(self) -> float | None:
        """Return the newest impedance reading, or None before the first reading."""

        self._raise_failure()
        with self._state_lock:
            return self._latest_impedance_ohms

    def check_health(self) -> None:
        """Surface an asynchronous acquisition failure on the caller thread."""

        self._raise_failure()

    def prepare(
        self,
        *,
        impedance_preflight_seconds: float | None = 2.0,
        max_impedance_ohms: float = 300_000.0,
        mains_frequency_hz: int = 60,
    ) -> GuardianPreflight:
        """Connect and check battery/impedance without starting raw EEG."""

        duration = (
            None
            if impedance_preflight_seconds is None
            else _positive_number(
                "impedance_preflight_seconds", impedance_preflight_seconds
            )
        )
        _positive_number("max_impedance_ohms", max_impedance_ohms)
        if mains_frequency_hz not in {50, 60}:
            raise ValueError("mains_frequency_hz must be 50 or 60")
        with self._state_lock:
            if self._prepared:
                assert self._preflight is not None
                return self._preflight
        try:
            self.connect()
            battery_percent = self.check_battery()
            impedance: float | None = None
            if duration is not None:
                self.start_impedance(mains_frequency_hz=mains_frequency_hz)
                try:
                    time.sleep(duration)
                finally:
                    self.stop_impedance()
                impedance = self.latest_impedance()
                if impedance is None:
                    raise RuntimeError(
                        "Guardian impedance preflight returned no readings"
                    )
                if impedance >= max_impedance_ohms:
                    raise RuntimeError(
                        f"Guardian impedance {impedance:.0f} ohm is not below "
                        f"configured {max_impedance_ohms:.0f} ohm threshold"
                    )
            result = GuardianPreflight(battery_percent, impedance)
        except BaseException:
            try:
                self.disconnect()
            except BaseException:
                pass
            raise
        with self._state_lock:
            self._preflight = result
            self._prepared = True
        return result

    @staticmethod
    def _sample_slot(timestamp: float) -> int:
        return int(round(timestamp * _GUARDIAN_SAMPLE_RATE_HZ))

    @staticmethod
    def _first_slot_at_or_after(timestamp: float) -> int:
        return int(math.ceil(timestamp * _GUARDIAN_SAMPLE_RATE_HZ - 1e-9))

    @staticmethod
    def _last_slot_at_or_before(timestamp: float) -> int:
        return int(math.floor(timestamp * _GUARDIAN_SAMPLE_RATE_HZ + 1e-9))

    @staticmethod
    def _sample_at_slot(sample: EEGSample, slot: int) -> EEGSample:
        return EEGSample(
            timestamp=slot / _GUARDIAN_SAMPLE_RATE_HZ,
            value_uv=sample.value_uv,
            valid=sample.valid,
            vendor_timestamp_unix=sample.vendor_timestamp_unix,
            host_receipt_timestamp=sample.host_receipt_timestamp,
        )

    def _overflow(self) -> GuardianQueueOverflowError:
        overflow = GuardianQueueOverflowError(
            "Guardian EEG handoff capacity overflowed; no samples were dropped silently"
        )
        with self._failure_lock:
            self._queue_overflowed = True
            if self._failure is None:
                self._failure = overflow
        task = self._recording_task
        loop = self._loop
        if task is not None and loop is not None:
            loop.call_soon_threadsafe(task.cancel)
        return overflow

    def _enqueue_samples(self, samples: tuple[EEGSample, ...]) -> None:
        canonical = tuple(
            (self._sample_slot(sample.timestamp), sample) for sample in samples
        )
        with self._sample_lock:
            accepted = [
                (slot, self._sample_at_slot(sample, slot))
                for slot, sample in canonical
                if slot >= self._closed_before_slot
            ]
            lost_count = len(canonical) - len(accepted)
            if self._consumer_mode == "window":
                new_slots = {
                    slot
                    for slot, _sample in accepted
                    if slot not in self._window_samples
                }
                would_overflow = (
                    len(self._window_samples) + len(new_slots)
                    > self.queue_capacity_samples
                )
            else:
                would_overflow = (
                    len(self._samples) + len(accepted) > self.queue_capacity_samples
                )
            if would_overflow:
                raise self._overflow()

            if lost_count:
                self._lost_sample_count += lost_count
                self._lost_block_count += 1
            for slot, sample in accepted:
                if self._consumer_mode != "drain":
                    self._window_samples.setdefault(slot, sample)
                if self._consumer_mode != "window":
                    self._samples.append(sample)

    def _capture_recording_id(self, candidate: Any = None) -> None:
        recording_id = candidate if isinstance(candidate, str) and candidate else None
        if recording_id is None and self._client is not None:
            getter = getattr(self._client, "get_recording_id", None)
            if callable(getter):
                try:
                    value = getter()
                except (AttributeError, RuntimeError, ValueError):
                    value = None
                if isinstance(value, str) and value:
                    recording_id = value
        if recording_id is not None:
            with self._state_lock:
                self._recording_id = recording_id

    def _recording_finished(self, task: asyncio.Task[str | None]) -> None:
        try:
            result = task.result()
        except asyncio.CancelledError:
            result = None
        except BaseException as exc:
            self._set_failure(exc)
            result = None
        self._capture_recording_id(result)
        self._recording_done.set()

    async def _start_async(
        self,
        *,
        recording_seconds: int,
    ) -> None:
        client = self._client
        if client is None:
            raise RuntimeError("Guardian client is unavailable")
        parser = GuardianLiveParser(None, lambda _sample: None, self.clock)

        def handle_raw_event(event: Any) -> None:
            parsed = parser.parse(event)
            self.mapper = parser.mapper
            self._enqueue_samples(parsed)

        self._live_parser = parser
        client.subscribe_live_insights(raw_eeg=True, handler=handle_raw_event)
        self._recording_done.clear()
        self._recording_task = asyncio.create_task(
            client.start_recording(
                recording_timer=recording_seconds,
                led_sleep=False,
                calc_latency=False,
            )
        )
        self._recording_task.add_done_callback(self._recording_finished)

    def start(self, *, recording_seconds: int) -> None:
        """Start raw EEG after the integration-owned attempt clock has begun."""

        if (
            isinstance(recording_seconds, bool)
            or not isinstance(recording_seconds, int)
            or recording_seconds <= 0
        ):
            raise ValueError("recording_seconds must be a positive integer")
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Guardian adapter is closed")
            if not self._connected:
                raise RuntimeError(
                    "Guardian connect() or prepare() must complete before start()"
                )
            if self._recording_started:
                raise RuntimeError("Guardian recording has already been started")
            if self._impedance_task is not None and not self._impedance_task.done():
                raise RuntimeError("stop Guardian impedance before starting recording")
            self._recording_started = True
        with self._sample_lock:
            self._samples.clear()
            self._window_samples.clear()
            self._consumer_mode = None
            self._closed_before_slot = 0
            self._finalize_cursor_slot = 0
            self._latest_window_start = None
            self._lost_sample_count = 0
            self._lost_block_count = 0
        self.mapper = None
        try:
            self._submit(self._start_async(recording_seconds=recording_seconds))
        except BaseException:
            with self._state_lock:
                self._recording_started = False
            raise

    def drain(self, *, cutoff_timestamp: float | None = None) -> tuple[EEGSample, ...]:
        """Legacy one-way drain; do not combine with replaceable window snapshots."""

        cutoff = (
            None
            if cutoff_timestamp is None
            else _host_timestamp("cutoff_timestamp", cutoff_timestamp)
        )
        self._raise_failure()
        with self._sample_lock:
            if self._consumer_mode == "window":
                raise RuntimeError("Guardian drain() cannot follow window() requests")
            if self._consumer_mode is None:
                self._consumer_mode = "drain"
                self._window_samples.clear()
            drained = [
                sample
                for sample in self._samples
                if cutoff is None or sample.timestamp <= cutoff
            ]
            self._samples = deque(
                sample
                for sample in self._samples
                if cutoff is not None and sample.timestamp > cutoff
            )
            drained.sort(key=lambda sample: sample.timestamp)
        self._raise_failure()
        return tuple(drained)

    def window(self, start: float, end: float) -> EEGWindow:
        """Return the current ordered 250 Hz snapshot for one closed interval.

        Missing positions are explicit invalid samples. Repeating the same start,
        or requesting a later overlapping interval, reflects packets that arrived
        since the previous request. Starts must advance monotonically; samples
        older than the newest start are thereafter counted as lost if they arrive.
        """

        requested_start = _host_timestamp("window start", start)
        requested_end = _host_timestamp("window end", end)
        if requested_end < requested_start:
            raise ValueError("window bounds must satisfy 0 <= start <= end")
        self._raise_failure()
        with self._sample_lock:
            if self._consumer_mode == "drain":
                raise RuntimeError("Guardian window() cannot follow drain() requests")
            if (
                self._latest_window_start is not None
                and requested_start < self._latest_window_start - 1e-9
            ):
                raise ValueError("Guardian window starts must be non-decreasing")
            if self._consumer_mode is None:
                self._consumer_mode = "window"
                self._samples.clear()

            first_slot = self._first_slot_at_or_after(requested_start)
            last_slot = self._last_slot_at_or_before(requested_end)
            if first_slot < self._closed_before_slot:
                raise ValueError("Guardian window starts before finalized data")
            self._latest_window_start = requested_start
            self._closed_before_slot = max(self._closed_before_slot, first_slot)
            self._finalize_cursor_slot = max(
                self._finalize_cursor_slot, self._closed_before_slot
            )
            self._window_samples = {
                slot: sample
                for slot, sample in self._window_samples.items()
                if slot >= self._closed_before_slot
            }

            if last_slot < first_slot:
                selected: tuple[EEGSample, ...] = ()
            else:
                selected = tuple(
                    self._window_samples.get(
                        slot,
                        EEGSample(
                            timestamp=slot / _GUARDIAN_SAMPLE_RATE_HZ,
                            value_uv=0.0,
                            valid=False,
                        ),
                    )
                    for slot in range(first_slot, last_slot + 1)
                )
        self._raise_failure()
        return EEGWindow(
            requested_start=requested_start,
            requested_end=requested_end,
            samples=selected,
            actual_start=(selected[0].timestamp if selected else None),
            actual_end=(selected[-1].timestamp if selected else None),
            completeness=(
                WindowCompleteness.COMPLETE if selected else WindowCompleteness.EMPTY
            ),
        )

    def finalize_before(self, timestamp: float) -> tuple[EEGSample, ...]:
        """Close and return the chronological grid strictly before timestamp.

        This is the persistence path: call it before advancing past data that a
        local raw recorder must retain. Missing positions are emitted invalid and
        late arrivals before the finalized boundary are subsequently discarded.
        """

        boundary = _host_timestamp("finalize timestamp", timestamp)
        self._raise_failure()
        with self._sample_lock:
            if self._consumer_mode == "drain":
                raise RuntimeError("Guardian finalize_before() cannot follow drain()")
            if self._consumer_mode is None:
                self._consumer_mode = "window"
                self._samples.clear()
            boundary_slot = self._first_slot_at_or_after(boundary)
            if boundary_slot < self._finalize_cursor_slot:
                raise ValueError(
                    "Guardian finalization boundaries must be non-decreasing"
                )
            finalized = tuple(
                self._window_samples.get(
                    slot,
                    EEGSample(
                        timestamp=slot / _GUARDIAN_SAMPLE_RATE_HZ,
                        value_uv=0.0,
                        valid=False,
                    ),
                )
                for slot in range(self._finalize_cursor_slot, boundary_slot)
            )
            self._closed_before_slot = max(self._closed_before_slot, boundary_slot)
            self._finalize_cursor_slot = boundary_slot
            self._window_samples = {
                slot: sample
                for slot, sample in self._window_samples.items()
                if slot >= boundary_slot
            }
        self._raise_failure()
        return finalized

    async def _stop_async(self) -> None:
        task = self._recording_task
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            result = await task
        except asyncio.CancelledError:
            result = None
        except BaseException as exc:
            self._set_failure(exc)
            result = None
        self._capture_recording_id(result)
        self._recording_done.set()

    def stop(self) -> None:
        """Cancel and await recording cleanup on the SDK owner loop."""

        with self._state_lock:
            started = self._recording_started
        if started:
            self._submit(
                self._stop_async(), timeout_seconds=_WORKER_STOP_TIMEOUT_SECONDS
            )
        self._raise_failure()

    async def _disconnect_async(self) -> None:
        if self._client is not None:
            await self._client.disconnect_device()

    def disconnect(self) -> None:
        """Disconnect BLE while leaving the reusable SDK owner loop alive."""

        if self.recording_active:
            raise RuntimeError("stop Guardian recording before disconnecting")
        loop = self._loop
        if loop is None or not loop.is_running() or not self.connected:
            return
        error: BaseException | None = None
        try:
            self._submit(
                self._stop_impedance_async(),
                timeout_seconds=_WORKER_STOP_TIMEOUT_SECONDS,
            )
        except BaseException as exc:
            error = exc
        try:
            self._submit(
                self._disconnect_async(),
                timeout_seconds=_WORKER_STOP_TIMEOUT_SECONDS,
            )
        except BaseException as exc:
            if error is None:
                error = exc
        with self._state_lock:
            self._connected = False
            self._prepared = False
            self._preflight = None
        if error is not None:
            raise error

    def close(self) -> None:
        """Stop recording, disconnect, and terminate the SDK owner loop."""

        with self._state_lock:
            if self._closed:
                return
        error: BaseException | None = None
        loop = self._loop
        thread = self._thread
        if loop is not None and loop.is_running():
            try:
                self._submit(
                    self._stop_impedance_async(),
                    timeout_seconds=_WORKER_STOP_TIMEOUT_SECONDS,
                )
            except BaseException as exc:
                error = exc
            try:
                self._submit(
                    self._stop_async(), timeout_seconds=_WORKER_STOP_TIMEOUT_SECONDS
                )
            except BaseException as exc:
                if error is None:
                    error = exc
            if self.connected:
                try:
                    self._submit(
                        self._disconnect_async(),
                        timeout_seconds=_WORKER_STOP_TIMEOUT_SECONDS,
                    )
                except BaseException as exc:
                    if error is None:
                        error = exc
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(_WORKER_STOP_TIMEOUT_SECONDS)
            if thread.is_alive() and error is None:
                error = RuntimeError(
                    "Guardian SDK worker did not stop within 15 seconds"
                )
        with self._state_lock:
            self._connected = False
            self._closed = True
        if error is None:
            try:
                self._raise_failure()
            except BaseException as exc:
                error = exc
        if error is not None:
            raise error

    def run(
        self,
        *,
        recording_seconds: int,
        on_sample: Callable[[EEGSample], None],
        impedance_preflight_seconds: float | None = 2.0,
        max_impedance_ohms: float = 300_000.0,
        mains_frequency_hz: int = 60,
        stop_requested: Callable[[], bool] | None = None,
    ) -> float | None:
        """Blocking compatibility wrapper around prepare/start/drain/stop/close."""

        if not callable(on_sample):
            raise TypeError("on_sample must be callable")
        if stop_requested is not None and not callable(stop_requested):
            raise TypeError("stop_requested must be callable or None")
        if stop_requested is not None and stop_requested():
            return None
        preflight: GuardianPreflight | None = None
        try:
            preflight = self.prepare(
                impedance_preflight_seconds=impedance_preflight_seconds,
                max_impedance_ohms=max_impedance_ohms,
                mains_frequency_hz=mains_frequency_hz,
            )
            if stop_requested is not None and stop_requested():
                return preflight.impedance_ohms
            self.start(recording_seconds=recording_seconds)
            while not self.recording_done:
                for sample in self.drain():
                    on_sample(sample)
                if stop_requested is not None and stop_requested():
                    break
                time.sleep(_STOP_POLL_SECONDS)
            self.stop()
            for sample in self.drain():
                on_sample(sample)
            return preflight.impedance_ohms
        finally:
            self.close()


if __name__ == "__main__":
    mapper = GuardianTimestampMapper(
        anchor_run_timestamp=2.0, anchor_unix_timestamp=1_700_000_000.0
    )
    print(mapper.map(1_700_000_000.004))
