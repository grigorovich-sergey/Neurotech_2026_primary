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

from eeg_pipeline.contracts import EEGSample

_STOP_POLL_SECONDS = 0.05
_WORKER_START_TIMEOUT_SECONDS = 10.0
_WORKER_STOP_TIMEOUT_SECONDS = 15.0
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class GuardianPreflight:
    """Guardian setup measurements collected before raw EEG starts."""

    battery_percent: float
    impedance_ohms: float | None


class GuardianQueueOverflowError(RuntimeError):
    """Raised instead of silently dropping raw EEG from the handoff queue."""


class GuardianTimestampMapper:
    """Map Guardian Unix seconds onto an integration-owned run-relative clock."""

    def __init__(
        self, *, anchor_run_timestamp: float, anchor_unix_timestamp: float
    ) -> None:
        self.anchor_run_timestamp = _host_timestamp(
            "anchor_run_timestamp", anchor_run_timestamp
        )
        self.anchor_unix_timestamp = _host_timestamp(
            "anchor_unix_timestamp", anchor_unix_timestamp
        )
        self._last_relative: float | None = None

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
        if self._last_relative is not None and relative < self._last_relative:
            raise ValueError("Guardian timestamps moved backwards")
        self._last_relative = relative
        return relative


class GuardianLiveParser:
    """Convert one SDK live-insights event into project EEGSample values."""

    def __init__(
        self,
        mapper: GuardianTimestampMapper,
        on_sample: Callable[[EEGSample], None],
        host_clock: Callable[[], float],
    ) -> None:
        if not callable(host_clock):
            raise TypeError("host_clock must be callable")
        self.mapper = mapper
        self.on_sample = on_sample
        self.host_clock = host_clock

    def __call__(self, event: Any) -> None:
        message = getattr(event, "message", event)
        if not isinstance(message, Mapping):
            raise ValueError("Guardian live event message must be a mapping")
        raw_eeg = message.get("raw_eeg")
        if not isinstance(raw_eeg, list):
            raise ValueError("Guardian live event is missing raw_eeg samples")
        host_receipt_timestamp = _host_timestamp(
            "host receipt timestamp", self.host_clock()
        )
        for raw_sample in raw_eeg:
            if not isinstance(raw_sample, Mapping):
                raise ValueError("Guardian raw_eeg entries must be mappings")
            if "timestamp" not in raw_sample or "ch1" not in raw_sample:
                raise ValueError("Guardian raw_eeg entries require timestamp and ch1")
            vendor_timestamp_unix = raw_sample["timestamp"]
            self.on_sample(
                EEGSample(
                    timestamp=self.mapper.map(vendor_timestamp_unix),
                    value_uv=raw_sample["ch1"],
                    valid=True,
                    vendor_timestamp_unix=vendor_timestamp_unix,
                    host_receipt_timestamp=host_receipt_timestamp,
                )
            )


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

    SDK callbacks append canonical samples to a bounded queue. Integration code
    remains responsible for draining and ingesting them on its own thread.
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
        self._failure: BaseException | None = None
        self._queue_overflowed = False

        self._worker_ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Any | None = None
        self._startup_error: BaseException | None = None

        self._prepared = False
        self._preflight: GuardianPreflight | None = None
        self._recording_started = False
        self._recording_done = threading.Event()
        self._recording_task: asyncio.Task[str | None] | None = None
        self._recording_id: str | None = None
        self._closed = False

    @property
    def client(self) -> Any | None:
        return self._client

    @property
    def preflight(self) -> GuardianPreflight | None:
        return self._preflight

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

    async def _impedance_preflight(
        self,
        *,
        duration_seconds: float,
        max_impedance_ohms: float,
        mains_frequency_hz: int,
    ) -> float:
        client = self._client
        if client is None:
            raise RuntimeError("Guardian client is unavailable")
        duration = _positive_number("impedance duration_seconds", duration_seconds)
        maximum = _positive_number("max_impedance_ohms", max_impedance_ohms)
        if mains_frequency_hz not in {50, 60}:
            raise ValueError("mains_frequency_hz must be 50 or 60")
        readings: list[float] = []

        def handle_impedance(value: Any) -> None:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError("Guardian impedance reading must be a finite number")
            readings.append(float(value))

        task = asyncio.create_task(
            client.stream_impedance(
                mains_freq_60hz=mains_frequency_hz == 60,
                handler=handle_impedance,
            )
        )
        try:
            await asyncio.sleep(duration)
        finally:
            client.stop_impedance()
            await task
        if not readings:
            raise RuntimeError("Guardian impedance preflight returned no readings")
        impedance = readings[-1]
        if impedance >= maximum:
            raise RuntimeError(
                f"Guardian impedance {impedance:.0f} ohm is not below configured "
                f"{maximum:.0f} ohm threshold"
            )
        return impedance

    async def _prepare_async(
        self,
        *,
        impedance_preflight_seconds: float | None,
        max_impedance_ohms: float,
        mains_frequency_hz: int,
    ) -> GuardianPreflight:
        client = self._client
        if client is None:
            raise RuntimeError("Guardian client is unavailable")
        try:
            await client.connect_device()
            battery_value = await client.check_battery()
            if (
                isinstance(battery_value, bool)
                or not isinstance(battery_value, Real)
                or not math.isfinite(float(battery_value))
                or not 0.0 <= float(battery_value) <= 100.0
            ):
                raise RuntimeError(
                    "Guardian battery preflight returned an invalid value"
                )
            impedance: float | None = None
            if impedance_preflight_seconds is not None:
                impedance = await self._impedance_preflight(
                    duration_seconds=impedance_preflight_seconds,
                    max_impedance_ohms=max_impedance_ohms,
                    mains_frequency_hz=mains_frequency_hz,
                )
            return GuardianPreflight(float(battery_value), impedance)
        except BaseException:
            try:
                await client.disconnect_device()
            except BaseException:
                pass
            raise

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
        self._ensure_worker()
        result = self._submit(
            self._prepare_async(
                impedance_preflight_seconds=duration,
                max_impedance_ohms=max_impedance_ohms,
                mains_frequency_hz=mains_frequency_hz,
            ),
            timeout_seconds=(None if duration is None else duration + 30.0),
        )
        with self._state_lock:
            self._preflight = result
            self._prepared = True
        return result

    def _enqueue_sample(self, sample: EEGSample) -> None:
        with self._sample_lock:
            if len(self._samples) >= self.queue_capacity_samples:
                overflow = GuardianQueueOverflowError(
                    "Guardian EEG handoff queue overflowed; no samples were dropped silently"
                )
                with self._failure_lock:
                    self._queue_overflowed = True
                    if self._failure is None:
                        self._failure = overflow
                task = self._recording_task
                loop = self._loop
                if task is not None and loop is not None:
                    loop.call_soon_threadsafe(task.cancel)
                raise overflow
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
        anchor_run_timestamp: float,
        anchor_unix_timestamp: float,
    ) -> None:
        client = self._client
        if client is None:
            raise RuntimeError("Guardian client is unavailable")
        self.mapper = GuardianTimestampMapper(
            anchor_run_timestamp=anchor_run_timestamp,
            anchor_unix_timestamp=anchor_unix_timestamp,
        )
        parser = GuardianLiveParser(self.mapper, self._enqueue_sample, self.clock)
        client.subscribe_live_insights(raw_eeg=True, handler=parser)
        self._recording_done.clear()
        self._recording_task = asyncio.create_task(
            client.start_recording(recording_timer=recording_seconds)
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
            if not self._prepared:
                raise RuntimeError("Guardian prepare() must complete before start()")
            if self._recording_started:
                raise RuntimeError("Guardian recording has already been started")
            self._recording_started = True
        with self._sample_lock:
            self._samples.clear()
        anchor_run_timestamp = _host_timestamp("clock timestamp", self.clock())
        anchor_unix_timestamp = _host_timestamp("Unix timestamp", time.time())
        try:
            self._submit(
                self._start_async(
                    recording_seconds=recording_seconds,
                    anchor_run_timestamp=anchor_run_timestamp,
                    anchor_unix_timestamp=anchor_unix_timestamp,
                )
            )
        except BaseException:
            with self._state_lock:
                self._recording_started = False
            raise

    def drain(self, *, cutoff_timestamp: float | None = None) -> tuple[EEGSample, ...]:
        """Return queued samples up to an optional closed timestamp cutoff."""

        cutoff = (
            None
            if cutoff_timestamp is None
            else _host_timestamp("cutoff_timestamp", cutoff_timestamp)
        )
        self._raise_failure()
        drained: list[EEGSample] = []
        with self._sample_lock:
            while self._samples:
                if cutoff is not None and self._samples[0].timestamp > cutoff:
                    break
                drained.append(self._samples.popleft())
        self._raise_failure()
        return tuple(drained)

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
                    self._stop_async(), timeout_seconds=_WORKER_STOP_TIMEOUT_SECONDS
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
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(_WORKER_STOP_TIMEOUT_SECONDS)
            if thread.is_alive() and error is None:
                error = RuntimeError(
                    "Guardian SDK worker did not stop within 15 seconds"
                )
        with self._state_lock:
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
