"""Thin official-IDUN-SDK adapter and Guardian timestamp mapping."""

import asyncio
from collections.abc import Callable, Mapping
import importlib
import math
from numbers import Real
import time
from typing import Any

from eeg_pipeline.contracts import EEGSample


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


class GuardianAdapter:
    """Run raw Guardian streaming through the SDK without exposing SDK objects downstream."""

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        address: str | None = None,
        api_token: str | None = None,
        debug: bool = False,
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
        self.clock = clock
        self.mapper: GuardianTimestampMapper | None = None
        if client_factory is None:
            try:
                module = importlib.import_module("idun_guardian_sdk")
            except ImportError as exc:
                raise RuntimeError(
                    "live Guardian mode requires installation with the 'guardian' extra"
                ) from exc
            client_factory = module.GuardianClient
        self.client = client_factory(address=address, api_token=api_token, debug=debug)

    async def _impedance_preflight(
        self,
        *,
        duration_seconds: float,
        max_impedance_ohms: float,
        mains_frequency_hz: int,
    ) -> float:
        if duration_seconds <= 0:
            raise ValueError("impedance duration_seconds must be positive")
        if max_impedance_ohms <= 0:
            raise ValueError("max_impedance_ohms must be positive")
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
            self.client.stream_impedance(
                mains_freq_60hz=mains_frequency_hz == 60,
                handler=handle_impedance,
            )
        )
        try:
            await asyncio.sleep(float(duration_seconds))
        finally:
            self.client.stop_impedance()
        await task
        if not readings:
            raise RuntimeError("Guardian impedance preflight returned no readings")
        impedance = readings[-1]
        if impedance >= max_impedance_ohms:
            raise RuntimeError(
                f"Guardian impedance {impedance:.0f} ohm is not below configured "
                f"{max_impedance_ohms:.0f} ohm threshold"
            )
        return impedance

    async def _run_async(
        self,
        *,
        recording_seconds: int,
        on_sample: Callable[[EEGSample], None],
        impedance_preflight_seconds: float | None,
        max_impedance_ohms: float,
        mains_frequency_hz: int,
    ) -> float | None:
        impedance: float | None = None
        if impedance_preflight_seconds is not None:
            impedance = await self._impedance_preflight(
                duration_seconds=impedance_preflight_seconds,
                max_impedance_ohms=max_impedance_ohms,
                mains_frequency_hz=mains_frequency_hz,
            )
        anchor_run_timestamp = _host_timestamp("clock timestamp", self.clock())
        anchor_unix_timestamp = _host_timestamp("Unix timestamp", time.time())
        self.mapper = GuardianTimestampMapper(
            anchor_run_timestamp=anchor_run_timestamp,
            anchor_unix_timestamp=anchor_unix_timestamp,
        )
        parser = GuardianLiveParser(self.mapper, on_sample, self.clock)
        self.client.subscribe_live_insights(raw_eeg=True, handler=parser)
        await self.client.start_recording(recording_timer=recording_seconds)
        return impedance

    def run(
        self,
        *,
        recording_seconds: int,
        on_sample: Callable[[EEGSample], None],
        impedance_preflight_seconds: float | None = 2.0,
        max_impedance_ohms: float = 300_000.0,
        mains_frequency_hz: int = 60,
    ) -> float | None:
        if (
            isinstance(recording_seconds, bool)
            or not isinstance(recording_seconds, int)
            or recording_seconds <= 0
        ):
            raise ValueError("recording_seconds must be a positive integer")
        return asyncio.run(
            self._run_async(
                recording_seconds=recording_seconds,
                on_sample=on_sample,
                impedance_preflight_seconds=impedance_preflight_seconds,
                max_impedance_ohms=max_impedance_ohms,
                mains_frequency_hz=mains_frequency_hz,
            )
        )


if __name__ == "__main__":
    mapper = GuardianTimestampMapper(
        anchor_run_timestamp=2.0, anchor_unix_timestamp=1_700_000_000.0
    )
    print(mapper.map(1_700_000_000.004))
