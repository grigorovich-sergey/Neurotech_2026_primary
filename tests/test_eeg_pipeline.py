import asyncio
from pathlib import Path

import h5py
import numpy as np
import pytest

from eeg_pipeline.buffer import EEGBuffer
from eeg_pipeline.contracts import EEGSample, QualityState
from eeg_pipeline.guardian import (
    GuardianAdapter,
    GuardianLiveParser,
    GuardianTimestampMapper,
)
from eeg_pipeline.pipeline import EEGPipeline
from eeg_pipeline.processing import FEATURE_NAMES, EEGFeatureExtractor, EEGPreprocessor, EEGQualityGate
from eeg_pipeline.recording import EEGHDF5Recorder, EEGHDF5Replay
from eeg_pipeline.synthetic import synthetic_eeg_samples


SAMPLE_RATE = 250.0


def _pipeline() -> EEGPipeline:
    return EEGPipeline(
        buffer=EEGBuffer(30.0),
        preprocessor=EEGPreprocessor(sample_rate_hz=SAMPLE_RATE),
        quality_gate=EEGQualityGate(sample_rate_hz=SAMPLE_RATE),
        feature_extractor=EEGFeatureExtractor(sample_rate_hz=SAMPLE_RATE),
    )


def _tone_samples(duration: float = 2.0) -> list[EEGSample]:
    return list(
        synthetic_eeg_samples(
            sample_rate_hz=SAMPLE_RATE,
            duration_seconds=duration,
            tones=[{"frequency_hz": 10.0, "amplitude_uv": 20.0}],
        )
    )


def test_guardian_timestamp_mapping_and_live_event_conversion() -> None:
    mapper = GuardianTimestampMapper(
        anchor_run_timestamp=2.0,
        anchor_unix_timestamp=1_700_000_000.0,
    )
    samples: list[EEGSample] = []
    parser = GuardianLiveParser(mapper, samples.append, lambda: 2.25)
    parser(
        {
            "raw_eeg": [
                {"timestamp": 1_700_000_000.004, "ch1": 12.5},
                {"timestamp": 1_700_000_000.008, "ch1": -3.0},
            ]
        }
    )

    assert samples[0].timestamp == pytest.approx(2.004, abs=1e-6)
    assert samples[1].timestamp == pytest.approx(2.008, abs=1e-6)
    assert [sample.value_uv for sample in samples] == [12.5, -3.0]
    assert [sample.vendor_timestamp_unix for sample in samples] == [
        1_700_000_000.004,
        1_700_000_000.008,
    ]
    assert [sample.host_receipt_timestamp for sample in samples] == [2.25, 2.25]
    with pytest.raises(ValueError, match="backwards"):
        mapper.map(1_700_000_000.006)
    with pytest.raises(ValueError, match="precedes"):
        GuardianTimestampMapper(
            anchor_run_timestamp=0.0,
            anchor_unix_timestamp=1_700_000_000.0,
        ).map(1_699_999_999.0)


def test_guardian_adapter_injects_shared_clock_and_anchors_after_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples: list[EEGSample] = []
    clock_values = iter((5.0, 5.25))

    class FakeClient:
        handler = None

        def subscribe_live_insights(self, *, raw_eeg: bool, handler: object) -> None:
            assert raw_eeg is True
            self.handler = handler

        async def start_recording(self, *, recording_timer: int) -> None:
            assert recording_timer == 1
            assert self.handler is not None
            self.handler(
                {"raw_eeg": [{"timestamp": 1_700_000_000.004, "ch1": 4.5}]}
            )

    client = FakeClient()
    monkeypatch.setattr("eeg_pipeline.guardian.time.time", lambda: 1_700_000_000.0)
    adapter = GuardianAdapter(
        clock=lambda: next(clock_values),
        client_factory=lambda **_: client,
    )

    adapter.run(
        recording_seconds=1,
        on_sample=samples.append,
        impedance_preflight_seconds=None,
    )

    assert adapter.mapper is not None
    assert adapter.mapper.anchor_run_timestamp == 5.0
    assert adapter.mapper.anchor_unix_timestamp == 1_700_000_000.0
    assert len(samples) == 1
    assert samples[0].timestamp == pytest.approx(5.004, abs=1e-6)
    assert samples[0].value_uv == 4.5
    assert samples[0].vendor_timestamp_unix == 1_700_000_000.004
    assert samples[0].host_receipt_timestamp == 5.25


def test_guardian_adapter_cooperatively_stops_and_awaits_cleanup() -> None:
    calls = 0

    class FakeClient:
        started = False
        cleaned = False

        def subscribe_live_insights(self, *, raw_eeg: bool, handler: object) -> None:
            assert raw_eeg is True

        async def start_recording(self, *, recording_timer: int) -> None:
            self.started = True
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned = True

    def stop_requested() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    client = FakeClient()
    adapter = GuardianAdapter(clock=lambda: 0.0, client_factory=lambda **_: client)

    adapter.run(
        recording_seconds=60,
        on_sample=lambda _: None,
        impedance_preflight_seconds=None,
        stop_requested=stop_requested,
    )

    assert client.started
    assert client.cleaned


def test_guardian_adapter_stops_during_impedance_preflight() -> None:
    calls = 0

    class FakeClient:
        impedance_running = False
        impedance_stopped = False
        recording_started = False

        async def stream_impedance(
            self, *, mains_freq_60hz: bool, handler: object
        ) -> None:
            self.impedance_running = True
            while self.impedance_running:
                await asyncio.sleep(0)

        def stop_impedance(self) -> None:
            self.impedance_running = False
            self.impedance_stopped = True

        async def start_recording(self, *, recording_timer: int) -> None:
            self.recording_started = True

    def stop_requested() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 3

    client = FakeClient()
    adapter = GuardianAdapter(clock=lambda: 0.0, client_factory=lambda **_: client)

    adapter.run(
        recording_seconds=60,
        on_sample=lambda _: None,
        impedance_preflight_seconds=10.0,
        stop_requested=stop_requested,
    )

    assert client.impedance_stopped
    assert not client.recording_started


def test_guardian_adapter_propagates_recording_failure_with_stop_callback() -> None:
    class FakeClient:
        def subscribe_live_insights(self, *, raw_eeg: bool, handler: object) -> None:
            assert raw_eeg is True

        async def start_recording(self, *, recording_timer: int) -> None:
            raise RuntimeError("recording failed")

    adapter = GuardianAdapter(
        clock=lambda: 0.0,
        client_factory=lambda **_: FakeClient(),
    )

    with pytest.raises(RuntimeError, match="recording failed"):
        adapter.run(
            recording_seconds=1,
            on_sample=lambda _: None,
            impedance_preflight_seconds=None,
            stop_requested=lambda: False,
        )


def test_window_is_closed_and_never_includes_sample_after_cutoff() -> None:
    buffer = EEGBuffer(10.0)
    for timestamp in (0.0, 0.1, 0.2, 0.3):
        buffer.add(EEGSample(timestamp, timestamp + 1.0))

    window = buffer.window(0.1, 0.2)

    assert [sample.timestamp for sample in window.samples] == [0.1, 0.2]
    assert window.actual_start == 0.1
    assert window.actual_end == 0.2
    assert all(sample.timestamp <= window.requested_end for sample in window.samples)


def test_future_sample_does_not_change_features_at_requested_cutoff() -> None:
    samples = _tone_samples(1.004)
    before_cutoff = _pipeline()
    with_future = _pipeline()
    for sample in samples:
        with_future.add_sample(sample)
        if sample.timestamp <= 1.0:
            before_cutoff.add_sample(sample)

    without_future = before_cutoff.features(0.0, 1.0)
    after_future_arrived = with_future.features(0.0, 1.0)

    assert without_future.quality_state is QualityState.USABLE
    assert after_future_arrived.quality_state is QualityState.USABLE
    np.testing.assert_array_equal(without_future.values, after_future_arrived.values)


def test_gap_is_unavailable_even_when_coverage_remains_high() -> None:
    pipeline = _pipeline()
    for sample in _tone_samples(1.0):
        if sample.timestamp != 0.5:
            pipeline.add_sample(sample)

    feature_window = pipeline.features(0.0, 1.0)

    assert feature_window.quality_state is QualityState.UNAVAILABLE
    assert "gap_exceeds_maximum" in feature_window.quality_reasons
    assert feature_window.values is None


def test_explicit_invalid_sample_is_rejected_and_has_no_features() -> None:
    pipeline = _pipeline()
    for sample in _tone_samples(1.0):
        pipeline.add_sample(
            EEGSample(sample.timestamp, sample.value_uv, valid=sample.timestamp != 0.5)
        )

    feature_window = pipeline.features(0.0, 1.0)

    assert feature_window.quality_state is QualityState.REJECTED
    assert "explicit_invalid_sample" in feature_window.quality_reasons
    assert feature_window.values is None


def test_peak_to_peak_quality_threshold_is_configurable_and_inclusive() -> None:
    samples = [
        EEGSample(index / SAMPLE_RATE, 1000.0 if index % 2 else 0.0)
        for index in range(251)
    ]
    buffer = EEGBuffer(2.0)
    for sample in samples:
        buffer.add(sample)
    window = buffer.window(0.0, 1.0)

    at_limit = EEGQualityGate(
        sample_rate_hz=SAMPLE_RATE, max_peak_to_peak_uv=1000.0
    ).evaluate(window)
    above_limit = EEGQualityGate(
        sample_rate_hz=SAMPLE_RATE, max_peak_to_peak_uv=999.0
    ).evaluate(window)

    assert at_limit[0] is QualityState.USABLE
    assert above_limit[0] is QualityState.REJECTED
    assert "peak_to_peak_exceeds_maximum" in above_limit[1]


def test_insufficient_duration_is_unavailable_and_has_no_features() -> None:
    pipeline = _pipeline()
    for sample in _tone_samples(0.5):
        pipeline.add_sample(sample)

    feature_window = pipeline.features(0.0, 0.5)

    assert feature_window.quality_state is QualityState.UNAVAILABLE
    assert "duration_below_minimum" in feature_window.quality_reasons
    assert feature_window.values is None


def test_known_alpha_tone_produces_stable_feature_order_and_dominant_alpha_power() -> None:
    pipeline = _pipeline()
    for sample in _tone_samples(2.0):
        pipeline.add_sample(sample)

    feature_window = pipeline.features(0.0, 2.0)

    assert feature_window.quality_state is QualityState.USABLE
    assert feature_window.feature_names == FEATURE_NAMES
    assert feature_window.values is not None
    spectral = feature_window.values[3:]
    assert spectral[2] == np.max(spectral)
    assert spectral[2] > 100 * max(spectral[0], spectral[1], spectral[3], spectral[4])


def test_raw_hdf5_round_trip_preserves_samples_and_validity(tmp_path: Path) -> None:
    original = [
        EEGSample(
            0.0,
            1.25,
            True,
            vendor_timestamp_unix=1_700_000_000.0,
            host_receipt_timestamp=0.02,
        ),
        EEGSample(0.004, -2.5, False),
        EEGSample(
            0.012,
            3.75,
            True,
            vendor_timestamp_unix=1_700_000_000.012,
            host_receipt_timestamp=0.03,
        ),
    ]
    path = tmp_path / "raw_eeg.h5"
    with EEGHDF5Recorder(path, sample_rate_hz=SAMPLE_RATE) as recorder:
        for sample in original:
            recorder.record(sample)

    replayed = list(EEGHDF5Replay(path).samples())

    assert replayed == original


def test_schema_v1_replay_marks_timing_metadata_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "legacy_raw_eeg.h5"
    with h5py.File(path, "w") as file:
        file.attrs["schema_version"] = 1
        file.attrs["sample_rate_hz"] = SAMPLE_RATE
        file.attrs["timebase"] = "run_relative_seconds"
        file.attrs["value_units"] = "microvolts"
        file.create_dataset("timestamp", data=np.array([0.0, 0.004]))
        file.create_dataset("value_uv", data=np.array([1.0, 2.0]))
        file.create_dataset("valid", data=np.array([True, False]))

    replayed = list(EEGHDF5Replay(path).samples())

    assert replayed == [EEGSample(0.0, 1.0, True), EEGSample(0.004, 2.0, False)]
    assert all(sample.vendor_timestamp_unix is None for sample in replayed)
    assert all(sample.host_receipt_timestamp is None for sample in replayed)


def test_repeated_replay_processing_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "raw_eeg.h5"
    with EEGHDF5Recorder(path, sample_rate_hz=SAMPLE_RATE) as recorder:
        for sample in _tone_samples(2.0):
            recorder.record(sample)

    results = []
    for _ in range(2):
        pipeline = _pipeline()
        for sample in EEGHDF5Replay(path).samples():
            pipeline.add_sample(sample)
        results.append(pipeline.features(0.0, 2.0))

    assert results[0].quality_state is QualityState.USABLE
    assert results[1].quality_state is QualityState.USABLE
    np.testing.assert_array_equal(results[0].values, results[1].values)


def test_synthetic_hdf5_replay_preserves_features(tmp_path: Path) -> None:
    original_pipeline = _pipeline()
    replay_pipeline = _pipeline()
    path = tmp_path / "raw_eeg.h5"
    with EEGHDF5Recorder(path, sample_rate_hz=SAMPLE_RATE) as recorder:
        for sample in _tone_samples(2.0):
            original_pipeline.add_sample(sample)
            recorder.record(sample)
    for sample in EEGHDF5Replay(path).samples():
        replay_pipeline.add_sample(sample)

    original = original_pipeline.features(0.0, 2.0)
    replayed = replay_pipeline.features(0.0, 2.0)

    assert original.quality_state is QualityState.USABLE
    assert replayed.quality_state is QualityState.USABLE
    assert original.feature_names == replayed.feature_names
    np.testing.assert_array_equal(original.values, replayed.values)
