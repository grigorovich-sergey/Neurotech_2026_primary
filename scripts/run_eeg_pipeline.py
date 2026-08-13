"""Run synthetic, replayed, or live Guardian EEG through the Instance 3 pipeline."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from eeg_pipeline.buffer import EEGBuffer
from eeg_pipeline.contracts import EEGFeatureWindow, EEGSample
from eeg_pipeline.credentials import load_guardian_api_token
from eeg_pipeline.guardian import GuardianAdapter
from eeg_pipeline.pipeline import EEGPipeline
from eeg_pipeline.processing import EEGFeatureExtractor, EEGPreprocessor, EEGQualityGate
from eeg_pipeline.recording import EEGHDF5Recorder, EEGHDF5Replay
from eeg_pipeline.synthetic import synthetic_eeg_samples
from foundations.config import load_resolved_config, save_resolved_config
from foundations.timebase import MonotonicClock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eeg_pipeline.yaml"


def _new_run_directory(output_root: str) -> Path:
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("output_root must be a non-empty path string")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = Path(output_root) / "eeg_pipeline" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _build_pipeline(config: dict[str, Any]) -> EEGPipeline:
    signal = _mapping(config, "signal")
    buffer_config = _mapping(config, "buffer")
    preprocessing = _mapping(config, "preprocessing")
    quality = _mapping(config, "quality")
    features = _mapping(config, "features")
    sample_rate = signal["sample_rate_hz"]
    return EEGPipeline(
        buffer=EEGBuffer(buffer_config["history_seconds"]),
        preprocessor=EEGPreprocessor(
            sample_rate_hz=sample_rate,
            low_hz=preprocessing["low_hz"],
            high_hz=preprocessing["high_hz"],
            order=preprocessing["order"],
        ),
        quality_gate=EEGQualityGate(
            sample_rate_hz=sample_rate,
            min_duration_seconds=quality["min_duration_seconds"],
            min_coverage=quality["min_coverage"],
            max_gap_seconds=quality["max_gap_seconds"],
            min_std_uv=quality["min_std_uv"],
            max_peak_to_peak_uv=quality["max_peak_to_peak_uv"],
        ),
        feature_extractor=EEGFeatureExtractor(
            sample_rate_hz=sample_rate,
            welch_nperseg_samples=features["welch_nperseg_samples"],
            welch_noverlap_samples=features["welch_noverlap_samples"],
        ),
    )


def _prepare_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(config)
    source = _mapping(resolved, "source")
    mode = source.get("mode")
    if mode not in {"synthetic", "replay", "live"}:
        raise ValueError("source.mode must be 'synthetic', 'replay', or 'live'")
    if not isinstance(_mapping(resolved, "recording").get("enabled"), bool):
        raise ValueError("recording.enabled must be a bool")
    window = _mapping(resolved, "window")
    start = window.get("start_seconds")
    end = window.get("end_seconds")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or start < 0
        or end < start
    ):
        raise ValueError("window must satisfy 0 <= start_seconds <= end_seconds")
    if mode == "replay":
        replay_path = source.get("replay_path")
        if not isinstance(replay_path, str) or not replay_path:
            raise ValueError("source.replay_path is required in replay mode")
        if not Path(replay_path).is_file():
            raise FileNotFoundError(replay_path)
        if not isinstance(source.get("replay_paced"), bool):
            raise ValueError("source.replay_paced must be a bool")
    return resolved


def _feature_summary(
    feature_window: EEGFeatureWindow,
    *,
    battery_percent: float | None,
    impedance_ohms: float | None,
    recording_id: str | None,
) -> dict[str, Any]:
    return {
        "requested_start": feature_window.requested_start,
        "requested_end": feature_window.requested_end,
        "actual_start": feature_window.actual_start,
        "actual_end": feature_window.actual_end,
        "sample_count": feature_window.sample_count,
        "completeness": feature_window.completeness.value,
        "quality_state": feature_window.quality_state.value,
        "quality_reasons": list(feature_window.quality_reasons),
        "feature_names": list(feature_window.feature_names),
        "values": (
            None
            if feature_window.values is None
            else [float(value) for value in feature_window.values]
        ),
        "guardian_battery_percent": battery_percent,
        "guardian_impedance_ohms": impedance_ohms,
        "guardian_recording_id": recording_id,
    }


def run_eeg_pipeline(config: dict[str, Any]) -> Path:
    """Execute one configured EEG run and persist raw input plus a feature summary."""

    resolved = _prepare_config(config)
    pipeline = _build_pipeline(resolved)
    run_directory = _new_run_directory(resolved["output_root"])
    save_resolved_config(resolved, run_directory / "resolved_config.json")

    source = resolved["source"]
    sample_rate = resolved["signal"]["sample_rate_hz"]
    recorder = (
        EEGHDF5Recorder(run_directory / "raw_eeg.h5", sample_rate_hz=sample_rate)
        if resolved["recording"]["enabled"]
        else None
    )

    def ingest(sample: EEGSample) -> None:
        if recorder is not None:
            recorder.record(sample)
        pipeline.add_sample(sample)

    battery_percent: float | None = None
    impedance_ohms: float | None = None
    recording_id: str | None = None
    try:
        if source["mode"] == "synthetic":
            synthetic = source["synthetic"]
            for sample in synthetic_eeg_samples(
                sample_rate_hz=sample_rate,
                duration_seconds=synthetic["duration_seconds"],
                tones=synthetic["tones"],
                noise_std_uv=synthetic["noise_std_uv"],
                seed=synthetic["seed"],
                gaps=synthetic["gaps"],
                invalid_intervals=synthetic["invalid_intervals"],
            ):
                ingest(sample)
        elif source["mode"] == "replay":
            for sample in EEGHDF5Replay(source["replay_path"]).replay(
                paced=source["replay_paced"]
            ):
                ingest(sample)
        else:
            guardian = source["guardian"]
            api_token = load_guardian_api_token(
                environment_variable=guardian["api_token_env"],
                token_file=guardian["api_token_file"],
                base_directory=PROJECT_ROOT,
            )
            standalone_clock: MonotonicClock | None = None

            def standalone_clock_now() -> float:
                nonlocal standalone_clock
                if standalone_clock is None:
                    standalone_clock = MonotonicClock()
                return standalone_clock.now()

            adapter = GuardianAdapter(
                clock=standalone_clock_now,
                address=guardian["address"],
                api_token=api_token,
                debug=guardian["debug"],
                queue_capacity_samples=guardian["queue_capacity_samples"],
            )
            impedance = guardian["impedance"]
            impedance_ohms = adapter.run(
                recording_seconds=guardian["recording_seconds"],
                on_sample=ingest,
                impedance_preflight_seconds=(
                    impedance["duration_seconds"] if impedance["enabled"] else None
                ),
                max_impedance_ohms=impedance["max_ohms"],
                mains_frequency_hz=impedance["mains_frequency_hz"],
            )
            preflight = adapter.preflight
            battery_percent = None if preflight is None else preflight.battery_percent
            recording_id = adapter.recording_id
    finally:
        if recorder is not None:
            recorder.close()

    window = resolved["window"]
    feature_window = pipeline.features(window["start_seconds"], window["end_seconds"])
    summary = _feature_summary(
        feature_window,
        battery_percent=battery_percent,
        impedance_ohms=impedance_ohms,
        recording_id=recording_id,
    )
    with (run_directory / "feature_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="partial YAML configuration overriding the project default",
    )
    args = parser.parse_args()
    try:
        config = load_resolved_config(DEFAULT_CONFIG, args.config)
        run_directory = run_eeg_pipeline(config)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(run_directory)
    print((run_directory / "feature_summary.json").read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()
