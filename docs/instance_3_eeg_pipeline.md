# Instance 3 — Guardian + EEG pipeline

`eeg_pipeline` turns ordered single-channel raw EEG into timestamp-addressable,
quality-aware generic features. It is deliberately independent of gaze episodes,
feedback labels, and the learning models.

## Time and window contract

`EEGSample(timestamp, value_uv, valid)` uses non-negative run-relative seconds and
microvolts. `EEGBuffer.window(start, end)` uses a **closed `[start, end]` interval**;
samples later than `end` are never returned or passed to preprocessing.

`EEGWindow` reports the requested interval, actual first/last returned sample,
raw samples, and `complete | partial | empty` availability. A partial boundary is
not fabricated or clipped. Quality is decided separately using the configured
duration, coverage, gap, validity, flatline, and amplitude criteria.

Instance 4 should consume `eeg_pipeline.contracts.EEGFeatureWindow`. Its fields are:

- `requested_start`, `requested_end`, `actual_start`, `actual_end`;
- `sample_count` and `completeness`;
- `quality_state` (`usable`, `unavailable`, or `rejected`) and `quality_reasons`;
- stable `feature_names` plus a one-dimensional `float64` `values` array when
  usable, otherwise `values=None`.

The feature contract is episode-agnostic. Instance 4 chooses which requested EEG
interval corresponds to an experiment episode.

## Processing version/configuration

The complete defaults are in `configs/eeg_pipeline.yaml`; every run saves the
fully resolved configuration. The initial processing is a 4th-order 1–40 Hz
Butterworth SOS band-pass at 250 Hz using zero-phase `sosfiltfilt`. There is no
resampling, interpolation, notch filtering, or per-window normalization.

Quality defaults are: minimum duration 1.0 s, minimum coverage 0.95, maximum
sample gap 0.006 s, minimum raw standard deviation 0.05 µV, and maximum raw
peak-to-peak amplitude 1000 µV. Missing/insufficient/gapped data is unavailable;
explicit invalidity, flatline, and gross amplitude violations are rejected.
These are engineering defaults to revisit with real Guardian data.

Feature order is fixed:

1. `std_uv`
2. `peak_to_peak_uv`
3. `mean_abs_diff_uv`
4. `delta_power_1_4_hz`
5. `theta_power_4_8_hz`
6. `alpha_power_8_13_hz`
7. `beta_power_13_30_hz`
8. `low_gamma_power_30_40_hz`

Band powers are absolute Welch PSD integrals. Defaults use 250 samples per Welch
segment and 125-sample overlap. Interpret derived feature values together with
the run's saved resolved configuration.

## Raw recording and replay

`EEGHDF5Recorder` writes `raw_eeg.h5` schema v1 before preprocessing. The file
contains `timestamp`, `value_uv`, and `valid` datasets plus `schema_version`,
`sample_rate_hz`, `timebase=run_relative_seconds`, and `value_units=microvolts`
metadata. Missing samples remain absent. `EEGHDF5Replay` reconstructs the same
ordered `EEGSample` values.

Run the hardware-independent default:

```bash
python scripts/run_eeg_pipeline.py
```

For replay, use a partial override such as:

```yaml
source:
  mode: replay
  replay_path: runs/eeg_pipeline/<run>/raw_eeg.h5
```

## Guardian live mode

Live acquisition is an optional dependency:

```bash
python -m pip install -e ".[guardian]"
```

The adapter uses `idun-guardian-sdk==0.1.23` directly (no LSL), subscribes to
`raw_eeg`, and maps each documented Guardian Unix timestamp as
`guardian_unix_timestamp - experiment_start_unix`. The standalone runner captures
the experiment origin after impedance preflight and immediately before recording
when it is not supplied;
a later integrated runner should provide the shared experiment origin.

The default preflight uses the SDK impedance stream and requires the latest
reading to be below the configured 300 kOhm threshold before recording. Realtime
IDUN Quality Score predictions are not required. Live Guardian acquisition has
not been hardware-verified in the implementation environment.
