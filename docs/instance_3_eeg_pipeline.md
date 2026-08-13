# Instance 3 — Guardian + EEG pipeline

`eeg_pipeline` turns ordered single-channel raw EEG into timestamp-addressable,
quality-aware generic features. It is deliberately independent of gaze episodes,
feedback labels, and the learning models.

## Time and window contract

`EEGSample(timestamp, value_uv, valid, vendor_timestamp_unix,
host_receipt_timestamp)` uses non-negative run-relative seconds and microvolts.
The final two optional fields retain live-acquisition diagnostics; they are `None`
for sources that do not provide them. `EEGBuffer.window(start, end)` uses a
**closed `[start, end]` interval**;
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

`EEGHDF5Recorder` writes `raw_eeg.h5` schema v2 before preprocessing. The file
contains `timestamp`, `value_uv`, `valid`, `vendor_timestamp_unix`, and
`host_receipt_timestamp` datasets plus `schema_version`, `sample_rate_hz`,
`timebase=run_relative_seconds`, and `value_units=microvolts` metadata. A NaN in
either diagnostic timestamp dataset means unavailable and replays as `None`;
missing samples remain absent. `EEGHDF5Replay` reads both schema v2 and legacy
schema v1, reconstructing ordered `EEGSample` values without inventing legacy
timing metadata.

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

The API token is resolved in this order:

1. the configured environment variable (`IDUN_API_TOKEN` by default);
2. the configured ignored file (`.secrets/idun_api_token` by default).

The file must contain exactly one non-empty line. `/.secrets/` is ignored by Git,
and POSIX token files with any group/other permissions are rejected. The token is
passed only to the SDK client; it is never written to resolved configuration,
events, summaries, or terminal output.

The adapter uses `idun-guardian-sdk==0.1.23` directly (no LSL). One persistent
worker thread and asyncio loop own the SDK client from Bluetooth connection through
disconnect. The explicit lifecycle is:

```python
adapter = GuardianAdapter(clock=attempt_clock.now, ...)
preflight = adapter.prepare(...)
# Start the shared attempt clock first.
adapter.start(recording_seconds=...)
samples = adapter.drain(cutoff_timestamp=cutoff)
adapter.stop()
remaining = adapter.drain()
adapter.close()
```

`prepare()` connects and reads battery plus the optional impedance hard gate but
does not subscribe to or record raw EEG. `start()` subscribes to `raw_eeg` and
captures a paired clock anchor only after the integration-owned attempt clock has
started:

```text
run_timestamp = anchor_run + guardian_unix_timestamp - anchor_unix
```

One host receipt timestamp is captured per SDK callback and attached to all raw
samples in that callback. The standalone runner creates its own monotonic clock at
recording start. Integrated live operation must instead pass the same deferred
attempt-clock callable used by the other hardware adapters. Vendor timestamps
remain the source of sensor sample time; callback order is not substituted for
them. Backward or negative mapped timestamps remain hard errors.

The default preflight uses the SDK impedance stream and requires the latest
reading to be below the configured 300 kOhm threshold before recording. Battery is
recorded as a diagnostic, not treated as a configurable scientific threshold.
Realtime IDUN Quality Score predictions are not required.

SDK callbacks append canonical `EEGSample` values to a bounded handoff queue.
`drain(cutoff_timestamp=t)` returns only samples at or before the closed cutoff and
leaves later samples queued. Instance 4 must drain immediately before each
EEG-dependent prediction and ingest those samples into `EEGPipeline` on the
integration thread. The SDK worker never writes HDF5 or mutates the pipeline.
Queue overflow raises `GuardianQueueOverflowError` and stops acquisition instead
of silently dropping scientifically relevant samples. The default capacity is
15,000 samples (60 seconds at 250 Hz).

`stop()` cancels and awaits the SDK recording task, captures its cloud recording
ID when available, and `close()` disconnects on the same SDK loop. The blocking
`run()` method remains as a standalone compatibility wrapper. Live Guardian
cancellation, timestamp units, and combined MindLink/Guardian timing still require
pilot validation with physical hardware.
