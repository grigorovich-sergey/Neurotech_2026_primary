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
either diagnostic timestamp dataset means unavailable and replays as `None`.
Finalized live Guardian gaps are recorded as zero-valued samples with `valid=false`;
replay preserves those explicit missing positions. `EEGHDF5Replay` also reads
legacy schema v1 without inventing unavailable timing metadata.

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
disconnect. Connection, fitting diagnostics, and acquisition are separate:

```python
adapter = GuardianAdapter(clock=attempt_clock.now, ...)
adapter.connect()
battery_percent = adapter.check_battery()
adapter.start_impedance(mains_frequency_hz=60)
impedance_ohms = adapter.latest_impedance()  # poll for the fitting display
adapter.stop_impedance()

# Start the shared attempt clock first.
adapter.start(recording_seconds=...)
raw_window = adapter.window(start, end)
stable_samples = adapter.finalize_before(next_window_start)
adapter.stop()
adapter.disconnect()
adapter.close()
```

`prepare()` remains a compatibility convenience that composes connection, battery,
and a finite impedance hard gate. Neither preparation route subscribes to raw EEG.
Impedance must be stopped before `start()`; `start()` subscribes only to raw EEG and
calls the working SDK recording path with `led_sleep=False` and
`calc_latency=False`.

The first raw block establishes the clock mapping. Its final sample is aligned to
the nearest 250 Hz point around that callback's host receipt time, then all vendor
timestamps use the fixed transform:

```text
run_timestamp = anchor_run + guardian_unix_timestamp - anchor_unix
```

One host receipt timestamp is captured per SDK callback and attached to all raw
samples in that callback. Integrated live operation must pass the same deferred
attempt-clock callable used by the other hardware adapters. Vendor timestamps,
not callback order or the SDK `sequence` field, determine sample position. Whole
callbacks may arrive out of timestamp order; timestamps moving backward inside one
raw block and negative mapped timestamps remain hard errors.

The default preflight uses the SDK impedance stream and requires the latest
reading to be below the configured 300 kOhm threshold before recording. Battery is
recorded as a diagnostic, not treated as a configurable scientific threshold.
Realtime IDUN Quality Score predictions are not required.

The preferred consumer path is the mutable 250 Hz timestamp grid. `window(start,
end)` returns the current closed snapshot in timestamp order. Positions whose
packets have not arrived are explicit `EEGSample(value_uv=0.0, valid=False)` values.
A late packet replaces those missing positions in a repeated or later overlapping
request. Each new request start closes all earlier positions; a packet arriving
behind that boundary is discarded and counted by `lost_sample_count` and
`lost_block_count`. There is no fixed reordering delay and no assumption that SDK
sequence numbers are contiguous.

`finalize_before(t)` irreversibly returns the chronological grid strictly before
`t`, including explicit invalid gaps, for append-only HDF5 persistence. Callers must
finalize before advancing past data they need to save. `EEGPipeline.features_from_window()`
evaluates a snapshot without mutating its append-only replay buffer. Asynchronous
SDK and capacity failures are surfaced by `check_health()` and all data methods.

The legacy `drain(cutoff_timestamp=...)` path remains for older standalone callers,
but it is one-way and cannot be mixed with `window()`/`finalize_before()`. Capacity
overflow raises `GuardianQueueOverflowError` instead of silently dropping data; the
default capacity is 15,000 samples (60 seconds at 250 Hz).

`stop()` cancels and awaits the SDK recording task, captures its cloud recording
ID when available. `disconnect()` releases BLE while keeping the owner loop alive;
`close()` performs final cleanup and terminates that loop. The blocking `run()`
method remains as a legacy compatibility wrapper. Live Guardian cancellation,
timestamp units, impedance behavior, and combined MindLink/Guardian timing still
require pilot validation with physical hardware.
