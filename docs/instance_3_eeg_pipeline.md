# Guardian and EEG pipeline

`eeg_pipeline` converts timestamped single-channel EEG into causal quality-checked
feature windows. It supports deterministic synthetic input, HDF5 replay, and live
IDUN Guardian acquisition. Hardware diagnostics such as battery and impedance are
recorded separately and never used as model features.

## Quick checks

The default command needs no hardware:

```bash
python scripts/run_eeg_pipeline.py
```

It writes `raw_eeg.h5`, `resolved_config.json`, and `feature_summary.json` below
`runs/eeg_pipeline/`. To verify replay, use the recording from that run:

```yaml
source:
  mode: replay
  replay_path: runs/eeg_pipeline/<run-id>/raw_eeg.h5
  replay_paced: false
```

```bash
python scripts/run_eeg_pipeline.py --config path/to/replay_eeg.yaml
```

For a live acquisition check, install the Guardian extra, provide the token, and
set `source.mode: live` in a partial override:

```bash
python -m pip install -e ".[dev,guardian]"
export IDUN_API_TOKEN="..."
python scripts/run_eeg_pipeline.py --config path/to/live_eeg.yaml
```

The token may instead be stored as one line in `.secrets/idun_api_token`; the
environment variable takes precedence. Tokens are not written to resolved config
or output artifacts.

## Processing behavior

The pipeline defaults to 250 Hz. The live adapter maps Guardian callback blocks to
that timestamp grid in a bounded store, reports packet loss/overflow, and exposes
samples in timestamp order. Feature requests include only data through the
requested end time, so future samples cannot affect an earlier decision. Finalized
samples can be recorded to HDF5 and replayed without changing their scientific
timestamps.

Usable windows pass duration, coverage, gap, variation, and peak-to-peak checks.
Accepted data is band-pass filtered from 1–40 Hz, then summarized by standard
deviation, peak-to-peak range, mean absolute difference, and Welch power in the
delta, theta, alpha, beta, and low-gamma bands. Rejected or incomplete windows
carry explicit quality state and reasons instead of fabricated feature values.

## Configuration and CLI

The default is `configs/eeg_pipeline.yaml`. The most relevant controls are:

- `source.mode`: `synthetic`, `replay`, or `live`;
- `source.guardian.recording_seconds`, queue capacity, and impedance limit;
- `window.start_seconds` / `window.end_seconds` for the standalone report;
- the 250 Hz signal rate and 1–40 Hz preprocessing band;
- `quality.*` acceptance thresholds and `recording.enabled`.

The only CLI option is `--config PATH`. Live practice and the main experiment use
`configs/practice_eeg.yaml` as a strict partial override for their fitting/runtime
values.

## Source files

| File | Responsibility |
| --- | --- |
| `src/eeg_pipeline/__init__.py` | Package boundary. |
| `src/eeg_pipeline/contracts.py` | EEG samples, window completeness, quality state, and feature records. |
| `src/eeg_pipeline/buffer.py` | Ordered bounded sample history and interval selection. |
| `src/eeg_pipeline/processing.py` | Filtering, quality gates, and Welch feature extraction. |
| `src/eeg_pipeline/pipeline.py` | Sample ingestion and feature-window orchestration. |
| `src/eeg_pipeline/credentials.py` | Guardian token resolution and file-permission checks. |
| `src/eeg_pipeline/guardian.py` | IDUN connection, diagnostics, acquisition, timestamp grid, and cleanup. |
| `src/eeg_pipeline/recording.py` | Raw EEG HDF5 writer and replay source. |
| `src/eeg_pipeline/synthetic.py` | Deterministic signals, gaps, and invalid intervals. |

## Focused tests

```bash
pytest tests/test_eeg_pipeline.py tests/test_guardian_credentials.py
```
