# Foundations

`foundations` defines the shared contracts and utilities used by every workflow:
strict configuration resolution, run-relative clocks, structured events,
canonical scene/gaze values, deterministic virtual input, and exact-order HDF5
record/replay. It contains no detection, EEG, learning, or hardware SDK logic.

## Quick check

Run a deterministic one-second recording:

```bash
python scripts/run_virtual_glasses.py
```

The run directory under `runs/virtual_glasses/` contains `resolved_config.json`,
`events.jsonl`, and `recording.h5`. Replay it with a partial override:

```yaml
mode: replay
recording_path: runs/virtual_glasses/<record-run>/recording.h5
replay_paced: false
```

```bash
python scripts/run_virtual_glasses.py --config path/to/replay.yaml
```

## Configuration and CLI

The default is `configs/virtual_glasses.yaml`. Important values are `mode`,
`duration_seconds`, scene/gaze rates, dropout/invalid probabilities, and
`recording_path`. The only CLI option is `--config PATH`.

Scientific timestamps are non-negative float seconds relative to source or
attempt start. UTC directory names identify runs but are not scientific time.
Scene images are RGB `uint8`; gaze coordinates are normalized to `[0, 1]` with
top-left origin. Invalid gaze carries no coordinates.

## Source files

| File | Responsibility |
| --- | --- |
| `src/foundations/__init__.py` | Package boundary. |
| `src/foundations/config.py` | YAML loading, recursive strict overrides, resolved JSON output. |
| `src/foundations/contracts.py` | Canonical `SceneFrame` and `GazeSample` validation. |
| `src/foundations/events.py` | JSON Lines event records and logger. |
| `src/foundations/fixtures.py` | Small deterministic scene/gaze fixtures for checks and tests. |
| `src/foundations/operator_gate.py` | Refreshable SPACE/Q/Esc operator gate. |
| `src/foundations/recording.py` | Canonical scene/gaze HDF5 writer and replay. |
| `src/foundations/timebase.py` | Monotonic run-relative clock. |
| `src/foundations/virtual_glasses.py` | Deterministic interleaved scene/gaze generator. |
| `src/foundations/workflow.py` | Virtual recording/replay run orchestration. |

## Contract checks

Focused checks cover contracts, strict config, events, virtual input, and replay:

```bash
pytest tests/test_contracts.py tests/test_config.py tests/test_events.py tests/test_virtual_glasses.py tests/test_replay.py
```
