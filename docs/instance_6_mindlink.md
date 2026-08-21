# AdHawk MindLink adapter

`mindlink` converts AdHawk MindLink scene video and `GAZE_IN_IMAGE` packets into
canonical `SceneFrame` and `GazeSample` values. It owns connection, tracker-ready
state, calibration, capture timestamps, bounded frame buffering, disconnect state,
and cleanup. It performs no detection, dwell, EEG, or learning logic.

## Hardware smoke test

Start the AdHawk Backend Service and ensure the vendor `adhawkapi` package is
available, then run:

```bash
python scripts/run_mindlink.py
python scripts/run_mindlink.py --config path/to/override.yaml
```

The runner connects, waits for tracker readiness, performs quick-start calibration,
waits for SPACE, then creates a fresh video receiver and starts scene/gaze capture.
The fresh receiver after calibration is required by the tested hardware lifecycle.
The display overlays valid gaze; Q or Esc stops the run. Resolved config is written
below `runs/mindlink/`.

## Data and timing

Pixel gaze is normalized as `x/(width-1)` and `y/(height-1)` with a top-left
origin. Missing, non-finite, or out-of-range gaze is invalid rather than clipped.
Confidence remains unavailable because the meaning of the remaining vendor values
is not used by the experiment.

Gaze vendor time and usable frame datetimes are anchored to the shared
run-relative clock. Host receipt and raw vendor values remain diagnostics.
Backward timestamps are dropped. The bounded frame queue drops its oldest item
when full to protect live responsiveness, and disconnect state stops capture; an
experiment attempt does not reconnect automatically.

## Configuration and CLI

The default `configs/mindlink.yaml` exposes only a few operational controls:
connection timeout, tracker-ready timeout, 35 mm calibration marker, calibration
timeout, returning-user flag, and frame queue size. The CLI exposes only
`--config PATH`.

## Source files

| File | Responsibility |
| --- | --- |
| `src/mindlink/__init__.py` | Public adapter and metadata exports. |
| `src/mindlink/adapter.py` | SDK lifecycle, calibration, video/gaze conversion, timing, buffering, and cleanup. |

`scripts/run_mindlink.py` contains the hardware smoke UI and reusable adapter,
calibration, and capture helpers used by live workflows.

## Focused tests

The tests inject a fake AdHawk API, so they run without glasses:

```bash
pytest tests/test_mindlink.py
```
