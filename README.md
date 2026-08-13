# NeuroTech 2026 Primary

Research code for testing whether personalized single-channel in-ear EEG adds
useful information to an adaptive gaze-selection system.

This initial foundation is hardware-independent. It defines scene/gaze
contracts, run-relative timing, YAML configuration, structured events, a
deterministic virtual-glasses source, and HDF5 recording/replay.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Virtual glasses

The default run is a small deterministic recording:

```bash
python scripts/run_virtual_glasses.py
```

Every run writes its fully resolved configuration to `resolved_config.json`.
Recording runs also write `recording.h5`; all runs write `events.jsonl`.

To change parameters, provide a partial YAML file. Only keys already present in
the default configuration are accepted.

```bash
python scripts/run_virtual_glasses.py --config path/to/override.yaml
```

For replay, an override can be as small as:

```yaml
mode: replay
recording_path: runs/virtual_glasses/<run-id>/recording.h5
```

Scientific timestamps are float seconds relative to experiment/source start.
The UTC run-directory name is only an output identifier and is not the
scientific timebase.

## Integrated experiment

Run the hardware-free deterministic path with virtual glasses, synthetic EEG,
the frozen session policy, scheduled feedback, immutable session artifacts, and
between-session training:

```bash
python scripts/run_integrated_experiment.py
```

The runner also supports replay, prerecorded-video/YOLOE input, and live Guardian
EEG. Live Guardian mode performs battery/impedance preflight with raw EEG off,
waits for `SPACE`, then starts the shared attempt clock before acquisition. See
[docs/instance_5_integration.md](docs/instance_5_integration.md) for configuration,
artifact lineage, cleanup guarantees, and retry behavior.

## Live hardware practice

Run the MindLink calibration and diagnostic display without creating an
experimental session or training input:

```bash
python scripts/run_practice_session.py
python scripts/run_practice_session.py --with-eeg
python scripts/run_practice_session.py --concise-decisions
```

Guardian EEG is optional and monitor-only; it never changes practice dwell.
For Guardian, place the single-line API token in the Git-ignored
`.secrets/idun_api_token` file or set `IDUN_API_TOKEN`. Complete MindLink
calibration and Guardian battery/impedance preflight first, then press `SPACE` to
create a fresh video receiver and start the shared attempt clock, gaze/video
streams, display, and raw EEG. Nothing is recorded during the post-calibration
wait. Press `Q` or `Esc` to stop after starting. Concise decision reporting is the
default; `--verbose-decisions` adds candidate/dwell transition lines. See
[docs/practice_session.md](docs/practice_session.md) for setup, displayed
diagnostics, artifacts, and current hardware-validation limitations.

## Tests

```bash
pytest
```
