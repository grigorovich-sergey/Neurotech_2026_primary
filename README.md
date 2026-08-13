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

## Integrated pre-hardware experiment — temporarily unavailable

The checked-in integrated runner still targets the superseded online-learning API
and currently fails during import. Do not use it for experiments until the
schedule/frozen-policy/session integration rewrite is complete. The command below
is retained only to identify the affected entry point:

```bash
python scripts/run_integrated_experiment.py
```

The existing [Instance 5 integration document](docs/instance_5_integration.md)
describes that obsolete implementation and is not a current experimental runbook.
Subsystem runners and the live hardware practice mode below remain available.

## Live hardware practice

Run the MindLink calibration and diagnostic display without creating an
experimental session or training input:

```bash
python scripts/run_practice_session.py
python scripts/run_practice_session.py --with-eeg
python scripts/run_practice_session.py --concise-decisions
```

Guardian EEG is optional and monitor-only; it never changes practice dwell.
Complete calibration first, then press `SPACE` in the terminal to create a fresh
video receiver and start the attempt clock, gaze/video streams, display, and
optional EEG. Nothing is acquired during the post-calibration wait. Press `Q` or
`Esc` to stop after starting. Verbose decision reporting is the default;
`--concise-decisions` retains lifecycle, selection, warning, error, and stop output
while suppressing candidate/dwell transition lines. See
[docs/practice_session.md](docs/practice_session.md) for setup, displayed
diagnostics, artifacts, and current hardware-validation limitations.

## Tests

```bash
pytest
```
