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

## Integrated pre-hardware experiment

Run the complete gaze interaction + EEG + paired-learning path with deterministic
synthetic inputs:

```bash
python scripts/run_integrated_experiment.py
```

The integrated runner also supports Foundation HDF5 replay and prerecorded video
with CSV or mouse gaze. See [docs/instance_5_integration.md](docs/instance_5_integration.md)
for input/feedback modes, configuration options, subsystem interfaces and ordering,
outputs, offline analysis, and restart behavior.

## Tests

```bash
pytest
```
