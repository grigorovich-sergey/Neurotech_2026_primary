# NeuroTech 2026 Primary

Research software for testing whether personalized single-channel in-ear EEG
adds useful information to gaze-based object selection. The system detects and
tracks objects, associates gaze with a candidate, compares a gaze-only condition
(`G`) with a gaze-plus-EEG condition (`E`), records contextual feedback, and
updates a participant-specific policy between successful sessions.

The user-facing experiment command is `scripts/run_experiment.py`. The
hardware-free integrated runner is a verification tool, not the live participant
workflow.

## Install

Create a Python 3.10+ environment and install the project from the repository
root:

```bash
python -m pip install -e ".[dev]"
```

Guardian support adds the pinned IDUN SDK:

```bash
python -m pip install -e ".[dev,guardian]"
```

All pip-installable runtime dependencies are declared in `pyproject.toml`.
`.[dev]` adds pytest and `.[guardian]` adds the pinned IDUN SDK. Live MindLink
use additionally requires the vendor-supplied `adhawkapi` installation and the
AdHawk Backend Service; that SDK is not distributed by this project.

The project was developed on an older-generation NVIDIA GPU. That experiment
computer therefore used an older compatible PyTorch/CUDA build, and the project
keeps NumPy below 2.3 for that hardware stack. These are deployment compatibility
constraints rather than requirements of the scientific logic. On another machine,
install the PyTorch/CUDA build appropriate for its GPU and validate the detector;
override `detector.device` for a CPU-only check. The live defaults use `cuda:0`.

## Live workflow

Run practice first. It exercises calibration, scene/gaze capture, detection,
tracking, dwell, display, and optional monitor-only EEG without creating
participant lineage or training data:

```bash
python scripts/run_practice_session.py
python scripts/run_practice_session.py --with-eeg
```

Run one experimental session only after the practice path is stable:

```bash
python scripts/check_feedback_key.py
python scripts/run_experiment.py --subject-id P017 --active-model G
python scripts/run_experiment.py --subject-id P017 --active-model E
```

`--active-model` selects the visible condition; the other condition is derived
as shadow. Raw EEG is required and recorded. Raw glasses HDF5 is disabled by
default for live responsiveness and can be enabled with `--record-glasses`.

## Hardware-independent checks

| Check | Command | Purpose |
| --- | --- | --- |
| Foundations | `python scripts/run_virtual_glasses.py` | Record deterministic scene/gaze samples and verify exact replay. |
| Gaze interaction | `python scripts/run_gaze_interaction.py` | Exercise detector, tracker, association, episodes, and dwell. |
| EEG pipeline | `python scripts/run_eeg_pipeline.py` | Process deterministic synthetic EEG and write a feature summary. |
| Learning | `python scripts/run_experiment_learning.py` | Exercise frozen policies, feedback labels, and between-session training. |
| Integrated verifier | `python scripts/run_integrated_experiment.py` | Run the complete hardware-free gaze/EEG/learning path. |
| Video/gaze harness | `python scripts/run_test_harness.py --config path/to/override.yaml` | Run prerecorded video with mouse or CSV gaze. |

Every runner accepts `--config PATH` unless its CLI is explicitly positional.
Custom YAML files are strict partial overrides of the runner's default: nested
mappings merge recursively, scalar/list values replace defaults, and unknown
keys fail.

## Session reports

Successful live sessions are published under
`runs/subjects/<subject-id>/lineage/`. Two standalone reports use that canonical
lineage:

```bash
python summarize_session.py P017 1
python scripts/report_triggered_events.py P017 1
python scripts/report_triggered_events.py P017 1 --details
python scripts/report_triggered_events.py P017 1 --csv triggered_P017_s001.csv
```

`summarize_session.py` reports training-eligible, scorable G/E outcomes and the
policy transition. `report_triggered_events.py` reports every visible selection,
including selections excluded from training. Their denominators are deliberately
different.

## Configuration map

| File | Used by | Key controls |
| --- | --- | --- |
| `configs/experiment.yaml` | Main live experiment | Subject/model runtime values, duration, queues, feedback key, display, subsystem config paths. |
| `configs/practice_session.yaml` | Live practice | Duration, optional EEG, terminal detail, queues, recording, subsystem overrides. |
| `configs/mindlink.yaml` | MindLink smoke/live paths | Connection timeouts, calibration, frame queue. |
| `configs/gaze_interaction.yaml` | Vision and gaze | YOLOE model/device/threshold/categories, ByteTrack, association, episodes, dwell. |
| `configs/eeg_pipeline.yaml` | EEG processing | Source mode, Guardian connection, 250 Hz signal, quality gate, feature window. |
| `configs/experiment_learning.yaml` | G/E policy and trainer | Timing, cold-start dwell, E floor/reduction, trainer search and minimum examples. |
| `configs/integration.yaml` | Hardware-free verifier | Synthetic/replay/video input, feedback, duration, subsystem overrides. |
| `configs/test_harness.yaml` | Video/gaze harness | Video path, mouse/CSV gaze, pacing, display. |

Resolved main and subsystem configurations are saved with each run.

## Source layout

| Source package | Responsibility |
| --- | --- |
| `src/foundations/` | Shared scene/gaze contracts, strict configuration, clocks, events, virtual input, and HDF5 replay. |
| `src/gaze_interaction/` | YOLOE detection, ByteTrack tracking, gaze association, candidate episodes, dwell, and diagnostics. |
| `src/eeg_pipeline/` | Guardian acquisition, timestamp-grid windows, quality gating, preprocessing, features, and EEG HDF5. |
| `src/experiment_learning/` | Active/shadow assignment, frozen participant policies, feedback records, and deterministic training. |
| `src/integration/` | Cross-subsystem orchestration, feedback routing, analysis, and the hardware-free verifier. |
| `src/mindlink/` | AdHawk MindLink connection, calibration, video, gaze, timestamps, and cleanup. |
| `src/practice_session/` | Non-experimental live hardware practice workflow. |
| `src/test_harness/` | Prerecorded video plus mouse/CSV gaze sources. |

## Documentation

- [Foundations](docs/instance_1_foundations.md)
- [Vision and gaze interaction](docs/instance_2_gaze_interaction.md)
- [Video and gaze harness](docs/instance_2_5_test_harness.md)
- [Guardian and EEG pipeline](docs/instance_3_eeg_pipeline.md)
- [Frozen policies and learning](docs/instance_4_experiment_learning.md)
- [Hardware-free integration verifier](docs/instance_5_integration.md)
- [AdHawk MindLink adapter](docs/instance_6_mindlink.md)
- [Live practice](docs/practice_session.md)
- [Main live experiment](docs/main_experiment_runner.md)

## Tests

```bash
pytest
```

The live experiment tests use injected detector, tracker, MindLink, Guardian,
clock, and input fakes; physical sensor validation remains a separate practice
step.
