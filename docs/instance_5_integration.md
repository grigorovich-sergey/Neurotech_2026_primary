# Hardware-free integration verifier

`scripts/run_integrated_experiment.py` verifies the combined foundations,
gaze-interaction, EEG, and frozen-policy learning contracts without participant
hardware. It is the supported synthetic/replay/video integration check;
`scripts/run_experiment.py` is the authoritative live experiment entry point.

## Quick check

```bash
python scripts/run_integrated_experiment.py
python scripts/run_integrated_experiment.py --config path/to/override.yaml
```

The default runs 12 deterministic scientific seconds using virtual glasses,
synthetic vision, synthetic EEG, and alternating synthetic feedback. A successful
run writes an attempt below `runs/integration/` and advances the configured
participant lineage below `runs/participants/`.

Use a new `participant.id` and `participant.artifact_directory` when repeating a
from-scratch check; reusing them intentionally advances to the next scheduled
session.

## Integrated behavior

The verifier resolves the next successful session from contiguous completed
artifacts, loads the corresponding frozen policy, and binds the active condition
from `configs/experiment_condition_schedule.csv`. Scene and gaze events are
processed in scientific-time order. EEG feature requests are causal through the
current cutoff, and the EEG decision produced at gaze update N can affect dwell
only from update N+1.

The action gate remains closed while feedback for an earlier episode is pending.
Episode endings, visible actions, feedback resolutions, cancellation instructions,
and trainer records pass through typed subsystem boundaries. Successful completion
writes the completed session and trains policy N+1; an incomplete run does not
advance the lineage.

Input modes are `synthetic`, HDF5 `replay`, and prerecorded `video`. Video uses
YOLOE/ByteTrack with mouse or CSV gaze; replay consumes a prior glasses recording.
Feedback may be synthetic, replayed from events, or read from the display keyboard.

## Configuration and CLI

`configs/integration.yaml` controls participant/lineage identity, the 12 s session
cap, input and feedback modes, raw-glasses recording, analysis, and strict subsystem
overrides. Its defaults use `configs/integration_gaze.yaml` and
`configs/integration_eeg.yaml`. The only CLI option is `--config PATH`.

A minimal prerecorded-video practice override is:

```yaml
participant:
  id: video-check-P001
  artifact_directory: runs/participants/video-check-P001
input:
  mode: video
  video:
    path: path/to/video.mp4
    gaze_mode: mouse
    paced: true
    show_window: true
feedback:
  mode: keyboard
```

## Source files

| File | Responsibility |
| --- | --- |
| `src/integration/__init__.py` | Package boundary. |
| `src/integration/orchestrator.py` | Cross-subsystem episode, action, and feedback routing. |
| `src/integration/vision.py` | Deterministic synthetic vision used with virtual frames. |
| `src/integration/live_input.py` | Keyboard, replay, and synthetic feedback drivers. |
| `src/integration/workflow.py` | Supported synthetic/replay/video verifier and lineage orchestration. |
| `src/integration/analysis.py` | Attempt and participant JSON/CSV summaries from persisted records. |

The verifier also uses the source packages described in Instances 1–4.

## Outputs and reports

Runs contain resolved configs, `events.jsonl`, optional raw sensor HDF5,
`completed_session.json`, `analysis_summary.json`, and `learning_curve.csv`.
Participant artifacts include numbered completed sessions, policies, and training
reports.

To regenerate analysis from an existing event log, set `events_path` and optional
`output_directory` in a strict override of `configs/integration_analysis.yaml`:

```bash
python scripts/analyze_integrated_experiment.py --config path/to/analysis.yaml
```

```bash
pytest tests/test_integration.py
```
