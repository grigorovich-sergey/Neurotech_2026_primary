# Instance 5 — Integration + scientific verification

This layer connects the already-approved gaze interaction, EEG, and experimental
learning subsystems into one pre-smart-glasses experiment. It does not redefine
their algorithms or scientific contracts.

## Quick start

From the repository root after installing the project dependencies:

```bash
python scripts/run_integrated_experiment.py
```

The default is hardware-free: Foundation `VirtualGlasses`, the verification-only
synthetic vision adapter, Instance 3 synthetic EEG, deterministic synthetic
press/timeout feedback, real gaze association/dwell, and real paired River models.
It runs for 12 seconds and produces an inspectable run under `runs/integration/`.

Use a partial YAML override for any non-default run:

```bash
python scripts/run_integrated_experiment.py --config path/to/override.yaml
```

Unknown override keys are rejected by `foundations.config`; every run persists the
fully resolved integration, gaze, EEG, and learning configurations.

## Interfaces and enforced ordering

The integration consumes existing public contracts directly:

- `foundations.contracts.SceneFrame` / `GazeSample` for scene and gaze input;
- `GazeInteractionPipeline.process_scene()` / `process_gaze()` for association,
  candidate episodes and adaptive dwell;
- `EEGPipeline.features(start, cutoff)` for the closed EEG feature window;
- `observation_from_interaction()` and `ExperimentController` for the paired G/E
  prediction, feedback, scoring and learning path.

For each gaze timestamp the runner performs these operations in order:

1. resolve any button press or timeout due by that timestamp;
2. ingest raw EEG only through that timestamp;
3. call `process_gaze(..., intent_score=held_active_score)` using only a score
   frozen on an earlier gaze update for the same episode;
4. route any ended episode and dwell trigger to `ExperimentController`;
5. if an episode ended retrospectively through the gaze-gap grace period, advance
   feedback time again to the current gaze timestamp;
6. build the current-match observation and call `consider_prediction()`;
7. hold its active-model score for the next gaze update and later updates of that
   same episode only.

This makes the N -> N+1 dwell rule mechanical. A direct object/episode identity
switch clears the old score before the new episode's first dwell update. A temporary
no-match inside the same episode may retain the already-frozen score, but missing
gaze never accumulates dwell.

The shadow probability is never passed to dwell. Instance 4 remains authoritative
for strict paired EEG skip, the one-button label truth table, score-before-update,
and common paired learning. No future scene can be selected by the Instance 2
associator, and no EEG after a prediction cutoff is selected by the Instance 3
closed-window path.

## Input and feedback modes

| `input.mode` | Scene/gaze path | Vision path | Typical feedback |
| --- | --- | --- | --- |
| `synthetic` | Foundation `VirtualGlasses` | deterministic verification adapter | `synthetic` |
| `replay` | Foundation scene/gaze HDF5 replay | deterministic verification adapter | `replay` JSONL |
| `video` | Instance 2.5 CFR video + gaze CSV or mouse | real YOLOE + ByteTrack | `keyboard` |

The synthetic vision adapter exists only because Foundation virtual images are
random pixels. During each configured visible interval it supplies one deterministic
tracked full-frame object so that the real association, episode, dwell, EEG and
learning paths are exercised. Prerecorded-video mode uses the actual YOLOE +
ByteTrack adapters instead.

EEG source selection stays in the authoritative EEG configuration. The integrated
path accepts its `source.mode: synthetic` or `source.mode: replay`; Guardian live
acquisition remains an Instance 3/hardware concern.

Feedback options are:

- `synthetic`: cycles `feedback.synthetic.press_cycle`; `false` means no press and
  lets Instance 4 time out, while `true` schedules a press after
  `press_delay_seconds`. The integration never derives the training label itself.
- `replay`: reads `integration_feedback_press` events from an earlier integrated
  `events.jsonl` (or, for compatibility, pressed Instance 4 result records) and
  fails if the recorded episode identity does not match the currently pending one.
- `keyboard`: key code 32 (space) by default. It is intended for paced video with
  the OpenCV window visible; a press is timestamped at the current scientific gaze
  time. This is a pre-hardware stand-in, not a Guardian/glasses button protocol.

## Main configuration options

`configs/integration.yaml` is the complete integration-level default.

| Key | Meaning |
| --- | --- |
| `output_root` | Root for timestamp-named integrated run directories. |
| `participant.id` | Pseudonymous participant identifier. |
| `participant.sequence_index` | Fixed participant sequence index controlling ABAB/BABA parity. |
| `participant.checkpoint_path` | Trusted-local Instance 4 participant checkpoint. |
| `participant.resume_checkpoint` | Load an existing compatible checkpoint when present. |
| `session.id_prefix` | Prefix; the allocated schedule index is appended as `-001`, `-002`, etc. |
| `input.mode` | `synthetic`, `replay`, or `video`. |
| `input.record_glasses` | Persist canonical scene/gaze input as Foundation HDF5. |
| `input.video.*` | Video path, `file`/`mouse` gaze mode, CSV path, pacing, window and diagnostic-frame options. |
| `feedback.mode` | `synthetic`, `replay`, or `keyboard`. |
| `synthetic_vision.*` | Verification-only object warmup/visible/blank timing and label. |
| `analysis.enabled` | Regenerate summary JSON and learning-curve CSV at successful completion. |
| `subsystem_config_overrides.*` | Optional partial YAML overrides for authoritative gaze, EEG and learning defaults. |

`configs/integration_gaze.yaml` and `configs/integration_eeg.yaml` are small default
partial overrides used only to lengthen the hardware-free integrated run. Dwell,
EEG processing/quality thresholds, learning features/models and feedback semantics
remain defined by their authoritative subsystem configurations.

### Replay example

Create a partial gaze override:

```yaml
source:
  recording_path: runs/integration/<old-run>/raw_glasses.h5
  replay_paced: false
```

and a partial EEG override:

```yaml
source:
  mode: replay
  replay_path: runs/integration/<old-run>/raw_eeg.h5
  replay_paced: false
```

Then point an integration override at them and the original feedback log:

```yaml
participant:
  checkpoint_path: runs/participants/replay-check-P001.pkl
input:
  mode: replay
feedback:
  mode: replay
  replay:
    events_path: runs/integration/<old-run>/events.jsonl
subsystem_config_overrides:
  gaze_interaction: path/to/replay_gaze.yaml
  eeg_pipeline: path/to/replay_eeg.yaml
```

Use a fresh checkpoint for a from-scratch reproducibility comparison; using an
existing participant checkpoint intentionally continues that participant's learning.

### Prerecorded video example

```yaml
input:
  mode: video
  video:
    path: path/to/video.mp4
    gaze_mode: file
    gaze_csv_path: path/to/gaze.csv
    paced: true
    show_window: true
feedback:
  mode: keyboard
```

Set `gaze_mode: mouse` and `gaze_csv_path: null` for the mouse-gaze stand-in. Real
YOLOE weights must be available/resolvable by Ultralytics for video mode.

## Outputs

One successful integrated run may contain:

| Artifact | Purpose |
| --- | --- |
| `events.jsonl` | Foundation JSONL stream including Instance 4 predictions/results plus integration episode/dwell/session events. |
| `raw_glasses.h5` | Canonical scene/gaze input when `input.record_glasses: true`. |
| `raw_eeg.h5` | Raw EEG supplied to the integrated pipeline when the resolved EEG config enables recording. |
| `resolved_integration_config.json` | Fully resolved integration settings. |
| `resolved_gaze_interaction_config.json` | Fully resolved authoritative gaze settings. |
| `resolved_eeg_pipeline_config.json` | Fully resolved authoritative EEG settings. |
| `resolved_experiment_learning_config.json` | Fully resolved authoritative learning settings. |
| `analysis_summary.json` | Descriptive paired G/E metrics, skips/reasons, outcomes, latency and controlled-intention agreement where present. |
| `learning_curve.csv` | Per-result cumulative G/E accuracy and F1 for learning/sample-efficiency inspection. |
| `diagnostics/` | Optional rendered frames in video mode. |

The participant checkpoint is stored at `participant.checkpoint_path`, not copied
into each run. It is the existing Instance 4 atomic, trusted-local pickle format.

`analysis_summary.json` treats `common_label` correctly as feedback-derived, not as
independently observed ground truth. False activations and missed intentions are
therefore descriptive against that common feedback label. Selection latency is
episode-start to action outcome for action episodes with a persisted episode start.

## Offline analysis

Analysis is generated only from persisted JSONL scientific records. To regenerate
it later, create a small partial override such as:

```yaml
events_path: runs/integration/<run-id>/events.jsonl
output_directory: runs/integration/<run-id>
```

and run:

```bash
python scripts/analyze_integrated_experiment.py --config path/to/analysis_override.yaml
```

This also writes `resolved_analysis_config.json` as the analysis entry point's
required resolved configuration provenance.

## Checkpoint/restart behavior

The runner allocates the next ABAB/BABA schedule slot and atomically checkpoints it
before processing session data. Each resolved feedback result checkpoints the paired
models through Instance 4.

On graceful source exhaustion, the active candidate is explicitly ended and its
legitimate feedback window is completed by a scheduled press or timeout. On abnormal
interruption, no synthetic label is created for an unresolved episode; the event log
records `integration_session_incomplete`. Restart loads the last compatible participant
checkpoint and starts a new schedule slot/session rather than rewinding sensors and
resuming mid-session. This avoids duplicate scoring/training.

At successful completion, G and E training-count deltas must both equal the number
of resolved session results and the cumulative paired training counts must remain
equal. A mismatch fails loudly instead of guessing recovery state.

## Hardware-only work still deferred

The integrated logic can be verified without smart glasses or Guardian hardware.
Still deferred are the future glasses SDK adapter, vendor gaze calibration/coordinate
conversion, clock synchronization, device latency/dropout tuning, physical feedback
button integration, real Guardian signal-quality threshold validation, and final
YOLOE/ByteTrack performance tuning on the actual scene-camera stream.
