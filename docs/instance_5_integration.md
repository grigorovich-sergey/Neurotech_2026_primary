# Instance 5 — Integration and scientific verification

Instance 5 connects the approved gaze, EEG, and frozen-policy learning
subsystems. It owns attempt lifecycle, scientific-time ordering, cross-subsystem
routing, immutable successful-session persistence, and between-session training.
It does not change object detection, EEG calculations, the engagement-index
formula, or feedback labels.

## Quick start

Install the project, then run the deterministic hardware-free path:

```bash
python -m pip install -e ".[dev]"
python scripts/run_integrated_experiment.py
```

The default uses virtual glasses, deterministic synthetic vision, synthetic EEG,
and deterministic press/timeout feedback for 12 scientific seconds. A successful
attempt writes a timestamped directory under `runs/integration/` and advances the
participant by exactly one predetermined schedule row.

Partial YAML overrides are recursively merged with
`configs/integration.yaml`. Unknown keys are rejected, and every run saves all
four resolved configurations.

```bash
python scripts/run_integrated_experiment.py --config path/to/override.yaml
```

## Frozen policy and schedule binding

Integration no longer uses mutable pickle checkpoints, parity allocation, or
online learning. `participant.artifact_directory` contains immutable policies,
completed sessions, and training reports. The next session number is the first
number after the contiguous successful `completed_session_NNN.json` artifacts.

The condition CSV must retain this exact header:

```csv
sequence_id,session_number,active_condition
```

Before an attempt, the runner:

1. hashes and loads the approved condition schedule;
2. resolves `participant.sequence_id` for the next successful session number;
3. loads `policy_session_NNN.json` and verifies participant, session, schedule,
   and policy SHA-256 bindings;
4. builds Instance 2 dwell values from
   `policy.dwell_parameters(active_condition)`;
5. records the exact attempt, schedule, condition, and policy identities.

Session 1 creates the immutable cold-start policy if it is absent. After a
successful session N, the runner persists its completed-session artifact and
calls the deterministic trainer once to create policy N+1. A failed attempt
creates neither a completed-session artifact nor policy N+1, so the next run
retries the same session and condition with the same frozen policy.

## Causal processing order

For every scene or gaze cutoff, Integration calls `drain_through` only through the
latest processed scientific timestamp. Under the PR #20 Guardian source this is a
health/finalization hook, not callback-order ingestion. For a confirmed gaze update
it then:

1. resolves feedback presses/timeouts due by the cutoff;
2. passes the score frozen on update N-1 to Instance 2 update N;
3. routes an ended episode or displayed action to the experiment controller;
4. calls `evaluate_update(...)` for the current confirmed match;
5. holds the returned episode score for update N+1.

That N-to-N+1 hold prevents EEG ending at update N from changing dwell already
accumulated on N. A direct track switch clears the old held score. The action gate
is closed while feedback is pending; newer candidates may remain provisional,
and accepted feedback cancellation instructions are applied through Instance 2's
typed `FEEDBACK_INTERRUPTION` cancellation path.

`evaluate_update` receives an EEG feature source, never a bare `EEGPipeline`.
Each `features(start, end)` request checks health through exactly `end`, closes data
older than the requested start, and evaluates the adapter's current ordered 250 Hz
snapshot. Missing positions are explicit `valid=False` samples. Late packets may
correct a later overlapping request, but a frozen episode decision is never
recalculated and future samples are never exposed. Battery, impedance, queue
overflow, SDK acquisition, and cleanup failures are hard attempt errors, not
missing-EEG fallbacks. Guardian diagnostics are provenance only and never model
features.

## Input and feedback modes

| Mode | Scene/gaze source | Vision | Typical feedback |
| --- | --- | --- | --- |
| `synthetic` | Foundation virtual glasses | deterministic verification adapter | `synthetic` |
| `replay` | recorded glasses HDF5 | deterministic verification adapter | `replay` |
| `video` | CFR video plus gaze CSV or mouse | YOLOE + ByteTrack | `keyboard` |

Synthetic vision exists only because virtual frames are random pixels. Video mode
uses the configured real YOLOE device (including CUDA when available) and the
ByteTrack adapter.

Feedback modes are:

- `synthetic`: deterministic press/no-press cycle; labels are still derived only
  by Instance 4's feedback truth table;
- `replay`: timestamped `integration_feedback_press` events from an earlier log,
  with strict pending-episode identity checks;
- `keyboard`: the configured OpenCV key, normally SPACE, for visible paced video.

Replay and video may run unpaced with synthetic/replayed EEG. Live Guardian input
requires real-time scene/gaze pacing: replay must set `source.replay_paced: true`,
and video must set `input.video.paced: true`. Synthetic glasses are automatically
paced when live Guardian is selected.

## Live Guardian attempts

Install the SDK extra and provide the token without putting it in YAML:

```bash
python -m pip install -e ".[dev,guardian]"
export IDUN_API_TOKEN="..."
```

The Git-ignored `.secrets/idun_api_token` single-line file is the fallback. A
minimal EEG override for the default 12-second attempt is:

```yaml
source:
  mode: live
  guardian:
    recording_seconds: 13
```

`recording_seconds` must exceed `session.maximum_duration_seconds`, leaving a
whole-second timer margin around startup and the final scientific cutoff.

The live lifecycle is intentionally strict:

1. construct `GuardianAdapter(clock=attempt_clock.now, ...)`;
2. after any gaze calibration owned by the hardware entry point, call
   `guardian.connect()` and `guardian.check_battery()`;
3. call `guardian.start_impedance()` with raw EEG still off;
4. poll `guardian.latest_impedance()` and continuously display `None` as waiting,
   otherwise showing both ohms and kOhms, until the operator presses SPACE;
5. stop impedance, enforce the configured impedance limit, and only then start the
   shared attempt clock;
6. construct the raw EEG recorder and `GuardianEEGFeatureSource`, then call
   `guardian.start(...)`;
7. periodically call `eeg_source.drain_through` at the latest processed scientific
   timestamp;
8. on completion or failure, attempt `guardian.stop()`,
   `eeg_source.drain_remaining()`, and `guardian.close()` independently.

`prepare()` is deliberately not used by this live fitting UI because it performs a
finite compatibility preflight. The current Integration verifier has no MindLink
input of its own; the live hardware entry point must complete MindLink calibration
before entering step 2. `run_practice_session.py --with-eeg` exercises that complete
calibration -> fitting -> SPACE order.

`guardian.impedance.duration_seconds` remains in the shared EEG config for the
standalone finite compatibility workflow; the Integration fitting gate ignores it
and continues until SPACE or abort.

All Guardian health checks, ordered-window evaluation, finalized raw HDF5 writes,
and experiment updates occur synchronously on the Integration thread. The SDK
callback only updates the bounded timestamp-grid store. If the operator presses Q
or Esc before SPACE, impedance stops and Guardian closes; the attempt clock, raw
EEG, experiment acquisition, and completed-session persistence never start.

## Configuration reference

| Key | Meaning |
| --- | --- |
| `output_root` | Root for timestamped run directories. |
| `participant.id` | Pseudonymous participant identity. |
| `participant.sequence_id` | Approved condition-schedule sequence, such as `g-first`. |
| `participant.artifact_directory` | Immutable policy/session/report lineage. |
| `session.id_prefix` | Human-readable session ID prefix. |
| `session.maximum_duration_seconds` | Hard attempt deadline. |
| `input.mode` | `synthetic`, `replay`, or `video`. |
| `input.record_glasses` | Save canonical scene/gaze HDF5. |
| `input.video.*` | Video/gaze paths, pacing, display, and diagnostic-frame options. |
| `feedback.mode` | `synthetic`, `replay`, or `keyboard`. |
| `synthetic_vision.*` | Verification-only object timing and label. |
| `analysis.enabled` | Generate JSON summary and learning-curve CSV. |
| `subsystem_config_overrides.*` | Partial gaze, EEG, and learning YAML overrides. |

### Replay example

Gaze override:

```yaml
source:
  recording_path: runs/integration/<old-run>/raw_glasses.h5
  replay_paced: false
```

EEG override:

```yaml
source:
  mode: replay
  replay_path: runs/integration/<old-run>/raw_eeg.h5
  replay_paced: false
```

Integration override:

```yaml
participant:
  id: replay-check-P001
  artifact_directory: runs/participants/replay-check-P001
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

Use a new participant artifact directory for a from-scratch reproducibility
comparison. Reusing one intentionally advances that participant's schedule.

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

Use `gaze_mode: mouse` and `gaze_csv_path: null` for the mouse stand-in.

## Outputs and failure semantics

A successful run may contain:

| Artifact | Purpose |
| --- | --- |
| `events.jsonl` | Attempt, policy decision, episode record, feedback, lifecycle, and provenance events. |
| `raw_glasses.h5` | Canonical scene/gaze input when enabled. |
| `raw_eeg.h5` | Raw EEG recorded before pipeline ingestion when enabled. |
| `completed_session.json` | Run-local copy of the immutable successful trainer input. |
| `resolved_*_config.json` | Fully resolved Integration and subsystem settings. |
| `analysis_summary.json` | Descriptive G/E outcomes, exclusions, feedback labels, and latency. |
| `learning_curve.csv` | Cumulative descriptive G/E accuracy and F1 over eligible records. |
| `diagnostics/` | Optional rendered video frames. |

The participant directory additionally receives
`completed_session_NNN.json`, `policy_session_NNN.json`, and
`policy_session_NNN.training_report.json`.

Abnormal processing or hardware failure emits `integration_session_incomplete`
after cleanup attempts and re-raises the original error. Incomplete attempts are
never trainer inputs. There is no mid-attempt sensor rewind or online model state
to resume. Completion and incomplete lifecycle events include Guardian expired
sample/block counts so late data that missed the retained correction horizon remain
visible in provenance.

## Offline analysis

Analysis reads only persisted `experiment_policy_decision` and
`experiment_episode_training_record` events. To regenerate:

```bash
python scripts/analyze_integrated_experiment.py --config path/to/analysis_override.yaml
```

`common_label` remains feedback-derived rather than independently observed truth;
reported false activations and missed intentions are descriptive against that
label.

## Hardware validation still required

The deterministic and fake-Guardian paths are covered by tests. Real Guardian
battery/impedance behavior, sustained timestamp-grid capacity, late/expired packet
rates, device latency and clock alignment, physical feedback hardware,
smart-glasses SDK ingestion, and final YOLOE/ByteTrack performance still require
pilot validation on the target system.
