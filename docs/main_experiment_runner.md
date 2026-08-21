# Main live experiment runner

`scripts/run_experiment.py` is the authoritative user-facing entry point for one
live participant session. It combines MindLink calibration/capture, Guardian EEG,
YOLOE/ByteTrack gaze interaction, a frozen G/E policy, contextual feedback,
between-session training, and attempt/participant reports.

## Preflight and invocation

Run the live practice workflow first, and measure the presenter key on the same
computer/backend used for the experiment:

```bash
python scripts/run_practice_session.py --with-eeg
python scripts/check_feedback_key.py
```

Then run exactly one selected active condition:

```bash
python scripts/run_experiment.py --subject-id P017 --active-model G
python scripts/run_experiment.py --subject-id P017 --active-model E
```

The active model controls visible actions; the other model is always evaluated as
shadow. A started session records the CLI assignment, and retries must use the same
active model for that session number.

## CLI options

| Option | Meaning |
| --- | --- |
| `--subject-id ID` | Required pseudonymous ID used for attempts and policy lineage. |
| `--active-model G|E` | Required visible condition; the other condition is shadow. |
| `--config PATH` | Strict partial override of `configs/experiment.yaml`. |
| `--record-glasses` | Opt in to raw scene/gaze HDF5; off by default for live responsiveness. |

Raw EEG is required and recorded. Install the Guardian extra and vendor AdHawk SDK;
the default detector expects CUDA device `cuda:0`.

## Setup and live processing

The operator sequence is MindLink connect/calibrate, Guardian connect/battery,
continuous impedance fitting, then SPACE. SPACE establishes the shared attempt
clock, persists the assignment, starts raw EEG, creates a fresh MindLink receiver,
and opens acquisition/display. Q, Esc, or Ctrl-C before SPACE aborts without a
completed session.

Callbacks feed a latest-scene queue and bounded gaze queue. Each UI loop processes
at most one scene and one small timestamp-sorted gaze batch, dropping and logging
overflow or late samples instead of allowing live latency to grow. EEG drains only
through processed scientific time. A policy decision is frozen once per episode;
the active condition drives dwell and the shadow result is retained for comparison.

A visible selection opens contextual feedback from its presentation time. For a
visible action, the presenter key rejects it and silence accepts it; for an eligible
no-action outcome, the key marks a missed desired action and silence accepts the
non-action. Feedback can cancel newer provisional candidates. Q/Esc after start
ends the attempt without committing a successful session.

## Successful completion and recovery

Reaching the configured duration is the normal successful endpoint. After sensor
cleanup, the runner creates the attempt-local completed session and analysis, then
stages and validates policy N+1 and its training report. It also checks any existing
next-policy or report copy for a digest conflict before advancing.

The numbered `completed_session_NNN.json` in the subject lineage is the session
commit marker. Only after that immutable write does the runner publish the training
report, policy, participant analysis, completion event, and terminal notice. These
post-commit outputs are deterministic or descriptive; failures are recorded in
`attempt_summary.json` as `post_commit_warnings`, while the completed session stays
successful. The next invocation reconstructs a missing report/policy from the
committed inputs before opening hardware.

If staging, cleanup, acquisition, or scientific finalization fails before the
commit marker, the lineage does not advance. The next run retries the same session,
assignment, and frozen policy. Too few training examples still produce a valid
carried-forward policy with an explicit training status.

## Configuration

`configs/experiment.yaml` supplies the 300 s duration, queue sizes, display,
presenter/stop keys, recording flags, analysis, and subsystem config paths. Important
live values come from `configs/practice_gaze.yaml` (including the 30 Hz tracker) and
`configs/practice_eeg.yaml` (including a 3 MOhm fitting limit and extended Guardian
recording time). Resolved main, MindLink, gaze, EEG, and learning configurations are
saved in every attempt.

## Source files

| File | Responsibility |
| --- | --- |
| `scripts/run_experiment.py` | Authoritative setup, live cadence, recordings, finalization, lineage commit, and CLI. |
| `src/integration/orchestrator.py` | Active/shadow episode, action, and feedback routing. |
| `src/integration/live_input.py` | Contextual feedback timing and key handling. |
| `src/integration/analysis.py` | Attempt and participant JSON/CSV reports. |
| `src/experiment_learning/assignment.py` | Immutable CLI assignment and retry validation. |
| `src/experiment_learning/state_machine.py` | Frozen decisions, feedback labels, and completed records. |
| `src/experiment_learning/trainer.py` | Deterministic policy N+1 and training report. |
| `src/experiment_learning/guardian_source.py` | Causal live EEG feature source. |
| `src/mindlink/adapter.py` | MindLink connection, calibration, capture, and timing. |
| `src/gaze_interaction/pipeline.py` | Scene/gaze, episode, dwell, and trigger processing. |

## Artifacts and reports

```text
runs/subjects/<subject-id>/
  attempts/<attempt-id>/
    events.jsonl
    raw_eeg.h5
    raw_glasses.h5                    # only with --record-glasses
    completed_session.json            # successful staging only
    staged_policy_session_NNN.json
    staged_policy_session_NNN.training_report.json
    analysis_summary.json
    learning_curve.csv
    participant_analysis_summary.json
    participant_learning_curve.csv
    resolved_*_config.json
    attempt_summary.json
  lineage/
    assignment_session_NNN.json
    completed_session_NNN.json        # successful-session commit marker
    policy_session_NNN.json
    policy_session_NNN.training_report.json
    participant_analysis_summary.json
    participant_learning_curve.csv
```

Two standalone scripts read the committed lineage. The training-focused summary
counts eligible/scorable records; the triggered-event report includes every visible
selection, so their denominators intentionally differ.

```bash
python summarize_session.py P017 1
python scripts/report_triggered_events.py P017 1 --details
```

## Focused tests

Tests inject all hardware and input dependencies while exercising the actual live
entry-point workflow:

```bash
pytest tests/test_live_experiment.py
```
