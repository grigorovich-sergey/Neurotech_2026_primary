# Main live experiment runner

`scripts/run_experiment.py` runs one live participant-specific experimental
session. It combines MindLink calibration/capture, Guardian fitting and raw EEG,
gaze interaction, the frozen G/E policy, contextual feedback, optional visual
replay artifacts, between-session training, and participant metrics.

This practice-style runner is the authoritative live path because it has been
tested and troubleshot on the experiment hardware. `integration.live_workflow`
retains the older strict global-timestamp-merger implementation as a reference
path; it is not the live acquisition path.

## Invocation

```bash
python scripts/run_experiment.py --subject-id P017 --active-model G
python scripts/run_experiment.py --subject-id P017 --active-model E
```

`--subject-id` selects both the recording directory and participant policy
lineage. `--active-model` accepts `G` or `E`; the other model is always the
shadow model. There is deliberately no `--shadow-model` or verbose-decision
argument. The terminal remains concise while `events.jsonl` retains detailed
episode, policy, trigger, feedback, and failure provenance.

The live path requires the vendor AdHawk SDK plus the Guardian extra
(`python -m pip install -e ".[dev,guardian]"`). The configured YOLOE device is
`cuda:0`; CUDA-enabled PyTorch remains an environment-specific installation.

The default `configs/experiment.yaml` is complete through explicit paths to the
authoritative MindLink, gaze, EEG, and learning configurations. `--config`
accepts a strict partial override. Fully resolved main and subsystem
configurations are saved in every attempt directory.

## Condition-selection provenance

Prototype sessions use CLI selection. At experimental start an immutable
`assignment_session_NNN.json` records the `cli` selection source, active and
derived shadow conditions, participant/session identity, and stable CLI-mode
binding.

The CSV loader remains available to the schedule-backed Integration verifier and
continues to require the exact header:

```csv
sequence_id,session_number,active_condition
```

Verifier events record `model_selection_source: csv`. The live runner keeps its
CSV configuration inactive for the first prototype round.

## Required setup and start order

1. Connect and calibrate MindLink. Video/gaze capture remains off.
2. Construct and connect Guardian with the deferred attempt clock.
3. Check battery and start continuous impedance.
4. Display battery and newest impedance until the operator presses SPACE.
5. Stop impedance and validate the fitting reading.
6. Start the shared attempt clock.
7. Persist the assignment, construct raw recorders and
   `GuardianEEGFeatureSource`, then start Guardian recording.
8. Create the display and start a fresh MindLink capture.

Q, Esc, or Ctrl-C before SPACE aborts setup without an assignment, experiment
recording, or completed session. Guardian `stop()`, EEG `drain_remaining()`, and
Guardian `close()` are independent cleanup operations.

The 3 MOhm impedance limit in `configs/practice_eeg.yaml` is a temporary live-lab
fitting override. The authoritative EEG-pipeline default remains 300 kOhm.

## Live ordering and visible actions

MindLink callbacks enqueue canonical values into a latest-scene queue and a
bounded latest-gaze queue. Raw glasses HDF5 is disabled by default because it
reduces live FPS below a usable level; pass `--record-glasses` only when visual
replay provenance is specifically required. Finalized EEG is persisted
asynchronously in HDF5 batches.

- Scene and gaze queue overflow drops the oldest queued value and records the drop
  explicitly in `events.jsonl`; this is an intentional responsiveness tradeoff.
- Each live loop consumes at most one scene and up to
  `processing.gaze_batch_size` gaze samples, sorting only that small batch by
  timestamp. There is no global timestamp merger on the live path.
- Samples that arrive behind already-processed scientific time are dropped and
  logged rather than moving the experiment clock backward.
- EEG drains only through the latest processed scientific timestamp. Exact
  feature requests retain PR #20's causal Guardian-window behavior.

The dwell crossing and visible presentation are separate events. A trigger records
its threshold-crossing timestamp. The selection is rendered/reported, the actual
presentation timestamp is recorded, and the feedback window opens from
presentation time.

The diagnostic OpenCV window is operator-only and may show active/shadow
condition, intent score, decision reason, queue/drop counters, and EEG state.
Participants cannot see it.

The HUD laser presenter Down/PageDown button is the contextual feedback input
after start. OpenCV extended-key codes are backend-specific, so run
`python scripts/check_feedback_key.py` on the experiment computer with the actual
presenter, then copy the measured value into `feedback.key_code`. A short
`FEEDBACK BUTTON PRESSED` marker appears on the operator overlay whenever the
configured feedback key is recognized. Silence and presses retain the approved
one-button truth table. Q or Esc ends an attempt without creating a
completed-session trainer input.

## Completion and retry rules

Experiments are sequential; no concurrent participant-session machinery is used.
Reaching `session.maximum_duration_seconds` (300 seconds by default) is the normal
successful endpoint. The active episode is canceled at the deadline and an
already-open feedback window receives its remaining real grace period.

After clean hardware/recording cleanup, the runner creates the completed-session
artifact, stages and validates policy N+1, publishes the completed session and
next policy to the subject lineage, and regenerates participant metrics.

A setup abort, operator termination, hardware/acquisition error, cleanup failure,
or scientific finalization error does not create a completed-session artifact and
does not advance training. An attempt started after SPACE retains its assignment.
Its retry uses the same session number, policy, and active condition; passing a
different `--active-model` fails before hardware setup.

Insufficient eligible examples still count as a successful session. The trainer
writes a carried-forward next policy and records that status.

## Files

```text
runs/subjects/<subject-id>/
  attempts/<attempt-id>/
    events.jsonl
    raw_glasses.h5                       # only with --record-glasses
    raw_eeg.h5
    completed_session.json              # successful attempts only
    analysis_summary.json
    learning_curve.csv
    participant_analysis_summary.json
    participant_learning_curve.csv
    resolved_*_config.json
    attempt_summary.json
  lineage/
    assignment_session_NNN.json
    completed_session_NNN.json
    policy_session_NNN.json
    policy_session_NNN.training_report.json
    participant_analysis_summary.json
    participant_learning_curve.csv
```

The attempt directory is always unique. The lineage contains successful sessions
plus initiated-session assignments.

## Controlled-intention placeholder

`controlled_intention` keys exist in the default configuration, but no trial
scheduler, cue UI, or training integration is implemented. `enabled: true` fails
validation clearly so the runner cannot infer an unapproved protocol.
