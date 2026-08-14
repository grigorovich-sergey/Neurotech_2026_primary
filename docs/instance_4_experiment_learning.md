# Instance 4: frozen session policies and between-session learning

`experiment_learning` owns the participant-specific G-versus-E experimental
policy, feedback attribution, scientific episode records, predetermined condition
schedule, and deterministic training between sessions.

This is the corrected design. It replaces the earlier River online learners,
mutable pickle checkpoint, and parity-allocated schedule. Nothing learns or
changes policy parameters during a session.

## Experimental distinction

Both conditions use one participant-specific gaze dwell threshold, `T_G`.

- **G** uses `T_G` unchanged.
- **E** may shorten `T_G` from one frozen EEG decision for the episode.
- EEG never identifies the object, creates an action, lengthens dwell, or lowers
  dwell below the policy's `minimum_e_threshold_s`.
- Missing/invalid EEG falls back behaviorally to `T_G` in both conditions and
  excludes the episode from paired training.

The session policy is loaded from one immutable JSON artifact. The active and
shadow outcomes are reconstructed from the same recorded gaze trajectory after
the feedback label is known. Training occurs once, after a successful session,
and writes the next session's policy.

## EEG enters the model as one variable

Instance 3 still returns all eight features in its fixed order, and every episode
record retains them. The current regression input is only this engagement index:

```text
beta_power_13_30_hz / (alpha_power_8_13_hz + theta_power_4_8_hz)
```

The exact formula is intentionally isolated in
`src/experiment_learning/eeg_indicator.py`:

```python
def engagement_index(eeg_features):
    beta = eeg_features["beta_power_13_30_hz"]
    alpha = eeg_features["alpha_power_8_13_hz"]
    theta = eeg_features["theta_power_4_8_hz"]
    denominator = alpha + theta
    if denominator <= 0.0:
        raise InvalidEEGIndicator("alpha power + theta power must be positive")
    return beta / denominator
```

There is no hidden epsilon, clipping, component-wise logarithm, or independent
coefficient for any of the eight original features. To test another scientific
formula later, change this small module and give the new formula a new identifier;
the original feature values retained in episode records allow deterministic
retraining without reprocessing raw EEG.

The participant-specific frozen model is:

```text
z = (engagement_index - mean) / scale
P_EEG = sigmoid(intercept + coefficient * z)
evidence = max(0, 2 * P_EEG - 1)
T_E = max(minimum_E, T_G * (1 - reduction_fraction * evidence))
```

With `T_G=1.0`, `reduction_fraction=0.4`, and `minimum_E=0.35`:

| `P_EEG` | Evidence | E dwell |
| ---: | ---: | ---: |
| 0.30 | 0.00 | 1.00 s |
| 0.50 | 0.00 | 1.00 s |
| 0.75 | 0.50 | 0.80 s |
| 1.00 | 1.00 | 0.60 s |

Session 1 sets the reduction to zero, so G and E behave identically before any
participant data exist.

## Predetermined condition schedule

`load_condition_schedule(path)` reads an approved CSV with exactly these columns:

```csv
sequence_id,session_number,active_condition
g-first,1,G
g-first,2,E
```

The loader verifies unique, contiguous session rows and computes SHA-256 over the
exact CSV bytes. `resolve_scheduled_condition(...)` binds the sequence ID and CSV
digest. A later edit to the CSV fails against a persisted binding instead of
silently changing a participant's assignment.

```python
from experiment_learning.schedule import (
    load_condition_schedule,
    resolve_scheduled_condition,
)

schedule = load_condition_schedule("configs/experiment_condition_schedule.csv")
scheduled = resolve_scheduled_condition(
    schedule,
    sequence_id="g-first",
    session_number=2,
    persisted_binding=prior_binding,
)
assert scheduled.condition.value == "E"
```

A failed attempt does not advance the schedule. A retry must resolve the same
session number and load the policy using the artifact digest saved when the
attempt began. `load_frozen_policy(..., expected_sha256=...)` enforces this even
if a newer policy file exists elsewhere.

The first live prototype round records an explicit CLI assignment rather than
activating this CSV path. `assignment_session_NNN.json` records
`selection_source: cli`, the active condition, and its derived complement. A
started retry must provide the same CLI active condition. The exact CSV loader and
header remain unchanged for the schedule-backed verifier and later protocol use;
its events identify `model_selection_source: csv`.

## Runtime workflow

For one successful attempt, Instance 5 should:

1. load the approved schedule and resolve the condition with its persisted digest;
2. load the participant policy for the exact session and saved artifact digest;
3. construct Instance 2's `DwellController` from
   `policy.dwell_parameters(active_condition)`;
4. process each gaze update with the held episode score from the preceding update;
5. pass `trigger_gate_open=controller.action_gate_open` to Instance 2;
6. process an ended episode or visible action before evaluating the current new
   episode update;
7. call `evaluate_update(...)` on every confirmed matched observation so the
   dwell trajectory begins before the EEG cutoff;
8. route announcements to `open_action_feedback(...)`, natural eligible silence
   to `open_no_action_feedback(...)`, presses to `accept_feedback(...)`, and time
   progression to `advance_time(...)`;
9. apply returned cancellation instructions through Instance 2's typed
   `cancel(..., FEEDBACK_INTERRUPTION)` path and enforce the experiment cooldown;
10. cancel remaining state at the session deadline, then create and persist the
    completed-session artifact;
11. call `train_next_session_policy(...)` exactly once and use its output only in
    the next successful session number.

Current Instance 2 applies a newly returned score on the following gaze update.
That N-to-N+1 ordering remains intentional: the EEG cutoff cannot influence dwell
already accumulated on update N.

### Live Guardian feature source

Live attempts should pass `GuardianEEGFeatureSource`, not a bare `EEGPipeline`, to
`ExperimentController.evaluate_update(...)`. The source requests the adapter's
current ordered snapshot for the exact closed interval and passes it to
`EEGPipeline.features_from_window(...)`. It never appends provisional or replaceable
samples to the pipeline's ordered replay buffer.

```python
from experiment_learning.guardian_source import GuardianEEGFeatureSource

guardian = GuardianAdapter(clock=attempt_clock.now, ...)
guardian.connect()
guardian.check_battery()
guardian.start_impedance()
# Poll guardian.latest_impedance() for fitting, then:
guardian.stop_impedance()

# At SPACE, start the shared attempt clock before raw EEG.
attempt_clock.start()
guardian.start(recording_seconds=session_recording_seconds)

eeg_source = GuardianEEGFeatureSource(
    guardian=guardian,
    pipeline=eeg_pipeline,
    recorder=eeg_recorder,
)

# Existing periodic calls remain the health/finalization hook. They retain the
# newest 30 seconds so future feature windows can still accept late packets.
eeg_source.drain_through(latest_processed_timestamp)
```

`features(start, end)` first checks Guardian health, finalizes and records data
strictly before `start`, then obtains `guardian.window(start, end)`. Not-yet-arrived
positions are explicit invalid samples, so the existing quality gate rejects that
window without fabricating EEG evidence. Late packets may correct a subsequent
overlapping request, but an already-frozen episode decision is never recomputed.
The source default retention horizon is 30 seconds; requests older than finalized
history fail explicitly.

`drain_through(cutoff)` no longer means callback-order ingestion. It checks
asynchronous acquisition health and incrementally records only samples older than
`cutoff - retention_seconds`. This preserves the existing Integration hook without
closing the recent mutable horizon. `drain_remaining()` finalizes through the last
processed scientific cutoff after recording stops. Finalized gaps are written to
raw HDF5 as `valid=false`. Acquisition, capacity, recorder, and cleanup failures
remain hard attempt errors.

Normal completion is ordered explicitly:

```python
guardian.stop()
eeg_source.drain_remaining()
guardian.disconnect()
guardian.close()
```

Keep stop, finalization, and close in independent cleanup blocks so disconnect is
still attempted if an earlier step fails. `recording_id`, battery, impedance,
capacity/loss status, and source finalization counters are diagnostics/provenance
only and must not enter the engagement index or frozen policy.

### Core construction example

```python
from experiment_learning.policy import load_frozen_policy
from experiment_learning.schedule import ScheduleBinding
from experiment_learning.state_machine import ExperimentController

policy = load_frozen_policy(
    "runs/P017/policy_session_003.json",
    expected_participant_id="P017",
    expected_session=3,
    expected_sha256=attempt_policy_digest,
)

controller = ExperimentController(
    policy=policy,
    policy_sha256=attempt_policy_digest,
    session_id="P017-S003",
    session_number=3,
    attempt_id="P017-S003-attempt-01",
    active_condition=scheduled.condition,
    schedule_binding=ScheduleBinding(
        policy.schedule_sequence_id,
        policy.schedule_sha256,
    ),
    minimum_prediction_elapsed_s=0.25,
    eeg_window_s=1.0,
    feedback_timeout_s=1.5,
    event_logger=events,
)
```

### Per-gaze example

```python
# held_score came from the previous confirmed update.
interaction = gaze_pipeline.process_gaze(
    gaze,
    intent_score=held_score,
    trigger_gate_open=controller.action_gate_open,
)

if interaction.ended_episode is not None:
    controller.open_no_action_feedback(
        interaction.ended_episode,
        interaction.ended_episode.end_timestamp,
    )

if interaction.dwell_trigger is not None:
    announce_object(interaction.dwell_trigger.episode_id)
    controller.open_action_feedback(
        interaction.dwell_trigger.episode_id,
        display_timestamp=interaction.dwell_trigger.timestamp,
    )

observation = observation_from_interaction(interaction)
if observation is not None and interaction.active_episode is not None:
    decision = controller.evaluate_update(
        interaction.active_episode,
        observation,
        eeg_source,
        instructed_intention=current_trial_instruction,
    )
    held_score = decision.intent_score
```

## Causal cutoff and missing EEG

The first confirmed match at or after `episode_start + 0.25 s` is eligible once a
full non-negative one-second history exists. Instance 4 requests exactly the closed
interval `[cutoff - 1.0 s, cutoff]` and freezes that episode's result. It never
retries, carries a previous result forward, or uses later EEG.

Unavailable, rejected, incomplete, invalid-formula, or wrong-signature EEG:

- returns `intent_score=None`, which means G-threshold behavioral fallback;
- keeps the original quality state/reasons and any available original features;
- excludes both G and E from paired training;
- still opens feedback after a visible announcement so it can be corrected;
- does not create a silent/no-action button target, because that label would enter
  a scientifically ineligible example.

## Feedback and newer candidates

Feedback uses the fixed half-open window `[outcome, outcome + timeout)`:

| Visible outcome | Press | Common label |
| --- | --- | ---: |
| Action | No | 1 |
| Action | Yes | 0 |
| No action | No | 0 |
| No action | Yes | 1 |

Only one earlier target can be open, but newer gaze candidates continue to
accumulate provisional dwell while its trigger gate is closed.

- An accepted press finalizes the earlier target and returns typed cancellation
  instructions for every newer provisional episode.
- A timeout finalizes the earlier target and simply reopens the gate; it does not
  cancel a still-valid newer candidate.
- `cancel_episode(...)` records explicit exclusion provenance for feedback or
  session-deadline cancellation.
- Controlled-intention trials retain both instructed intention and feedback label,
  but are evaluation-only and excluded by the trainer.

## Episode records and censoring

Every resolved/canceled episode emits `experiment_episode_training_record` with
schema `experiment_episode_training_record_v1`. It includes:

- participant/session/attempt/episode identities and active condition;
- frozen policy digest and causal EEG interval;
- all eight original Instance 3 features;
- engagement-index identifier, human-readable formula, value, probability, and
  positive-only evidence;
- compact `(timestamp, accumulated_matched_dwell_s)` trajectory;
- actual display or natural endpoint and feedback fields;
- common label, controlled intention, G/E required dwell and outcomes;
- exclusion and cancellation lineage.

When the active policy displays an action, behavior after display is contaminated.
The record therefore ends its scientific trajectory there. A shadow threshold that
did not cross within that prefix is `counterfactual_censored`, not a fabricated
no-action. The trainer never invents post-announcement gaze continuation.

After a successful attempt, `controller.completed_session(timestamp)` creates
`experiment_completed_session_v1`. Failed/incomplete attempts are not trainer
inputs and do not advance the schedule.

## Deterministic between-session trainer

`train_next_session_policy(completed_sessions, prior_policy, output_path, config)`:

- requires successful sessions `1..N` for exactly one participant;
- rejects duplicate sessions/episodes, cross-participant data, schedule drift,
  lineage drift, and old schemas;
- excludes controlled, canceled, unusable-EEG, missing-label, or otherwise marked
  episodes with counts in the report;
- standardizes only the scalar engagement index;
- fits `intercept + coefficient*z` by balanced binary cross-entropy with
  configurable L2, deterministic SciPy L-BFGS-B, and exact zero initialization;
- searches G base thresholds first (`0.50..1.50 s`, step `0.05 s`), then searches
  reductions (`0.0..0.5`) while holding the chosen base fixed;
- requires at least 20 evaluated examples and 5 from each label class for fitting
  and for every censor-aware candidate;
- writes a carried-forward next artifact when data or fit stability is insufficient.

The selection objective is:

```text
2.0 * false positives
+ 1.0 * false negatives
+ 0.25 * mean true-positive latency in seconds
```

Tie-breaking is lower loss, fewer false positives, fewer misses, lower latency,
closest to the prior policy, then the safer higher base or smaller reduction.
Precision, recall, F1, counts, censoring, latency, full candidate tables, source
digests, and deterministic ordering are retained in
`experiment_training_report_v1`.

There is no trainer seed because there is no stochastic training step.

## Immutable artifacts

`experiment_policy_v1` records participant/session lineage, schedule digest,
source attempt digests, G threshold, EEG formula/signature, normalization,
one-variable logistic parameters, adjustment rule/bounds, and cold-start status.

Policy, completed-session, and training-report files use canonical JSON. Writing
identical content to the same path is idempotent; different content at an existing
path fails loudly. Legacy `.pkl` River checkpoints and old online-learning record
schemas are rejected rather than migrated because they were produced under the
superseded scientific design.

## Configuration

The complete default is `configs/experiment_learning.yaml`; partial overrides use
the project-wide recursive merge and unknown-key rejection.

Important options:

| Key | Default | Meaning |
| --- | ---: | --- |
| `sequence_id` | `g-first` | Row sequence in the approved CSV |
| `schedule_path` | schedule CSV | Predetermined G/E schedule |
| `timing.minimum_prediction_elapsed_s` | `0.25` | Earliest episode-relative EEG freeze |
| `timing.eeg_window_s` | `1.0` | Closed backward EEG history |
| `timing.feedback_timeout_s` | `1.5` | Half-open feedback duration |
| `cold_start_policy.base_threshold_s` | `1.0` | Session-1 common G/E base |
| `cold_start_policy.minimum_e_threshold_s` | `0.35` | Absolute E floor |
| `trainer.minimum_examples` | `20` | Minimum paired examples |
| `trainer.minimum_per_class` | `5` | Minimum examples for each label |
| `trainer.l2` | `0.0` | Scalar logistic coefficient penalty |
| `trainer.base_min_s/max_s/step_s` | `0.5/1.5/0.05` | G threshold grid |
| `trainer.reduction_values` | `0.0..0.5` | E maximum-reduction grid |
| `trainer.objective.*` | `2/1/0.25` | FP/FN/latency weights |
| `trainer.optimizer.*` | `1000`, `1e-12` | Deterministic L-BFGS-B limits |

Example partial override:

```yaml
participant_id: P017
sequence_id: e-first
trainer:
  l2: 0.1
```

Run the hardware-free workflow:

```bash
python scripts/run_experiment_learning.py
python scripts/run_experiment_learning.py --config path/to/override.yaml
```

It saves resolved configuration, schedule-bound policies, completed-session
artifacts, training reports, scientific JSONL events, and a deterministic summary.
The synthetic seed controls only generated data; the trainer itself is seedless.

## Instance 5 integration status

Instance 5 now implements this contract. It uses schedule-bound immutable JSON
policies and completed sessions, constructs dwell parameters from the active
condition, calls `evaluate_update`, routes explicit action/no-action/feedback
operations, applies typed cancellations, and trains only after successful closure.

Live attempts use the persistent Guardian lifecycle and
`GuardianEEGFeatureSource`: fitting impedance precedes SPACE, the attempt clock
starts before raw EEG, and every feature request evaluates a mutable ordered
snapshot on the Integration thread. Instance 5's live UI now uses continuous
impedance rather than `prepare()`; the hardware practice path demonstrates gaze
calibration -> live impedance -> SPACE -> recording. Adding MindLink as an
experimental Integration input remains separately scoped. The experiment
controller and predict-score-update order are unchanged.

Instances 1–3 require no algorithm changes. `river==0.22.0` remains removed;
existing NumPy and SciPy dependencies cover the corrected implementation.
