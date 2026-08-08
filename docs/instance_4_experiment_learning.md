# Instance 4 — Experimental logic + parallel learning

`experiment_learning` owns the participant-specific G-versus-E experiment state
machine. It consumes Instance 2 candidate/dwell contracts and Instance 3
`EEGFeatureWindow` values; it does not reproduce gaze association, dwell
accumulation, EEG preprocessing, or quality gating.

The scientific order is fixed:

```text
candidate -> freeze G and E -> active-only dwell control -> action/no action
          -> feedback/timeout -> one common label -> score both -> update both
```

## Prediction and EEG timing

Each eligible candidate gets at most one frozen paired prediction. The cutoff is
the timestamp of the first confirmed matched gaze observation satisfying both:

1. `timestamp >= episode.start_timestamp + minimum_prediction_elapsed_s`
   (default `0.25 s`), and
2. a full non-negative `eeg_window_s` history can be requested (default `1.0 s`).

Thus an episode beginning at run time `4.00 s` can first predict at `4.25 s`, and
its EEG request is the closed interval `[3.25, 4.25]`. An episode beginning at
run time `0.00 s` cannot predict at `0.25 s`; the earliest possible cutoff is
`1.00 s`, because `[cutoff - 1.0, cutoff]` must stay inside the run timebase.

The EEG window always ends exactly at the prediction cutoff. It may contain
pre-candidate EEG; this is intentional because EEG is treated as slow context,
not as a brain-click signal. Samples later than the cutoff are prohibited.

If Instance 3 reports `UNAVAILABLE`, `REJECTED`, or no feature vector when the
eligible request is made, the whole pair is unavailable: G and E both expose no
model-derived dwell score, neither predicts/scores/updates, baseline Instance 2
dwell is used, and the episode is never retried. This paired-complete-case rule
is fixed for the comparison; zero fill, carry-forward, and G-only training are
not runtime options.

## Model features

G uses exactly seven features in this order:

| Feature | Meaning / units |
| --- | --- |
| `candidate_elapsed_s` | Cutoff minus candidate start, seconds |
| `matched_dwell_s` | Instance 2 accumulated confirmed-match dwell, seconds |
| `gaze_center_dx_norm` | Gaze x minus candidate-box center x, normalized scene units |
| `gaze_center_dy_norm` | Gaze y minus candidate-box center y, normalized scene units |
| `candidate_width_norm` | Candidate-box width, normalized scene units |
| `candidate_height_norm` | Candidate-box height, normalized scene units |
| `candidate_area_norm` | Normalized width × height |

E is those same seven features followed by Instance 3's eight features, unchanged
and in its published order:

```text
std_uv
peak_to_peak_uv
mean_abs_diff_uv
delta_power_1_4_hz
theta_power_4_8_hz
alpha_power_8_13_hz
beta_power_13_30_hz
low_gamma_power_30_40_hz
```

Object/track identity, object label, absolute session time, detector confidence,
and gaze confidence are not model features. A prediction requires a confirmed
current match, so gaze features are not imputed.

G and E are independent River `StandardScaler -> LogisticRegression` pipelines.
Both begin with neutral `P(intent=1) = 0.5`. The returned score is that probability
in `[0, 1]`; `0.5` is the default binary scoring threshold. Each model owns its
own scaler and learned weights.

## Instance 5 integration workflow

The current Instance 2 API accepts `intent_score` before it returns the
`InteractionUpdate` from which Instance 4 constructs the current observation.
Therefore a score frozen on gaze update N begins controlling dwell on update
N+1. This one-sample boundary is explicit rather than hidden.

For each gaze sample, use this order:

1. call Instance 2 with the last frozen active score (or `None`);
2. tell Instance 4 about an ended episode and/or `DwellTrigger` from that update;
3. build `GazeContextObservation` from the `InteractionUpdate`;
4. ask Instance 4 to consider/fetch the paired prediction using the EEG pipeline;
5. retain only the returned `decision.intent_score` for the next gaze update;
6. route physical feedback presses to `button_press(timestamp)` and call
   `advance_time(timestamp)` from the experiment timer so silence can time out.

Example skeleton:

```python
from experiment_learning.features import observation_from_interaction

held_score = None

for gaze in gaze_stream:
    update = gaze_pipeline.process_gaze(gaze, intent_score=held_score)

    # Close/open previous experimental state before considering a new candidate.
    if update.ended_episode is not None:
        experiment.on_episode_end(update.ended_episode)
    if update.dwell_trigger is not None:
        experiment.on_dwell_trigger(update.dwell_trigger)

    observation = observation_from_interaction(update)
    if observation is not None and update.active_episode is not None:
        decision = experiment.consider_prediction(
            update.active_episode,
            observation,
            eeg_pipeline,  # exposes .features(start, end) -> EEGFeatureWindow
            instructed_intention=current_trial_instruction,
        )
        held_score = decision.intent_score
    elif update.active_episode is None:
        held_score = None

    experiment.advance_time(float(gaze.timestamp))
```

If a candidate switches, `consider_prediction` for the new episode returns
`None` until its own eligibility cutoff, so the old score is not retained for
the new episode. During a temporary no-match gap the previous score can remain
held; Instance 2 does not accumulate dwell while the match is absent.

There is one edge case worth preserving in integration: if Instance 2 emits a
`DwellTrigger` on the same update before a paired prediction has ever been frozen,
call `on_dwell_trigger` before `consider_prediction`. Instance 4 marks that episode
`action_before_prediction` and never trains it. This prevents a baseline action
from being retrospectively scored as model-controlled.

### Active versus shadow output

Both models predict on a usable eligible episode. `PredictionRecord` stores both
probabilities, but `PredictionDecision.intent_score` is copied only from the
session's active `Condition.G` or `Condition.E` model. No method exposes a combined
G/E control score. `None` means Instance 2 should use its baseline dwell rule.

## Feedback workflow

An action opens its feedback window at `DwellTrigger.timestamp`. A predicted
episode that ends without a trigger opens a no-action feedback window at its
`CandidateEpisode.end_timestamp`. The default window is half-open
`[open, open + 1.5 s)`: a press exactly at the deadline is too late and the case
has already timed out.

| Visible outcome | Press before deadline | Common label |
| --- | --- | ---: |
| Action | No | 1 |
| Action | Yes | 0 |
| No action | No | 0 |
| No action | Yes | 1 |

Only one feedback window can be open. An episode that begins while another
episode's feedback is pending is suppressed for its whole lifetime. Presses with
no open window or at/after the deadline are ignored and logged.

Both frozen predictions are scored against the common label before either model
updates. `ScoredPredictions` must be materialized before the paired learner accepts
an update, making update-before-score difficult to perform accidentally.

Controlled trials pass `instructed_intention=0|1` at prediction time. That value is
recorded separately in `EpisodeResultRecord`; it never replaces the feedback label
used for training. Example: a controlled trial may record
`instructed_intention=0` and `common_label=1`, which is precisely the disagreement
needed later to evaluate feedback-label reliability.

## Session schedule and participant checkpoints

`SessionSchedule(participant_sequence_index=N)` uses sequence-index parity rather
than randomness:

```text
even N: G, E, G, E, ...
odd  N: E, G, E, G, ...
```

Call `allocate_next()` once when a session is committed. The synthetic runner
saves immediately after allocation so a resume does not silently reuse the same
schedule slot.

Participant checkpoints are trusted-local standard-library pickle files. They
contain both River pipelines/scalers, G/E training counters, participant ID,
feature signature, model configuration, River version, and schedule state.
`ExperimentController` atomically saves after every finalized trained episode;
call `save_session_checkpoint()` again at session end.

Resume example:

```python
from experiment_learning.checkpoint import load_participant_checkpoint
from experiment_learning.models import ModelConfig

model_config = ModelConfig(learning_rate=0.01, l2=0.0, decision_threshold=0.5)
state = load_participant_checkpoint(
    "checkpoints/P017.pkl",
    expected_participant_id="P017",
    expected_participant_sequence_index=16,
    expected_model_config=model_config,
)
```

Wrong participant, schema, feature signature, model configuration, sequence index,
training counters, or River version fails loudly. There is no migration layer.
Because pickle can execute code during loading, do not load checkpoints obtained
from an untrusted source.

## Configuration: what is adjustable and what is fixed

The complete synthetic default is `configs/experiment_learning.yaml`. Important
adjustable values are:

| Config path | Default | Purpose |
| --- | ---: | --- |
| `timing.minimum_prediction_elapsed_s` | `0.25` | Earliest candidate-relative cutoff |
| `timing.eeg_window_s` | `1.0` | Backward EEG context length |
| `timing.feedback_timeout_s` | `1.5` | Feedback window duration |
| `model.learning_rate` | `0.01` | River SGD learning rate |
| `model.l2` | `0.0` | Logistic-regression L2 penalty |
| `model.decision_threshold` | `0.5` | Binary scoring threshold |
| `participant_sequence_index` | `0` | G-first/E-first counterbalance parity |
| `seed` | `42` | Deterministic synthetic fixture RNG |
| `episodes` | `2000` | Synthetic stress episode count |
| `sessions` | `4` | Synthetic schedule/session count |

The defaults are starting experimental parameters, not claims of physiological or
hardware optimality. The paired missing-EEG behavior, common feedback truth table,
single frozen prediction, active/shadow isolation, controlled-intention separation,
and score-before-update order are scientific rules, not YAML switches.

As with other project runners, a custom YAML is a partial strict override. Unknown
keys fail instead of being ignored. For example:

```yaml
episodes: 100
participant_id: synthetic-smoke
participant_sequence_index: 1
model:
  learning_rate: 0.005
resume_check:
  enabled: false
```

Run it with:

```bash
python scripts/run_experiment_learning.py --config path/to/override.yaml
```

The default run needs no CV model or Guardian:

```bash
python scripts/run_experiment_learning.py
```

Each synthetic run writes `resolved_config.json`, `events.jsonl`,
`participant_state.pkl`, and `summary.json`. The runner intentionally exercises
both active conditions, all four feedback cases, usable/rejected/unavailable EEG,
controlled trials, and a checkpoint reload. Synthetic classifier accuracy is not
a scientific performance claim.

The only new runtime dependency is `river==0.22.0` (BSD-3-Clause). It is pinned
because that release keeps the repository's existing Python `>=3.10` floor;
Instance 4 imports River normally and does not copy external model code.

## Scientific records and events

`experiment_prediction` contains participant/session/episode identity, active
condition, cutoff, G/E probability or explicit paired unavailability, exact EEG
request/quality, and the active control score.

`experiment_episode_result` contains visible action/no-action, feedback timing and
press/timeout state, common label, both frozen probabilities and thresholded
correctness results, update status/reason, and optional instructed intention.

Auxiliary events such as `experiment_feedback_ignored`,
`experiment_episode_suppressed`, and `experiment_outcome_unscored` explain why an
interaction did not enter the paired experiment. Instance 5 should preserve these
events alongside the main scientific records rather than infer missing-data reasons
from absent rows.
