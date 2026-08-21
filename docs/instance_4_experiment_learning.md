# Frozen policies and between-session learning

`experiment_learning` owns G/E assignment records, per-session frozen policies,
episode outcomes, contextual feedback labels, immutable completed-session inputs,
and deterministic training of the next policy. It does not acquire EEG or decide
which tracked object is under gaze.

## Quick check

```bash
python scripts/run_experiment_learning.py
python scripts/run_experiment_learning.py --config path/to/override.yaml
```

The deterministic default runs four synthetic sessions and writes policies,
completed sessions, training reports, events, resolved config, and `summary.json`
below `runs/experiment_learning/`.

## Runtime policy

`G` always uses the frozen gaze dwell threshold. `E` evaluates one causal EEG
window per candidate episode after the prediction cutoff. Its approved indicator
is:

`beta_power_13_30_hz / (alpha_power_8_13_hz + theta_power_4_8_hz)`

The frozen logistic model maps that indicator to positive EEG evidence. Evidence
may shorten E dwell, but never below the frozen minimum and never by more than the
frozen reduction fraction. Unusable EEG falls back to the G threshold and marks
the episode ineligible for training; policy parameters never change during a
session.

One feedback button supplies the common label:

| Visible action | Button | Label |
| --- | --- | --- |
| Yes | No | 1 — desired action |
| Yes | Yes | 0 — undesired action |
| No | No | 0 — correct non-action |
| No | Yes | 1 — missed desired action |

Visible actions open feedback even when their EEG record is excluded. A no-action
feedback target opens only for an otherwise eligible paired episode. Feedback for
one episode is exclusive, and an accepted press cancels newer provisional episodes
according to typed cancellation instructions.

## Training and lineage

Training runs only between successful sessions and consumes immutable completed
sessions in deterministic order. It fits the EEG logistic mapping, evaluates G
base thresholds and allowed E reduction fractions, and favors fewer false
positives with the configured weighted objective. If there are too few eligible
examples or a stable fit is unavailable, it writes a carried-forward policy and
an explicit status rather than failing the completed session.

Policies bind participant, successful session number, assignment identity/digest,
source attempt digests, feature signature, and scientific formula. The live runner
uses a CLI assignment artifact; the hardware-free verifier uses the condition CSV.

## Configuration and CLI

The default is `configs/experiment_learning.yaml`. Key values are the 0.25 s
minimum prediction time, 1.0 s EEG window, 2.0 s feedback timeout, 1.0 s cold-start
G threshold, 0.5 s E floor, and trainer minimum of 20 examples with at least five
per class. The threshold grid spans 0.5–1.5 s and E reduction candidates span
0–0.5.

The standalone CLI exposes only `--config PATH`. Its `participant_id`, schedule,
episode count, and session count control the synthetic check; the same timing,
cold-start, and trainer sections are used by integrated paths.

## Source files

| File | Responsibility |
| --- | --- |
| `src/experiment_learning/__init__.py` | Package boundary. |
| `src/experiment_learning/contracts.py` | Conditions, observations, decisions, outcomes, and completed-session records. |
| `src/experiment_learning/artifacts.py` | Canonical digests, immutable JSON writes, and validated loads. |
| `src/experiment_learning/assignment.py` | CLI/CSV model assignment records and retry binding. |
| `src/experiment_learning/schedule.py` | Condition CSV validation, hashing, and session lookup. |
| `src/experiment_learning/eeg_indicator.py` | Approved one-dimensional EEG indicator. |
| `src/experiment_learning/features.py` | Instance 3 feature-signature mapping. |
| `src/experiment_learning/policy.py` | Frozen policy validation, dwell parameters, and cold start. |
| `src/experiment_learning/state_machine.py` | Episode evaluation, feedback, cancellation, and trainer records. |
| `src/experiment_learning/guardian_source.py` | Causal Guardian feature-window adapter. |
| `src/experiment_learning/sessions.py` | Completed-session serialization and digest validation. |
| `src/experiment_learning/trainer.py` | Deterministic between-session model and threshold selection. |
| `src/experiment_learning/synthetic.py` | Standalone multi-session verification workflow. |

## Focused tests

```bash
pytest tests/test_experiment_learning.py
```
