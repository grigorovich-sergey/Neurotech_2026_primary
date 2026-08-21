# Live practice session

`scripts/run_practice_session.py` exercises the live MindLink, YOLOE/ByteTrack,
gaze association, fixed gaze-only dwell, display, and optional Guardian quality
monitoring. It creates diagnostic artifacts only: no subject lineage, experimental
feedback labels, completed session, or training input.

## Practice commands

Run glasses/vision first, then add EEG:

```bash
python scripts/run_practice_session.py
python scripts/run_practice_session.py --with-eeg
```

Useful variants are:

```bash
python scripts/run_practice_session.py --verbose-decisions
python scripts/run_practice_session.py --without-eeg --concise-decisions
python scripts/run_practice_session.py --config path/to/override.yaml --with-eeg
```

`--with-eeg` and `--without-eeg` are mutually exclusive. So are
`--verbose-decisions` and `--concise-decisions`. `--config PATH` applies a strict
partial override of `configs/practice_session.yaml`.

## Operator flow

MindLink connects and calibrates before acquisition. With EEG enabled, Guardian
then connects, checks battery, and streams fitting impedance while raw EEG remains
off. SPACE accepts the fitting value, establishes time zero, starts Guardian raw
EEG, creates the fresh MindLink video receiver, and opens the display. Q, Esc, or
Ctrl-C before SPACE aborts without starting sensor recordings; Q or Esc after start
ends practice.

The display reports objects, gaze, candidate, dwell, selections, rates, drops, and
optional EEG quality. EEG never changes practice dwell. Concise terminal mode keeps
lifecycle, selection, warning, and stop messages; verbose mode adds candidate,
episode, dwell, and trigger transitions. Both write the same structured event log.

## Configuration

The default practice cap is 3600 s, EEG is off, raw glasses recording is off, raw
EEG recording is on when EEG is enabled, and the display is on. Processing uses a
latest-scene queue, a bounded gaze queue, and batches of up to 64 gaze samples.
`configs/practice_gaze.yaml` and `configs/practice_eeg.yaml` provide the live
subsystem overrides. Guardian credentials follow the token rules in the EEG doc.

Runs are written below `runs/practice/` with resolved configs, `events.jsonl`, an
environment manifest, `practice_summary.json`, MindLink timing metadata, and any
enabled `practice_glasses.h5` / `practice_eeg.h5` recordings.

## Source files

| File | Responsibility |
| --- | --- |
| `src/practice_session/__init__.py` | Public practice runner export. |
| `src/practice_session/runner.py` | Setup gate, live queues, vision/gaze processing, display, EEG status, recording, and cleanup. |

The runner composes `src/mindlink/`, `src/gaze_interaction/`, `src/eeg_pipeline/`,
and `src/foundations/`; their file maps are in the corresponding instance docs.

## Focused tests

```bash
pytest tests/test_practice_session.py
```
