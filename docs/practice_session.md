# Live practice session

This diagnostic checks the complete live glasses path without creating an
experimental session or training input. It runs MindLink calibration, fresh
post-calibration scene capture, canonical gaze, YOLOE/ByteTrack, gaze association,
and fixed gaze-only dwell. Guardian EEG can be enabled as a monitor-only source.

Before launch, start the AdHawk Backend Service and use the same vendor SDK setup
as the MindLink smoke runner. For EEG, install the Guardian extra and export the
configured token variable (default `IDUN_API_TOKEN`):

```text
python -m pip install -e ".[guardian]"
export IDUN_API_TOKEN=...
```

```text
python scripts/run_practice_session.py
python scripts/run_practice_session.py --with-eeg
python scripts/run_practice_session.py --config path/to/partial_override.yaml --with-eeg
```

Press `Q` or `Esc` in the display to stop. The same integration-owned
`MonotonicClock` is passed to MindLink and Guardian. Guardian shutdown is
cooperative: the practice stop signal cancels and awaits the SDK recording task so
its cleanup can finish.

The display includes recognized objects, a high-contrast labelled gaze bullseye,
current gaze validity/coordinates, current candidate, fixed dwell,
selection banners, separate received- and processed-scene rates, gaze rate and
validity, frame/processing drops, and optional EEG rate/quality. EEG never changes
practice dwell. The configured 30 Hz ByteTrack rate is provisional: tune it together
with the lost-track buffer and association age after measuring the effective
processed-scene rate on the target computer.

The terminal prints timestamped lifecycle notices plus selection triggers, no-frame
warnings, failures, and the final stop reason/artifact directory. It deliberately
does not print every frame or gaze sample.

Artifacts are written below `runs/practice/<run-id>/`. They are deliberately outside
the participant/session hierarchy and never include a completed-session record,
schedule binding, participant policy, feedback labels, or trainer output. Raw glasses
recording is off by default because uncompressed 1280x720 video is very large. Raw
EEG and MindLink timing metadata are retained by default when their sources are used.
Each run also writes `environment_manifest.json` with the Python/platform details and
installed versions (or explicit unavailability) for OpenCV, Ultralytics, Supervision,
AdHawk, and the Guardian SDK. `practice_summary.json` persists received and processed
scene rates separately and labels the tracker-rate setting as provisional.

Live Guardian cancellation and combined MindLink/Guardian timing still require pilot
validation on the experiment computer. The practice maximum duration is a safety cap,
not the experimental session timer.
