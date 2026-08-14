# Live practice session

This diagnostic checks the complete live glasses path without creating an
experimental session or training input. It runs MindLink calibration, waits for an
explicit operator start, then runs fresh scene capture, canonical gaze,
YOLOE/ByteTrack, gaze association, and fixed gaze-only dwell. Guardian EEG can be
enabled as a monitor-only source.
Practice inherits the gaze-interaction detector defaults: a `0.45` confidence
threshold and the category allowlist for chair, laptop, cellphone, tablet, and wall
poster. Only accepted categories are tracked, displayed, and eligible for a
practice selection. Edit the terms or replace the category list through the
configured gaze-interaction override when pilot observations expose different
YOLOE wording.

Before launch, start the AdHawk Backend Service and use the same vendor SDK setup
as the MindLink smoke runner. For EEG, install the Guardian extra and create the
repository-root token file `.secrets/idun_api_token` containing only the token:

```text
python -m pip install -e ".[guardian]"
```

`/.secrets/` is ignored by Git. On POSIX, use mode `0600` for the token file; the
loader rejects broader permissions. The `IDUN_API_TOKEN` environment variable is
still supported and takes precedence over the file. Neither source is copied into
resolved configuration or diagnostics.

```text
python scripts/run_practice_session.py
python scripts/run_practice_session.py --with-eeg
python scripts/run_practice_session.py --verbose-decisions
python scripts/run_practice_session.py --concise-decisions
python scripts/run_practice_session.py --config path/to/partial_override.yaml --with-eeg
```

The launch lifecycle is intentionally strict:

```text
connect tracker
-> calibrate fully
-> connect Guardian and check battery with raw EEG OFF
-> stream and continuously display fitting impedance with video/gaze/raw EEG OFF
-> press SPACE to accept the fit
-> stop impedance
-> start the attempt clock, Guardian raw EEG, display, and fresh MindLink capture
```

The fitting line is refreshed while the gate waits. `None` is shown as waiting for
the first reading; received values are shown in both ohms and kOhms. `prepare()` is
not used because its impedance check is finite rather than an operator-controlled
fitting display. Enter is not required. `Q`, `Esc`, or Ctrl-C stops impedance and
closes Guardian without creating a video receiver, enabling gaze streams, starting
Guardian raw EEG, opening the video display, starting the attempt clock, or creating
sensor recording files.

`MindLinkAdapter.start_capture()` is deliberately called only after the gate. It
creates a new `VideoReceiver` at that point; no receiver or video transport is
pre-created or preserved through calibration. This lifecycle is mandatory for the
future experimental runner as well because the target hardware stopped delivering
frame callbacks when a receiver survived calibration and a post-calibration pause.

After acquisition starts, press `Q` or `Esc` in the display to stop. The same
integration-owned attempt clock, whose zero is the SPACE signal, is passed to
MindLink and Guardian. Practice uses `GuardianEEGFeatureSource` on the practice
thread: health checks drain only through the current attempt cutoff, status uses
an ordered replaceable window, and only finalized chronological samples reach
the raw HDF5 recorder. Late callback blocks can therefore correct a still-open
status window without being appended behind newer samples. Shutdown attempts
Guardian recording stop, `drain_remaining()`, and Guardian close independently,
captures the cloud recording ID, and disconnects on the SDK owner loop.

The display includes recognized objects, a high-contrast labelled gaze bullseye,
current gaze validity/coordinates, current candidate, fixed dwell,
selection banners, separate received- and processed-scene rates, gaze rate and
validity, frame/processing drops, and optional EEG rate/quality. EEG never changes
practice dwell. The configured 30 Hz ByteTrack rate is provisional: tune it together
with the lost-track buffer and association age after measuring the effective
processed-scene rate on the target computer.

Before SPACE, terminal notices use the explicit `[practice setup]` prefix. After
SPACE, attempt-relative timestamps start at zero. Terminal style is controlled by
`terminal.verbose_decisions`, which defaults to `false`. `--verbose-decisions` and
`--concise-decisions` override that setting for one run and are mutually exclusive.

Verbose mode reports Instance 2's actual transition outputs: candidate/episode
starts, candidate switches, temporary no-match pauses and resumptions, episode-end
reasons, dwell crossings at 25/50/75%, dwell triggers, and resulting practice
selections. Concise mode suppresses the transition-rich lines while retaining
lifecycle notices, selections, no-frame warnings, failures, and the final stop
reason/artifact directory. Both modes write the same detailed structured events to
`events.jsonl`; the setting affects terminal output only. Neither mode prints every
frame or gaze sample or recreates Instance 2's decision logic.

Artifacts are written below `runs/practice/<run-id>/`. They are deliberately outside
the participant/session hierarchy and never include a completed-session record,
schedule binding, participant policy, feedback labels, or trainer output. Raw glasses
recording is off by default because uncompressed 1280x720 video is very large. Raw
EEG and MindLink timing metadata are retained by default when their sources are used.
Each run also writes `environment_manifest.json` with the Python/platform details and
installed versions (or explicit unavailability) for OpenCV, Ultralytics, Supervision,
AdHawk, and the Guardian SDK. `practice_summary.json` persists received and processed
scene rates separately, labels the tracker-rate setting as provisional, and records
Guardian battery, impedance, cloud recording ID, sample rate, sample-gap count/largest
gap, mean/maximum callback receipt lag, and queue-overflow state. Tokens are excluded.

Live Guardian cancellation and combined MindLink/Guardian timing still require pilot
validation on the experiment computer. The practice maximum duration is a safety cap,
not the experimental session timer.
