# Instance 2.5 — temporary video + gaze harness

This temporary development harness feeds an ordinary prerecorded video and either
mouse-driven or prerecorded CSV gaze into the existing `gaze_interaction`
pipeline. It is a controlled pointer proxy for pre-hardware development, not a
smart-glasses or eye-tracker simulator.

Install the repository before running scripts:

```text
python -m pip install -e ".[dev]"
```

The harness has one entry point:

```text
python scripts/run_test_harness.py --config path/to/partial_override.yaml
```

The default file is `configs/test_harness.yaml`. A custom file is a strict partial
override: unknown keys are rejected. `video_path` must be supplied because the
repository does not include test footage. The source assumes ordinary
constant-frame-rate video and assigns frame `i` the deterministic run-relative
timestamp `i / fps`. Wall-clock time is used only to pace display when
`playback.paced` is true.

## Mouse mode and recording

A minimal mouse-mode override is:

```yaml
video_path: path/to/video.mp4
gaze:
  mouse_csv_output_path: runs/example_mouse_gaze.csv
```

Move the cursor over the displayed scene to provide the current normalized gaze
point. A point that cannot be represented inside the scene is emitted as invalid;
coordinates are never clipped or repaired. One mouse gaze sample is supplied per
decoded video frame. Set `mouse_csv_output_path` to `null` to disable recording.

## CSV replay mode

Replay the same video with a saved trajectory using a partial override such as:

```yaml
video_path: path/to/video.mp4
gaze:
  mode: file
  csv_input_path: runs/example_mouse_gaze.csv
```

The CSV schema is exactly:

```text
timestamp,x,y,validity
```

`timestamp` is non-negative run/video-relative seconds. Valid `x` and `y` are
normalized `[0,1]` coordinates with a top-left origin. `validity` is `true` or
`false` (`1` and `0` are also accepted while reading). Invalid samples may have
empty coordinates. Rows must be in non-decreasing timestamp order. Replay does not
interpolate gaze, fabricate samples, or replace scientific timestamps with wall
clock time.

The harness loads `configs/gaze_interaction.yaml` as the authoritative Instance 2
configuration. If `gaze_interaction_config` names a partial YAML override, it is
resolved against that file using the normal Foundation config rules; detector,
tracker, association, episode, and dwell settings are not duplicated in the
harness config. The harness supplies no learned intent score.

Each run writes `resolved_config.json` and
`resolved_gaze_interaction_config.json` under `runs/test_harness/<run-id>/`.
Existing Instance 2 diagnostic rendering is used for the scene, gaze, tracks,
candidate, and dwell state; `visualization.save_frames` optionally saves rendered
PNG frames in the run directory.
