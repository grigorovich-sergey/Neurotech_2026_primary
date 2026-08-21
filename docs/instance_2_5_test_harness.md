# Prerecorded video and gaze harness

The test harness feeds an ordinary constant-frame-rate video and either mouse or
CSV gaze through the real `gaze_interaction` pipeline. It is useful for tuning
detection categories, tracking, association, and dwell before a live session.

## Practice run

Create a partial override because the repository does not include test video:

```yaml
video_path: path/to/video.mp4
gaze:
  mode: mouse
  mouse_csv_output_path: runs/example_mouse_gaze.csv
```

```bash
python scripts/run_test_harness.py --config path/to/video_mouse.yaml
```

Move the cursor over the displayed video to generate normalized gaze. Replay the
saved trajectory with the same video:

```yaml
video_path: path/to/video.mp4
gaze:
  mode: file
  csv_input_path: runs/example_mouse_gaze.csv
```

```bash
python scripts/run_test_harness.py --config path/to/video_replay.yaml
```

CSV columns are `timestamp,x,y,validity`. Timestamps are non-decreasing video
seconds; valid coordinates are normalized `[0, 1]`. Replay does not interpolate
or repair samples.

## Configuration and CLI

`configs/test_harness.yaml` controls the video path, mouse/file gaze mode,
optional CSV paths, playback pacing, display, and diagnostic frame saving. The
harness resolves `configs/gaze_interaction.yaml`; `gaze_interaction_config` may
name a strict partial override. The only CLI option is `--config PATH`.

Each run under `runs/test_harness/` writes `resolved_config.json`,
`resolved_gaze_interaction_config.json`, and optional diagnostic frames.

## Source files

| File | Responsibility |
| --- | --- |
| `src/test_harness/__init__.py` | Package boundary. |
| `src/test_harness/video.py` | CFR video decoder with deterministic frame timestamps. |
| `src/test_harness/gaze.py` | Mouse gaze, CSV writer, and timestamp-ordered CSV replay. |

The executable orchestration is `scripts/run_test_harness.py`; the exercised
vision/gaze modules are listed in [Vision and gaze interaction](instance_2_gaze_interaction.md).

```bash
pytest tests/test_gaze_pipeline.py tests/test_integration.py
```
