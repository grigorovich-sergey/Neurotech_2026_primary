# Vision and gaze interaction

`gaze_interaction` converts canonical scene and gaze samples into tracked
objects, one current candidate episode, accumulated dwell, and at most one
selection trigger per episode. It does not acquire glasses data, process EEG,
or assign feedback labels.

## Quick check

```bash
python scripts/run_gaze_interaction.py
python scripts/run_gaze_interaction.py --config path/to/override.yaml
```

The default uses deterministic virtual frames and is useful for pipeline and
dropout checks. Use the video/gaze harness or recorded scene input for meaningful
object-recognition checks.

## Processing behavior

YOLOE detections above the configured confidence threshold are normalized and
filtered into project categories before ByteTrack. Association uses only a
tracked scene at or before the gaze timestamp and rejects stale scenes. When
boxes overlap, the smallest containing box wins with deterministic tie-breaks.

The same track continues an episode. Invalid/no-match gaze pauses it; a different
track or an expired grace interval ends it. Dwell accumulates only between
confirmed matches and ignores unobserved long gaps. A pending threshold crossing
waits while feedback for an earlier action is open, then releases once if the
same episode is still current.

The dwell requirement is bounded by the configured baseline, minimum, maximum,
and optional intent score. The gaze pipeline consumes an intent score but does
not calculate it.

## Configuration and CLI

The default is `configs/gaze_interaction.yaml`; `configs/practice_gaze.yaml`
overrides live camera/tracker settings. Key defaults are:

- YOLOE `yoloe-26n-seg-pf.pt`, confidence `0.3`, image size `640`, device
  `cuda:0`;
- category filter enabled for chair, laptop, backpack, TV, mug, poster, table,
  handbag, and box categories;
- ByteTrack frame rate `10` (`30` in live practice/main);
- scene age `0.25 s`, episode grace `0.15 s`, and maximum gaze gap `0.10 s`;
- baseline dwell `1.0 s` with bounded intent reduction.

The CLI exposes only `--config PATH`. Category lists replace the default list
wholesale in an override.

## Source files

| File | Responsibility |
| --- | --- |
| `src/gaze_interaction/__init__.py` | Package boundary. |
| `src/gaze_interaction/contracts.py` | Detection, tracked-object, scene, and normalized-box records. |
| `src/gaze_interaction/detector.py` | YOLOE adapter and category mapping. |
| `src/gaze_interaction/tracker.py` | Supervision ByteTrack adapter. |
| `src/gaze_interaction/association.py` | Timestamp-safe gaze-to-object association. |
| `src/gaze_interaction/episodes.py` | Candidate episode lifecycle and end reasons. |
| `src/gaze_interaction/dwell.py` | Observed dwell accumulation and bounded threshold trigger. |
| `src/gaze_interaction/pipeline.py` | Scene/gaze orchestration and cancellation boundary. |
| `src/gaze_interaction/visualization.py` | Diagnostic tracks, gaze, candidate, and dwell overlays. |

Runs write resolved configuration and `events.jsonl`; optional diagnostic frames
are controlled by `visualization.*`.

```bash
pytest tests/test_gaze_association.py tests/test_gaze_episodes_dwell.py tests/test_gaze_pipeline.py
```
