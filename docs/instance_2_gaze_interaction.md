# Instance 2 — Vision + gaze interaction

This subsystem consumes the canonical `foundations.contracts.SceneFrame` and
`GazeSample` types. It does not define a glasses API, EEG logic, intent model, or
experimental feedback/labeling logic.

## Detection and tracking

`gaze_interaction.detector.YOLOEDetector` wraps Ultralytics YOLOE and the default
weights are `yoloe-26n-seg-pf.pt`. Constructing the adapter is the point at which
Ultralytics may resolve/download weights; importing the package never downloads a
model. Foundation scene frames are RGB, so the adapter explicitly converts them to
the BGR ndarray convention expected at the Ultralytics prediction boundary.

Detector boxes are clipped to the image, degenerate boxes are discarded, and the
local `BoundingBox` uses normalized `[0, 1]` coordinates with origin at top-left,
`+x` right, and `+y` down. `ByteTrackAdapter` uses Supervision's ByteTrack and its
track IDs are unique/meaningful only inside one pipeline run. A tracker ID change
is a new interaction identity; there is no cross-track identity recovery.

## Association and candidate episodes

Association uses the newest tracked scene with timestamp `<=` the gaze timestamp
and rejects it when it is older than `association.max_scene_age_seconds`. It never
uses a future frame. Invalid gaze, an empty tracked scene, or no containing box
produces no candidate. Canonical gaze is never clipped or repaired.

When boxes overlap, the containing box with the smallest original area wins;
higher confidence and then lower track ID are deterministic tie-breakers. The
configurable normalized box margin defaults to zero.

The frozen downstream episode contract lives at
`gaze_interaction.episodes.CandidateEpisode`:

```python
CandidateEpisode(
    episode_id: int,
    track_id: int,
    label: str | None,
    start_timestamp: float,
    last_match_timestamp: float,
    end_timestamp: float | None,
    end_reason: EpisodeEndReason | None,
)
```

All timestamps are non-negative run-relative seconds. The first association starts
an episode. Repeated association with the same track continues it. Invalid gaze,
no match, or a temporarily missing track pauses it without extending dwell; the
same track resumes the episode only within `gap_grace_seconds`. Once the grace
interval is exceeded, the episode ends at `last_match_timestamp + grace` with
`gap_timeout`. A different track within the grace interval ends it immediately
with `candidate_change` and starts another episode. An active episode at source
end uses `source_end` unless its grace interval had already expired. `label` is the
track label captured when the episode starts; track identity, not later label text,
determines continuity.

These are candidate/dwell episodes, not physiological fixation classifications.

## Dwell boundary

`DwellController` accumulates only intervals between consecutive confirmed gaze
matches for the same episode. Explicit no-match/invalid samples pause dwell and
`max_sample_gap_seconds` prevents an unobserved long interval from being counted.
Changing episodes resets dwell. At most one `DwellTrigger` is emitted per episode.

Integration may pass `trigger_gate_open=False` to `DwellController.advance()` or
`GazeInteractionPipeline.process_gaze()` while an earlier feedback target remains
open. Dwell continues accumulating, but a threshold crossing becomes
`DwellState.trigger_pending` and emits no action. If the same episode is still
active when the gate reopens, the pending trigger is emitted exactly once on the
next confirmed matching gaze update. Its timestamp is the release update's
timestamp. No-match updates do not release it, and an episode end or candidate
change discards it.

`GazeInteractionPipeline.cancel(timestamp, reason)` explicitly ends and clears the
current candidate, dwell, and pending-trigger state without fabricating a trigger.
The reason must be `EpisodeEndReason.FEEDBACK_INTERRUPTION` or
`EpisodeEndReason.SESSION_DURATION_REACHED`. It returns an
`InteractionCancellation` containing the ended episode, cleared post-cancellation
dwell state, and whether a pending trigger was discarded. Episode IDs remain
run-unique after cancellation. Feedback attribution, session timing, and cooldown
remain Integration responsibilities.

The external `intent_score` is either `None` or a finite value in `[0, 1]`.
Invalid scores are rejected. `None` uses baseline dwell. Otherwise the requirement
is:

`clip(baseline_seconds * (1 - maximum_reduction_fraction * intent_score), minimum_seconds, maximum_seconds)`

This is a bounded adaptation of the GazeIntent dwell-scaling principle. The
controller consumes only the score; it does not estimate intent itself.

## Demo

Install the repository and test dependencies with:

```text
python -m pip install -e ".[dev]"
```

Run the complete configured workflow with:

```text
python scripts/run_gaze_interaction.py
python scripts/run_gaze_interaction.py --config path/to/partial_override.yaml
```

The default virtual images are deterministic random pixels, useful for
no-detection/dropout plumbing rather than meaningful detector evaluation. For
meaningful visual detection diagnostics, use replayed real scene data. Resolved
configuration is saved as JSON and runtime interaction state is written to JSONL;
diagnostic PNG frames are saved by default.

No external project source code is copied. Runtime reuse is through Ultralytics
YOLOE (AGPL-3.0/Ultralytics licensing), Supervision's ByteTrack adapter (MIT), and
OpenCV. The adaptive dwell rule is based on the published GazeIntent approach:
<https://arxiv.org/abs/2404.13829>.

Supervision is constrained below 0.31 because its 0.30 release deprecates the
verified `ByteTrack` API and announces its removal in 0.31.
