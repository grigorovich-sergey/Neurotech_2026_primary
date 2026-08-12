# Instance 6 — AdHawk MindLink-197 adapter

This subsystem owns only the hardware boundary from the AdHawk MindLink-197 SDK to
canonical `foundations.contracts.SceneFrame` and `GazeSample` values. It performs no
object detection, tracking, gaze association, dwell logic, EEG processing, feedback,
or experiment orchestration. Integration supplies the shared attempt clock and feeds
the canonical outputs into Instance 2.

## Validated hardware path

Pilot testing established the working SDK path: `FrontendApi` for connection,
tracking, calibration and eye-tracking streams, plus `VideoReceiver` and
`start_video_stream(*receiver.address)` for scene video. `GAZE_IN_IMAGE` supplies
image-space gaze; the first two values are used as pixel x/y while the remaining
values are retained only as uninterpreted diagnostics. Observed scene frames are
1280x720 at roughly 30 Hz and gaze packets roughly 125 Hz, but the adapter does not
hard-code either rate or frame size.

The standalone smoke runner deliberately uses the empirically required lifecycle:
connect -> tracker ready -> quick-start calibration -> wait for SPACE -> create a
fresh `VideoReceiver` -> start gaze/video capture. Creating the receiver before the
calibration/pause caused a pilot failure in which no frame callbacks arrived.

## Public adapter boundary

```python
adapter = MindLinkAdapter(clock=attempt_clock.now)
adapter.connect(...)
calibration_result = adapter.calibrate(...)
adapter.start_capture(
    on_scene_frame=...,
    on_gaze_sample=...,
    on_frame_metadata=...,
    on_gaze_metadata=...,
)
adapter.stop_capture()
adapter.close()
```

`calibrate()` and `start_capture()` contain no key input. `start_capture()` creates a
fresh receiver each time and emits RGB `uint8` `SceneFrame` values plus normalized
`GazeSample` values. The executable `scripts/run_mindlink.py` alone owns the SPACE
pause and OpenCV gaze-circle display used for hardware smoke testing.

Pixel gaze follows the repository convention `x/(width-1)`, `y/(height-1)` with a
top-left origin. Missing, non-finite, or out-of-range values are invalid rather than
clipped or repaired. Confidence remains `None` because the meaning of the third and
fourth `GAZE_IN_IMAGE` values has not been established.

## Timing

The adapter accepts the integration-owned shared run-relative clock. Gaze vendor
timestamps are anchored to that clock on the first capture packet and subsequent
vendor deltas determine canonical gaze timestamps. The video callback's
`tracker_timestamp` was consistently `0.0` in pilot testing and is treated as
unavailable. Populated frame `datetime` values are similarly anchored on the first
frame and their deltas determine later frame timestamps. Host callback receipt times
and raw vendor timestamps are exposed separately as diagnostics. Backward vendor
timestamps are dropped rather than silently reordered.

## Failure behavior

Repeated tracker-ready events are harmless. A bounded frame queue drops the oldest
queued encoded frame when full and increments `dropped_frame_count` to avoid growing
latency. Disconnect state is explicit and stops capture; there is no automatic
reconnection during an experiment attempt. The vendor SDK is imported lazily, so
synthetic/replay workflows remain usable when `adhawkapi` is absent.

## Smoke runner

```text
python scripts/run_mindlink.py
python scripts/run_mindlink.py --config path/to/partial_override.yaml
```

The complete default is `configs/mindlink.yaml`; resolved configuration is saved
under `runs/mindlink/<run-id>/resolved_config.json`. The smoke runner performs the
validated calibration/SPACE/fresh-receiver sequence, prints frame dimensions once,
and overlays canonical gaze on the displayed scene until Q/Esc.

Real-hardware validation still required after integration: sustained run dropout
rate, scene/gaze latency, long-run clock behavior, disconnect behavior during an
attempt, and whether a supported extended/multi-point calibration mode exists in the
installed SDK.
