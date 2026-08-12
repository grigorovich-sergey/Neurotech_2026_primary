from datetime import datetime, timedelta
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mindlink.adapter import MindLinkAdapter, _normalize_pixel_gaze


class CallbackEvent:
    def __init__(self):
        self.callback = None
    def add_callback(self, callback):
        self.callback = callback


class FakeReceiver:
    def __init__(self, log):
        self.log = log
        self.address = ("127.0.0.1", 9000)
        self.frame_received_event = CallbackEvent()
        self.shutdown_called = False
    def start(self):
        self.log.append("receiver.start")
    def shutdown(self):
        self.shutdown_called = True
        self.log.append("receiver.shutdown")


class FakeApi:
    def __init__(self, sdk, log):
        self.sdk = sdk
        self.log = log
        self.handlers = {}
        self.connect_cb = None
        self.disconnect_cb = None
        self.calibration_response = (0, 6)
        self.video_address = None
    def register_stream_handler(self, packet_type, handler):
        self.handlers[packet_type] = handler
    def start(self, *, connect_cb, disconnect_cb):
        self.connect_cb = connect_cb; self.disconnect_cb = disconnect_cb
        self.log.append("api.start")
        connect_cb(None)
    def enable_tracking(self, enabled):
        self.log.append(f"tracking:{enabled}")
        self.handlers[self.sdk.PacketType.TRACKER_READY]()
    def quick_start_gui(self, **kwargs):
        self.log.append("calibrate")
        kwargs["callback"](*self.calibration_response)
    def set_et_stream_control(self, streams, enabled):
        self.log.append(f"streams:{enabled}")
    def start_video_stream(self, *address):
        self.video_address = address
        self.log.append("video.start")
    def stop_video_stream(self, *address):
        self.log.append("video.stop")
    def shutdown(self):
        self.log.append("api.shutdown")


def fake_sdk():
    return SimpleNamespace(
        frontend=SimpleNamespace(),
        PacketType=SimpleNamespace(TRACKER_READY="ready", EYETRACKING_STREAM="et"),
        EyeTrackingStreamTypes=SimpleNamespace(GAZE="gaze", GAZE_IN_IMAGE="image"),
        MarkerSequenceMode=SimpleNamespace(FIXED_GAZE="fixed"),
    )


def make_adapter(clock_values=None, frame_queue_size=2):
    sdk = fake_sdk(); log=[]; api=FakeApi(sdk,log); receivers=[]
    if clock_values is None:
        counter = iter(float(i) for i in range(100))
        clock=lambda: next(counter)
    else:
        counter=iter(clock_values); clock=lambda: next(counter)
    def receiver_factory():
        receiver=FakeReceiver(log); receivers.append(receiver); return receiver
    adapter=MindLinkAdapter(clock=clock, frame_queue_size=frame_queue_size, sdk=sdk, api_factory=lambda: api, video_receiver_factory=receiver_factory)
    return adapter,api,receivers,log


def encoded_image(rgb):
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    ok,data=cv2.imencode('.png',bgr)
    assert ok
    return data.tobytes()


def wait_until(predicate, timeout=1.0):
    deadline=time.time()+timeout
    while time.time()<deadline:
        if predicate(): return
        time.sleep(0.01)
    assert predicate()


def test_pixel_normalization_uses_existing_project_convention_and_never_clips():
    assert _normalize_pixel_gaze(1279,719,width=1280,height=720)==(1.0,1.0)
    assert _normalize_pixel_gaze(0,0,width=1280,height=720)==(0.0,0.0)
    assert _normalize_pixel_gaze(1279.1,100,width=1280,height=720) is None
    assert _normalize_pixel_gaze(float('nan'),100,width=1280,height=720) is None


def test_connect_calibrate_and_capture_create_receiver_only_at_capture_start():
    adapter,api,receivers,log=make_adapter()
    adapter.connect(); result=adapter.calibrate()
    assert result==(0,6)
    assert receivers==[]
    adapter.start_capture(on_scene_frame=lambda _:None,on_gaze_sample=lambda _:None)
    assert len(receivers)==1
    assert log.index('calibrate') < log.index('receiver.start') < log.index('video.start')
    adapter.close()


def test_frame_conversion_is_rgb_and_frame_datetime_preserves_spacing():
    # host times: first frame receipt 3.0, second 10.0; vendor delta should determine second canonical time.
    adapter,api,receivers,log=make_adapter([3.0,10.0])
    scenes=[]; metadata=[]
    adapter.connect(); adapter.start_capture(on_scene_frame=scenes.append,on_gaze_sample=lambda _:None,on_frame_metadata=metadata.append)
    receiver=receivers[0]
    rgb=np.zeros((2,3,3),dtype=np.uint8); rgb[0,0]=[255,0,0]
    t0=datetime(2026,8,12,12,0,0)
    receiver.frame_received_event.callback(0.0,encoded_image(rgb),t0)
    receiver.frame_received_event.callback(0.0,encoded_image(rgb),t0+timedelta(seconds=.04))
    wait_until(lambda: len(scenes)==2)
    np.testing.assert_array_equal(scenes[0].image,rgb)
    assert scenes[0].timestamp==pytest.approx(3.0)
    assert scenes[1].timestamp==pytest.approx(3.04)
    assert metadata[0].tracker_timestamp is None
    adapter.close()


def test_gaze_uses_vendor_spacing_and_is_invalid_until_scene_shape_known():
    adapter,api,receivers,log=make_adapter([1.0,1.1,1.2])
    gazes=[]; adapter.connect(); adapter.start_capture(on_scene_frame=lambda _:None,on_gaze_sample=gazes.append)
    handler=api.handlers[api.sdk.PacketType.EYETRACKING_STREAM]
    handler(SimpleNamespace(timestamp=100.0,gaze_in_image=(640.0,360.0,15.0,15.0)))
    assert not gazes[-1].valid
    rgb=np.zeros((720,1280,3),dtype=np.uint8)
    receivers[0].frame_received_event.callback(0.0,encoded_image(rgb),datetime(2026,8,12,12,0,0))
    wait_until(lambda: adapter._scene_shape==(720,1280))
    handler(SimpleNamespace(timestamp=100.008,gaze_in_image=(1279.0,719.0,15.0,15.0)))
    assert gazes[-1].timestamp==pytest.approx(1.008)
    assert gazes[-1].valid and gazes[-1].x_normalized==1.0 and gazes[-1].y_normalized==1.0
    adapter.close()


def test_backward_gaze_timestamp_is_dropped():
    adapter,api,receivers,log=make_adapter([2.0,2.1])
    gazes=[]; adapter.connect(); adapter.start_capture(on_scene_frame=lambda _:None,on_gaze_sample=gazes.append)
    handler=api.handlers[api.sdk.PacketType.EYETRACKING_STREAM]
    handler(SimpleNamespace(timestamp=50.0,gaze_in_image=None))
    handler(SimpleNamespace(timestamp=49.0,gaze_in_image=None))
    assert len(gazes)==1
    assert adapter.dropped_gaze_timestamp_count==1
    adapter.close()


def test_repeated_tracker_ready_is_harmless_and_disconnect_stops_capture():
    disconnected=[]
    sdk=fake_sdk(); log=[]; api=FakeApi(sdk,log); receiver=FakeReceiver(log)
    adapter=MindLinkAdapter(clock=lambda:1.0,sdk=sdk,api_factory=lambda:api,video_receiver_factory=lambda:receiver,on_disconnect=disconnected.append)
    adapter.connect(); api.handlers[sdk.PacketType.TRACKER_READY](); adapter.start_capture(on_scene_frame=lambda _:None,on_gaze_sample=lambda _:None)
    assert adapter.is_capturing
    api.disconnect_cb('lost')
    assert not adapter.is_capturing and not adapter.is_connected
    assert adapter.disconnect_error=='lost' and disconnected==['lost']
    adapter.close()


def test_frame_queue_drops_oldest_without_unbounded_growth(monkeypatch):
    adapter,api,receivers,log=make_adapter(frame_queue_size=1)
    scenes=[]; adapter.connect(); adapter.start_capture(on_scene_frame=scenes.append,on_gaze_sample=lambda _:None)
    # Stop worker briefly so two callbacks deterministically contend for one slot.
    adapter._frame_stop.set(); adapter._frame_worker.join(timeout=1.0); adapter._frame_stop.clear()
    receiver=receivers[0]; t=datetime(2026,8,12,12,0,0)
    data=encoded_image(np.zeros((2,2,3),dtype=np.uint8))
    receiver.frame_received_event.callback(0.0,data,t)
    receiver.frame_received_event.callback(0.0,data,t+timedelta(seconds=.04))
    assert adapter.dropped_frame_count==1
    adapter.close()
