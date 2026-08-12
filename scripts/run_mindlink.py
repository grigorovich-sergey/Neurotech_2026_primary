"""Calibrate MindLink-197, then show a live scene/gaze hardware smoke demo."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import queue
import time

import cv2

from foundations.config import load_resolved_config, save_resolved_config
from foundations.contracts import GazeSample, SceneFrame
from foundations.timebase import MonotonicClock
from mindlink import MindLinkAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mindlink.yaml"



def create_mindlink_adapter(*, clock, config: dict, on_disconnect=None) -> MindLinkAdapter:
    """Construct the reusable adapter with integration-owned clock and config."""
    capture = config["capture"]
    return MindLinkAdapter(
        clock=clock,
        frame_queue_size=capture["frame_queue_size"],
        on_disconnect=on_disconnect,
    )


def calibrate_mindlink(adapter: MindLinkAdapter, config: dict):
    """Run vendor calibration only; no key input or capture is performed."""
    calibration = config["calibration"]
    return adapter.calibrate(
        marker_size_mm=calibration["marker_size_mm"],
        returning_user=calibration["returning_user"],
        timeout_seconds=calibration["timeout_seconds"],
    )


def start_mindlink_capture(
    adapter: MindLinkAdapter,
    *,
    on_scene_frame,
    on_gaze_sample,
    on_frame_metadata=None,
    on_gaze_metadata=None,
) -> None:
    """Start scene/gaze capture only; no calibration, key input, or UI."""
    adapter.start_capture(
        on_scene_frame=on_scene_frame,
        on_gaze_sample=on_gaze_sample,
        on_frame_metadata=on_frame_metadata,
        on_gaze_metadata=on_gaze_metadata,
    )

def _run_directory(output_root: str) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root / "mindlink" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def _wait_for_space() -> bool:
    print("Calibration complete. Press SPACE to start video (Q/Esc to quit).")
    try:
        import msvcrt
    except ImportError:
        return input("Press Enter to start video, or type q then Enter to quit: ").strip().lower() != "q"
    while True:
        key = msvcrt.getwch()
        if key == " ":
            return True
        if key.lower() == "q" or key == "\x1b":
            return False


def run_smoke_demo(config: dict) -> Path:
    run_directory = _run_directory(config["output_root"])
    save_resolved_config(config, run_directory / "resolved_config.json")

    clock = MonotonicClock()
    scenes: queue.Queue[SceneFrame] = queue.Queue(maxsize=2)
    latest_gaze: list[GazeSample | None] = [None]

    def on_scene(frame: SceneFrame) -> None:
        try:
            scenes.put_nowait(frame)
        except queue.Full:
            try:
                scenes.get_nowait()
            except queue.Empty:
                pass
            scenes.put_nowait(frame)

    def on_gaze(sample: GazeSample) -> None:
        latest_gaze[0] = sample

    adapter = create_mindlink_adapter(clock=clock.now, config=config)
    try:
        connection = config["connection"]
        print("Starting connection...")
        adapter.connect(
            connect_timeout_seconds=connection["connect_timeout_seconds"],
            tracker_ready_timeout_seconds=connection["tracker_ready_timeout_seconds"],
        )
        print("Tracker ready")

        print("Launching quick-start calibration...")
        result = calibrate_mindlink(adapter, config)
        print(f"Calibration result: {result}")
        if not _wait_for_space():
            return run_directory

        # start_capture creates a fresh VideoReceiver here, after calibration and the pause.
        start_mindlink_capture(
            adapter, on_scene_frame=on_scene, on_gaze_sample=on_gaze
        )

        demo = config["smoke_demo"]
        window = demo["window_name"]
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("Video demo running. Press Q or Esc in the video window to quit.")
        first_frame = True
        wait_started = time.monotonic()
        warned = False

        while True:
            try:
                frame = scenes.get(timeout=0.1)
            except queue.Empty:
                if not adapter.is_connected:
                    raise RuntimeError("MindLink disconnected during capture")
                if not warned and time.monotonic() - wait_started > demo["no_frame_warning_seconds"]:
                    print("No video frames received after configured warning interval.")
                    warned = True
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            if first_frame:
                print("Frame shape:", frame.image.shape)
                first_frame = False

            image_bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
            gaze = latest_gaze[0]
            if gaze is not None and gaze.valid:
                height, width = frame.image.shape[:2]
                x = int(round(gaze.x_normalized * (width - 1)))
                y = int(round(gaze.y_normalized * (height - 1)))
                cv2.circle(image_bgr, (x, y), 14, (0, 0, 255), 2)
                cv2.circle(image_bgr, (x, y), 3, (0, 0, 255), -1)

            cv2.imshow(window, image_bgr)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        adapter.close()
        cv2.destroyAllWindows()
    return run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="partial YAML configuration overriding the project default")
    args = parser.parse_args()
    try:
        config = load_resolved_config(DEFAULT_CONFIG, args.config)
        run_directory = run_smoke_demo(config)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(run_directory)


if __name__ == "__main__":
    main()
