"""Report the raw OpenCV code emitted by the experiment HUD presenter.

Run this on the experiment computer with the same display environment used by
``scripts/run_experiment.py``. OpenCV extended-key values are backend-specific,
so the measured value should be copied into ``feedback.key_code`` rather than
guessed from ASCII or virtual-key tables.
"""

from __future__ import annotations

import cv2
import numpy as np


def main() -> None:
    window = "NeuroTech feedback-key check"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    canvas = np.zeros((240, 900, 3), dtype=np.uint8)
    messages = (
        "Focus this window and press the HUD presenter Down/PageDown button.",
        "The raw cv2.waitKeyEx code will be printed in the terminal.",
        "Press Q or Esc to quit.",
    )
    for index, message in enumerate(messages):
        cv2.putText(
            canvas,
            message,
            (20, 55 + index * 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    try:
        while True:
            cv2.imshow(window, canvas)
            key = cv2.waitKeyEx(20)
            if key < 0:
                continue
            if key in (27, ord("q"), ord("Q")):
                break
            print(
                f"raw key code: {key} (hex 0x{key:x}); "
                f"set feedback.key_code: {key} after confirming this is the presenter button",
                flush=True,
            )
    finally:
        cv2.destroyWindow(window)


if __name__ == "__main__":
    main()
