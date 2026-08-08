"""Feed the auto-framed video into a virtual camera for Zoom / Meet / etc.

Any app that lists webcams will see a virtual camera whose feed is the
smoothly framed output. Select it in Zoom, Google Meet (Chrome), OBS, or any
interview platform — no integration needed on their side.

Requirements:
    pip install pyvirtualcam
    macOS: also install OBS (its virtual camera backend), then run OBS once.
    Windows: OBS or Unity Capture. Linux: v4l2loopback.

Run:  python vcam.py
"""

from __future__ import annotations

import sys
import time

import cv2

from facestay import AutoFramer

try:
    import pyvirtualcam
except ImportError:
    sys.exit("pyvirtualcam is not installed. Run: pip install pyvirtualcam")

OUTPUT_SIZE = (1280, 720)
FPS = 30


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam (device 0)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Warm up: AVFoundation (macOS) may deliver no frames right after opening
    # (or until camera permission is granted to the terminal).
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.1)
    else:
        sys.exit(
            "Could not read from the webcam. On macOS, grant camera access:\n"
            "System Settings -> Privacy & Security -> Camera -> enable Terminal"
        )

    framer = AutoFramer(output_size=OUTPUT_SIZE)

    try:
        with pyvirtualcam.Camera(
            width=OUTPUT_SIZE[0],
            height=OUTPUT_SIZE[1],
            fps=FPS,
            fmt=pyvirtualcam.PixelFormat.BGR,
        ) as cam:
            print(f"Virtual camera running: {cam.device}")
            print("Select it as your webcam in Zoom/Meet. Ctrl+C to stop.")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                cam.send(framer.process(frame))
                cam.sleep_until_next_frame()
    except KeyboardInterrupt:
        pass
    finally:
        framer.release()
        cap.release()


if __name__ == "__main__":
    main()
