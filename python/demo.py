"""Webcam demo for facestay.

Opens two windows:
  - "Facestay - Output": the final smoothly framed video
  - "Facestay - Debug":  full camera view with detection overlays (D toggles)

Keyboard:
  Q / ESC   quit
  + / -     position smoothing speed up/down
  [ / ]     zoom smoothing speed down/up
  R         reset smoothing to defaults
  D         toggle debug window
  F         toggle fullscreen output
  1-9       track only the N largest faces
  0         track all faces

Run:  python demo.py
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from facestay import AutoFramer, FramingConfig

OUTPUT_WIN = "Facestay - Output"
DEBUG_WIN = "Facestay - Debug"

DEFAULT_POSITION_SPEED = 0.08
DEFAULT_ZOOM_SPEED = 0.05


PERMISSION_HINT = (
    "Could not read frames from the webcam.\n"
    "On macOS, make sure your terminal app has camera access:\n"
    "  System Settings -> Privacy & Security -> Camera -> enable Terminal\n"
    "(then fully quit and reopen the terminal and run the demo again)."
)


def open_camera(width: int = 1920, height: int = 1080, fps: int = 30) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0). " + PERMISSION_HINT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # AVFoundation (macOS) can deliver no frames for the first moments after
    # opening — and delivers none at all until camera permission is granted.
    # Warm up patiently instead of failing on the first read.
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        ok, _ = cap.read()
        if ok:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera: {actual_w}x{actual_h}")
            return cap
        time.sleep(0.1)
    cap.release()
    raise RuntimeError(PERMISSION_HINT)


def draw_output_overlay(out: np.ndarray, framer: AutoFramer) -> np.ndarray:
    h, w = out.shape[:2]
    state = framer.state

    # Zoom bar, bottom-right.
    bar_w, bar_h = 160, 10
    x0, y0 = w - bar_w - 20, h - 30
    cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
    zoom_min, zoom_max = framer.config.zoom_min, framer.config.zoom_max
    frac = (state.zoom - zoom_min) / (zoom_max - zoom_min)
    frac = max(0.0, min(1.0, frac))
    cv2.rectangle(
        out, (x0, y0), (x0 + int(bar_w * frac), y0 + bar_h), (0, 200, 0), -1
    )
    cv2.putText(
        out, f"{state.zoom:.2f}x", (x0, y0 - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )

    n = len(state.tracked_subjects)
    label = f"AUTO | faces: {n}" + ("" if n or state.body_center else " | recentering")
    cv2.putText(
        out, label, (20, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def draw_debug_overlay(
    frame: np.ndarray, framer: AutoFramer, fps: float
) -> np.ndarray:
    dbg = frame.copy()
    state = framer.state
    cfg = framer.config

    # Crop window (green).
    x, y, cw, ch = state.crop
    cv2.rectangle(dbg, (x, y), (x + cw, y + ch), (0, 255, 0), 2)

    # Faces: blue boxes, red center dots.
    face_avg = None
    if state.tracked_subjects:
        xs = [s.cx for s in state.tracked_subjects]
        ys = [s.cy for s in state.tracked_subjects]
        face_avg = (sum(xs) / len(xs), sum(ys) / len(ys))
    for i, s in enumerate(state.tracked_subjects):
        bx, by, bw, bh = (int(v) for v in s.bbox)
        cv2.rectangle(dbg, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
        cv2.putText(
            dbg, f"#{i}", (bx, by - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA,
        )
        cv2.circle(dbg, (int(s.cx), int(s.cy)), 4, (0, 0, 255), -1)

    # Body center: cyan dot, plus blend line to the face average.
    if state.body_center is not None:
        bcx, bcy = int(state.body_center[0]), int(state.body_center[1])
        cv2.circle(dbg, (bcx, bcy), 6, (255, 255, 0), -1)
        cv2.putText(
            dbg, "BODY", (bcx + 10, bcy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA,
        )
        if face_avg is not None:
            cv2.line(
                dbg, (int(face_avg[0]), int(face_avg[1])), (bcx, bcy),
                (200, 200, 200), 1, cv2.LINE_AA,
            )

    # Semi-transparent info panel.
    panel = dbg.copy()
    cv2.rectangle(panel, (10, 10), (430, 150), (0, 0, 0), -1)
    dbg = cv2.addWeighted(panel, 0.55, dbg, 0.45, 0)
    limit = cfg.max_subjects if cfg.max_subjects is not None else "all"
    lines = [
        f"FPS: {fps:.1f}   faces: {len(state.tracked_subjects)}   "
        f"body: {'yes' if state.body_center else 'no'}",
        f"zoom: {state.zoom:.2f}x   pos speed: {cfg.position_speed:.3f}   "
        f"zoom speed: {cfg.zoom_speed:.3f}",
        f"tracking: {limit} largest face(s)",
        "keys: Q quit  +/- pos  [/] zoom  R reset  D debug",
        "      F fullscreen  1-9/0 face limit",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            dbg, text, (20, 36 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return dbg


def main() -> None:
    cap = open_camera()
    framer = AutoFramer(output_size=(1280, 720))
    cfg = framer.config

    show_debug = True
    fullscreen = False
    cv2.namedWindow(OUTPUT_WIN, cv2.WINDOW_NORMAL)

    fps = 0.0
    prev = time.perf_counter()

    failed_reads = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads > 30:
                    print("Camera read failed repeatedly, stopping.")
                    break
                time.sleep(0.05)
                continue
            failed_reads = 0

            out = framer.process(frame)

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            cv2.imshow(OUTPUT_WIN, draw_output_overlay(out, framer))
            if show_debug:
                cv2.imshow(DEBUG_WIN, draw_debug_overlay(frame, framer, fps))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # Q or ESC
                break
            elif key in (ord("+"), ord("=")):
                cfg.position_speed = min(0.5, cfg.position_speed + 0.01)
            elif key == ord("-"):
                cfg.position_speed = max(0.01, cfg.position_speed - 0.01)
            elif key == ord("["):
                cfg.zoom_speed = max(0.01, cfg.zoom_speed - 0.01)
            elif key == ord("]"):
                cfg.zoom_speed = min(0.5, cfg.zoom_speed + 0.01)
            elif key == ord("r"):
                cfg.position_speed = DEFAULT_POSITION_SPEED
                cfg.zoom_speed = DEFAULT_ZOOM_SPEED
            elif key == ord("d"):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow(DEBUG_WIN)
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    OUTPUT_WIN,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            elif ord("1") <= key <= ord("9"):
                cfg.max_subjects = key - ord("0")
            elif key == ord("0"):
                cfg.max_subjects = None
    finally:
        framer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
