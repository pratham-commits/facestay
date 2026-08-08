"""AutoFramer: the frame-in / frame-out public API.

    from facestay import AutoFramer

    framer = AutoFramer(output_size=(1280, 720))
    while True:
        frame = ...            # BGR numpy array from anywhere
        out = framer.process(frame)   # smoothly framed BGR output

It does not own a camera and does not open windows, so it drops into any
pipeline: OpenCV capture, LiveKit agents, ffmpeg, virtual cameras.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detectors import FaceDetector, PoseDetector, PoseResult
from .framer import FramingConfig, FramingEngine, FramingState, Subject

# Approximate face height/width proportion, used when synthesizing a face
# box from pose head keypoints.
_FACE_ASPECT = 1.4


class AutoFramer:
    def __init__(
        self,
        output_size: Tuple[int, int] = (1280, 720),
        config: Optional[FramingConfig] = None,
        use_pose_fallback: bool = True,
        detect_every: int = 1,
    ) -> None:
        """
        Args:
            output_size: (width, height) of the returned frames.
            config: full FramingConfig for fine tuning; output_size inside it
                is overridden by the output_size argument.
            use_pose_fallback: run the pose model when no face is visible.
                Disable to save CPU if subjects always face the camera.
            detect_every: run detection every Nth frame (>=1). Smoothing
                continues on every frame, so 2 is usually invisible and
                nearly halves CPU usage.
        """
        self.config = config or FramingConfig()
        self.config.output_size = output_size
        self.engine = FramingEngine(self.config)
        self._face_detector = FaceDetector()
        self._pose_detector = PoseDetector() if use_pose_fallback else None
        self._detect_every = max(1, int(detect_every))
        self._frame_index = 0
        self._last_subjects: List[Subject] = []
        self._last_body: Optional[Tuple[float, float]] = None
        self._last_time: Optional[float] = None

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Detect, smooth, crop, and resize one BGR frame."""
        h, w = frame_bgr.shape[:2]

        if self._frame_index % self._detect_every == 0:
            subjects = self._face_detector.detect(frame_bgr)
            pose: Optional[PoseResult] = None
            if self._pose_detector is not None:
                # Pose runs when faces are missing (fallback) or present
                # (for the horizontal blend); skip it only if disabled.
                pose = self._pose_detector.detect(frame_bgr)
            if not subjects and pose is not None and pose.head_center is not None:
                # Face detector missed but pose sees the head: synthesize a
                # face subject from the keypoints so tracking (and zoom,
                # when the size is estimable) continues seamlessly.
                subjects = [self._subject_from_head(pose, w, h)]
            self._last_subjects = subjects
            self._last_body = pose.body_center if pose is not None else None
        self._frame_index += 1

        now = time.perf_counter()
        dt = None
        if self._last_time is not None:
            # Clamp against timer glitches and long stalls (e.g. debugger).
            dt = min(max(now - self._last_time, 1.0 / 120.0), 0.25)
        self._last_time = now

        x, y, cw, ch = self.engine.update(
            w, h, self._last_subjects, self._last_body, dt=dt
        )
        crop = frame_bgr[y : y + ch, x : x + cw]
        return cv2.resize(crop, self.config.output_size, interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def _subject_from_head(pose: PoseResult, frame_w: int, frame_h: int) -> Subject:
        cx, cy = pose.head_center  # type: ignore[misc]
        if pose.face_width is not None:
            fw = pose.face_width
            fh = fw * _FACE_ASPECT
            area_ratio = (fw * fh) / float(frame_w * frame_h)
        else:
            # Unknown size: anchor position only; area_ratio 0 is excluded
            # from zoom calculations by the engine.
            fw = fh = 0.0
            area_ratio = 0.0
        return Subject(
            cx=cx,
            cy=cy,
            bbox=(cx - fw / 2.0, cy - fh / 2.0, fw, fh),
            area_ratio=area_ratio,
        )

    # ------------------------------------------------------------------
    @property
    def state(self) -> FramingState:
        """Last frame's crop rect, zoom, and detections (for debug UIs)."""
        return self.engine.state

    def reset(self) -> None:
        self.engine.reset()
        self._frame_index = 0
        self._last_subjects = []
        self._last_body = None
        self._last_time = None

    def release(self) -> None:
        self._face_detector.release()
        if self._pose_detector is not None:
            self._pose_detector.release()
