"""MediaPipe-backed detectors, isolated so they are easy to swap out.

Only this file (and nothing in framer.py) knows about MediaPipe. Uses the
Tasks API (mediapipe >= 0.10, including 1.x where the legacy mp.solutions
API was removed). Face detection uses BlazeFace: for framing we only need
bounding boxes and centers, which is far cheaper than a full face mesh.
Pose landmarks provide a torso-center fallback for when the face is not
visible (subject turned around, walked away, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from .framer import Subject
from .models import FACE_DETECTOR_URL, POSE_LANDMARKER_URL, get_model

# Pose landmark indices.
_TORSO_LANDMARKS = (11, 12, 23, 24)  # shoulders + hips
_NOSE, _LEFT_EYE, _RIGHT_EYE, _LEFT_EAR, _RIGHT_EAR = 0, 2, 5, 7, 8
_HEAD_LANDMARKS = (_NOSE, _LEFT_EYE, _RIGHT_EYE, _LEFT_EAR, _RIGHT_EAR)
_POSE_MIN_VISIBILITY = 0.5

# Video-mode timestamps must be monotonically increasing; the exact frame
# rate does not matter for detection, so we assume ~30 fps.
_FRAME_INTERVAL_MS = 33


class FaceDetector:
    """Detects faces and returns them as framing Subjects."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=get_model(FACE_DETECTOR_URL)),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self._timestamp_ms = 0

    def detect(self, frame_bgr: np.ndarray) -> List[Subject]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += _FRAME_INTERVAL_MS
        result = self._detector.detect_for_video(image, self._timestamp_ms)

        subjects: List[Subject] = []
        for det in result.detections:
            box = det.bounding_box
            if box.width <= 0 or box.height <= 0:
                continue
            subjects.append(
                Subject(
                    cx=box.origin_x + box.width / 2.0,
                    cy=box.origin_y + box.height / 2.0,
                    bbox=(
                        float(box.origin_x),
                        float(box.origin_y),
                        float(box.width),
                        float(box.height),
                    ),
                    area_ratio=(box.width * box.height) / float(w * h),
                )
            )
        return subjects

    def release(self) -> None:
        self._detector.close()


@dataclass
class PoseResult:
    """What the pose model contributes to framing."""

    # Torso center (shoulders + hips average), if visible.
    body_center: Optional[Tuple[float, float]]
    # Head center from nose/eyes/ears keypoints, if visible. Used as a face
    # proxy when the dedicated face detector misses (distance, motion blur,
    # steep camera angles).
    head_center: Optional[Tuple[float, float]]
    # Approximate face width in pixels (from ear or eye spacing), if
    # estimable. Lets the head proxy also drive auto-zoom.
    face_width: Optional[float]


class PoseDetector:
    """Body pose: torso center plus a head estimate as a face fallback."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=get_model(POSE_LANDMARKER_URL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def detect(self, frame_bgr: np.ndarray) -> Optional[PoseResult]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += _FRAME_INTERVAL_MS
        result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]

        def visible_points(indices):
            pts = {}
            for idx in indices:
                lm = landmarks[idx]
                visibility = lm.visibility if lm.visibility is not None else 0.0
                if visibility > _POSE_MIN_VISIBILITY:
                    pts[idx] = (lm.x * w, lm.y * h)
            return pts

        torso = visible_points(_TORSO_LANDMARKS)
        body_center = None
        if torso:
            body_center = (
                sum(p[0] for p in torso.values()) / len(torso),
                sum(p[1] for p in torso.values()) / len(torso),
            )

        head = visible_points(_HEAD_LANDMARKS)
        head_center = None
        face_width = None
        if head:
            head_center = (
                sum(p[0] for p in head.values()) / len(head),
                sum(p[1] for p in head.values()) / len(head),
            )
            if _LEFT_EAR in head and _RIGHT_EAR in head:
                face_width = math.dist(head[_LEFT_EAR], head[_RIGHT_EAR])
            elif _LEFT_EYE in head and _RIGHT_EYE in head:
                # Interocular distance is ~40% of face width.
                face_width = math.dist(head[_LEFT_EYE], head[_RIGHT_EYE]) / 0.4

        if body_center is None and head_center is None:
            return None
        return PoseResult(
            body_center=body_center, head_center=head_center, face_width=face_width
        )

    def release(self) -> None:
        self._landmarker.close()
