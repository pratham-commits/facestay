"""The framing brain: pure math, no camera, no ML.

This module decides *where to crop* given detected subjects. It is
deliberately free of OpenCV/MediaPipe imports so the exact same logic can be
ported to other languages (see js/src/framer.ts for the TypeScript twin).

Pipeline per frame:
  1. Blend face centers (and body center fallback) into a target point.
  2. Derive a target zoom from average face size (with a moving average and
     a dead-zone to prevent "breathing").
  3. Exponentially smooth current center/zoom toward the targets.
  4. Emit a crop rectangle clamped to the input frame.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple


@dataclass
class Subject:
    """A detected subject (usually a face) in pixel coordinates."""

    cx: float
    cy: float
    # Bounding box (x, y, w, h) in pixels.
    bbox: Tuple[float, float, float, float]
    # Fraction of the input frame area this subject occupies (0-1).
    area_ratio: float


@dataclass
class FramingConfig:
    output_size: Tuple[int, int] = (1280, 720)

    # Exponential smoothing speeds, expressed as per-frame factors at the
    # 30 fps reference rate (current += (target - current) * speed). The
    # engine converts them with the measured frame time, so the feel is the
    # same at any frame rate. Lower = heavier/slower camera.
    position_speed: float = 0.08
    zoom_speed: float = 0.05

    # Auto-zoom: the tracked face(s) should occupy ~this fraction of the
    # output frame area. ~0.10 gives a head-and-shoulders composition;
    # larger values crop tighter on the face.
    target_face_ratio: float = 0.10
    zoom_min: float = 0.6
    zoom_max: float = 3.0
    # Ignore zoom target changes smaller than this fraction (anti-breathing).
    zoom_deadzone: float = 0.05
    # Moving-average window (frames) for face size.
    face_size_history: int = 15

    # Face/body blend when both are visible.
    face_weight: float = 0.7

    # Headroom: shift the crop center down by this fraction of the crop
    # height, so faces sit above center (roughly the photographic upper
    # third) instead of dead-center with the head near the top edge.
    vertical_bias: float = 0.12

    # Track only the N largest faces (None = all).
    max_subjects: Optional[int] = None

    # When nothing is detected, drift back to frame center at this speed.
    recenter_speed: float = 0.02

    # After this many frames without any face, ease the zoom target out to
    # zoom_out_level (losing the subject should widen the shot, not freeze).
    zoom_out_after: int = 30
    zoom_out_level: float = 1.0


@dataclass
class FramingState:
    """Read-only snapshot of the engine, useful for debug overlays."""

    crop: Tuple[int, int, int, int] = (0, 0, 0, 0)
    zoom: float = 1.0
    target: Tuple[float, float] = (0.0, 0.0)
    tracked_subjects: List[Subject] = field(default_factory=list)
    body_center: Optional[Tuple[float, float]] = None


class FramingEngine:
    """Stateful crop-window controller. Feed it detections, get crop rects."""

    REFERENCE_FPS = 30.0

    def __init__(self, config: Optional[FramingConfig] = None) -> None:
        self.config = config or FramingConfig()
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None
        self._zoom: float = 1.0
        self._target_zoom: float = 1.0
        self._face_ratio_history: Deque[float] = deque(
            maxlen=self.config.face_size_history
        )
        self._frames_without_face = 0
        self.state = FramingState()

    # ------------------------------------------------------------------
    def update(
        self,
        frame_w: int,
        frame_h: int,
        subjects: Sequence[Subject],
        body_center: Optional[Tuple[float, float]] = None,
        dt: Optional[float] = None,
    ) -> Tuple[int, int, int, int]:
        """Advance one frame. Returns crop rect (x, y, w, h) in pixels.

        dt is the elapsed time since the previous frame in seconds; when
        omitted, the 30 fps reference interval is assumed.
        """
        cfg = self.config
        if dt is None:
            dt = 1.0 / self.REFERENCE_FPS

        if self._cx is None:
            self._cx, self._cy = frame_w / 2.0, frame_h / 2.0

        tracked = self._select_subjects(subjects)
        tx, ty, position_speed = self._target_center(
            frame_w, frame_h, tracked, body_center
        )
        self._update_target_zoom(tracked)

        self._cx = self._smooth(self._cx, tx, self._alpha(position_speed, dt))
        self._cy = self._smooth(self._cy, ty, self._alpha(position_speed, dt))
        self._zoom = self._smooth(
            self._zoom, self._target_zoom, self._alpha(cfg.zoom_speed, dt)
        )

        crop = self._crop_rect(frame_w, frame_h)
        self.state = FramingState(
            crop=crop,
            zoom=self._zoom,
            target=(tx, ty),
            tracked_subjects=list(tracked),
            body_center=body_center,
        )
        return crop

    def reset(self) -> None:
        self._cx = None
        self._cy = None
        self._zoom = 1.0
        self._target_zoom = 1.0
        self._face_ratio_history.clear()
        self._frames_without_face = 0

    # ------------------------------------------------------------------
    def _select_subjects(self, subjects: Sequence[Subject]) -> List[Subject]:
        chosen = sorted(subjects, key=lambda s: s.area_ratio, reverse=True)
        if self.config.max_subjects is not None:
            chosen = chosen[: self.config.max_subjects]
        return chosen

    def _target_center(
        self,
        frame_w: int,
        frame_h: int,
        tracked: Sequence[Subject],
        body_center: Optional[Tuple[float, float]],
    ) -> Tuple[float, float, float]:
        """Returns (tx, ty, speed). Speed is lowered when drifting to center."""
        cfg = self.config
        _, crop_h = self._crop_dims(frame_w, frame_h)
        headroom = cfg.vertical_bias * crop_h

        if tracked:
            fx = sum(s.cx for s in tracked) / len(tracked)
            fy = sum(s.cy for s in tracked) / len(tracked)
            if body_center is not None:
                # Blend the body in horizontally for stability; vertical
                # composition is driven by the face alone so the headroom
                # bias below is not doubled by the (lower) torso center.
                fx = fx * cfg.face_weight + body_center[0] * (1 - cfg.face_weight)
            # Aim below the face so it sits above center (headroom).
            return fx, fy + headroom, cfg.position_speed
        if body_center is not None:
            # Torso-only tracking: the head is above the torso, so aim
            # upward to keep it in frame.
            return body_center[0], body_center[1] - headroom, cfg.position_speed
        # Nobody visible: drift gently back to frame center.
        return frame_w / 2.0, frame_h / 2.0, cfg.recenter_speed

    def _update_target_zoom(self, tracked: Sequence[Subject]) -> None:
        cfg = self.config
        # Subjects synthesized from pose keypoints may have no size estimate
        # (area_ratio == 0); they anchor position but should not drive zoom.
        ratios = [s.area_ratio for s in tracked if s.area_ratio > 0]
        if not ratios:
            # No face size available: hold briefly, then ease the target out
            # to a wide shot instead of freezing at a stale zoom level.
            self._frames_without_face += 1
            if self._frames_without_face > cfg.zoom_out_after:
                self._target_zoom += (cfg.zoom_out_level - self._target_zoom) * 0.1
            return
        self._frames_without_face = 0
        avg_ratio = sum(ratios) / len(ratios)
        self._face_ratio_history.append(avg_ratio)
        smoothed_ratio = sum(self._face_ratio_history) / len(self._face_ratio_history)

        # Face ratios are relative to the *input* frame; zooming in by Z makes
        # the face Z^2 larger in the output, hence the sqrt.
        desired = math.sqrt(cfg.target_face_ratio / smoothed_ratio)
        desired = max(cfg.zoom_min, min(cfg.zoom_max, desired))

        # Dead-zone: only move the target when the change is meaningful.
        if abs(desired - self._target_zoom) / max(self._target_zoom, 1e-6) > cfg.zoom_deadzone:
            self._target_zoom = desired

    @staticmethod
    def _smooth(current: float, target: float, alpha: float) -> float:
        return current + (target - current) * alpha

    @classmethod
    def _alpha(cls, speed: float, dt: float) -> float:
        """Convert a per-frame speed at 30 fps into a dt-correct factor.

        Repeated exponential smoothing composes as (1 - speed)^n, so raising
        to dt * 30 gives identical behavior at any frame rate.
        """
        return 1.0 - (1.0 - speed) ** (dt * cls.REFERENCE_FPS)

    def _crop_dims(self, frame_w: int, frame_h: int) -> Tuple[float, float]:
        """Current crop size: zoom applied to the largest aspect-matched fit."""
        out_w, out_h = self.config.output_size
        aspect = out_w / out_h

        # zoom == 1.0 means the crop is as large as possible while matching
        # the output aspect; zoom == 2.0 shows half that width (2x closer).
        if frame_w / frame_h > aspect:
            base_h = float(frame_h)
            base_w = base_h * aspect
        else:
            base_w = float(frame_w)
            base_h = base_w / aspect

        crop_w = min(base_w / max(self._zoom, 1e-6), frame_w)
        crop_h = min(base_h / max(self._zoom, 1e-6), frame_h)
        return crop_w, crop_h

    def _crop_rect(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Crop window sized by zoom, aspect-matched to output, clamped."""
        crop_w, crop_h = self._crop_dims(frame_w, frame_h)

        x = self._cx - crop_w / 2.0
        y = self._cy - crop_h / 2.0
        x = max(0.0, min(x, frame_w - crop_w))
        y = max(0.0, min(y, frame_h - crop_h))
        return int(round(x)), int(round(y)), int(round(crop_w)), int(round(crop_h))
