/**
 * The framing brain: pure math, no camera, no ML.
 *
 * Direct port of python/facestay/framer.py — keep the two in sync.
 * Decides where to crop given detected subjects; detection and pixel
 * shuffling live elsewhere.
 */

export interface Subject {
  cx: number;
  cy: number;
  /** Bounding box [x, y, w, h] in pixels. */
  bbox: [number, number, number, number];
  /** Fraction of the input frame area this subject occupies (0-1). */
  areaRatio: number;
}

export interface FramingConfig {
  outputSize: [number, number];
  /**
   * Per-frame smoothing factor at the 30 fps reference rate
   * (current += (target - current) * speed); converted with the measured
   * frame time so the feel is frame-rate independent. Lower = heavier.
   */
  positionSpeed: number;
  zoomSpeed: number;
  /**
   * Tracked face(s) should occupy ~this fraction of the output area.
   * ~0.10 gives head-and-shoulders; larger crops tighter on the face.
   */
  targetFaceRatio: number;
  zoomMin: number;
  zoomMax: number;
  /** Ignore zoom target changes smaller than this fraction. */
  zoomDeadzone: number;
  /** Moving-average window (frames) for face size. */
  faceSizeHistory: number;
  /** Face weight in the horizontal face/body blend. */
  faceWeight: number;
  /**
   * Headroom: shift the crop center down by this fraction of crop height so
   * faces sit above center (photographic upper third).
   */
  verticalBias: number;
  /** Track only the N largest faces (null = all). */
  maxSubjects: number | null;
  /** Drift-to-center speed when nothing is detected. */
  recenterSpeed: number;
  /** Frames without a face before the zoom target eases out wide. */
  zoomOutAfter: number;
  zoomOutLevel: number;
}

export const DEFAULT_CONFIG: FramingConfig = {
  outputSize: [1280, 720],
  positionSpeed: 0.08,
  zoomSpeed: 0.05,
  targetFaceRatio: 0.1,
  zoomMin: 0.6,
  zoomMax: 3.0,
  zoomDeadzone: 0.05,
  faceSizeHistory: 15,
  faceWeight: 0.7,
  verticalBias: 0.12,
  maxSubjects: null,
  recenterSpeed: 0.02,
  zoomOutAfter: 30,
  zoomOutLevel: 1.0,
};

const REFERENCE_FPS = 30;

export interface FramingState {
  crop: [number, number, number, number];
  zoom: number;
  target: [number, number];
  trackedSubjects: Subject[];
  bodyCenter: [number, number] | null;
}

export class FramingEngine {
  readonly config: FramingConfig;
  state: FramingState;

  private cx: number | null = null;
  private cy: number | null = null;
  private zoom = 1.0;
  private targetZoom = 1.0;
  private faceRatioHistory: number[] = [];
  private framesWithoutFace = 0;

  constructor(config?: Partial<FramingConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.state = {
      crop: [0, 0, 0, 0],
      zoom: 1.0,
      target: [0, 0],
      trackedSubjects: [],
      bodyCenter: null,
    };
  }

  /**
   * Advance one frame. Returns crop rect [x, y, w, h] in pixels.
   * dt is the elapsed time since the previous frame in seconds; when
   * omitted, the 30 fps reference interval is assumed.
   */
  update(
    frameW: number,
    frameH: number,
    subjects: Subject[],
    bodyCenter: [number, number] | null = null,
    dt: number = 1 / REFERENCE_FPS,
  ): [number, number, number, number] {
    const cfg = this.config;

    if (this.cx === null || this.cy === null) {
      this.cx = frameW / 2;
      this.cy = frameH / 2;
    }

    const tracked = this.selectSubjects(subjects);
    const [tx, ty, positionSpeed] = this.targetCenter(
      frameW,
      frameH,
      tracked,
      bodyCenter,
    );
    this.updateTargetZoom(tracked);

    const posAlpha = alphaForDt(positionSpeed, dt);
    this.cx += (tx - this.cx) * posAlpha;
    this.cy += (ty - this.cy) * posAlpha;
    this.zoom += (this.targetZoom - this.zoom) * alphaForDt(cfg.zoomSpeed, dt);

    const crop = this.cropRect(frameW, frameH);
    this.state = {
      crop,
      zoom: this.zoom,
      target: [tx, ty],
      trackedSubjects: tracked,
      bodyCenter,
    };
    return crop;
  }

  reset(): void {
    this.cx = null;
    this.cy = null;
    this.zoom = 1.0;
    this.targetZoom = 1.0;
    this.faceRatioHistory = [];
    this.framesWithoutFace = 0;
  }

  // ------------------------------------------------------------------
  private selectSubjects(subjects: Subject[]): Subject[] {
    const chosen = [...subjects].sort((a, b) => b.areaRatio - a.areaRatio);
    if (this.config.maxSubjects !== null) {
      return chosen.slice(0, this.config.maxSubjects);
    }
    return chosen;
  }

  private targetCenter(
    frameW: number,
    frameH: number,
    tracked: Subject[],
    bodyCenter: [number, number] | null,
  ): [number, number, number] {
    const cfg = this.config;
    const [, cropH] = this.cropDims(frameW, frameH);
    const headroom = cfg.verticalBias * cropH;

    if (tracked.length > 0) {
      let fx = tracked.reduce((s, t) => s + t.cx, 0) / tracked.length;
      const fy = tracked.reduce((s, t) => s + t.cy, 0) / tracked.length;
      if (bodyCenter) {
        // Blend the body in horizontally for stability; vertical composition
        // is driven by the face alone so the headroom bias is not doubled.
        fx = fx * cfg.faceWeight + bodyCenter[0] * (1 - cfg.faceWeight);
      }
      // Aim below the face so it sits above center (headroom).
      return [fx, fy + headroom, cfg.positionSpeed];
    }
    if (bodyCenter) {
      // Torso-only tracking: the head is above the torso, so aim upward.
      return [bodyCenter[0], bodyCenter[1] - headroom, cfg.positionSpeed];
    }
    // Nobody visible: drift gently back to frame center.
    return [frameW / 2, frameH / 2, cfg.recenterSpeed];
  }

  private updateTargetZoom(tracked: Subject[]): void {
    const cfg = this.config;
    // Subjects without a size estimate anchor position but not zoom.
    const ratios = tracked.map((t) => t.areaRatio).filter((r) => r > 0);
    if (ratios.length === 0) {
      // No face size available: hold briefly, then ease out to a wide shot
      // instead of freezing at a stale zoom level.
      this.framesWithoutFace++;
      if (this.framesWithoutFace > cfg.zoomOutAfter) {
        this.targetZoom += (cfg.zoomOutLevel - this.targetZoom) * 0.1;
      }
      return;
    }
    this.framesWithoutFace = 0;
    const avgRatio = ratios.reduce((a, b) => a + b, 0) / ratios.length;

    this.faceRatioHistory.push(avgRatio);
    if (this.faceRatioHistory.length > cfg.faceSizeHistory) {
      this.faceRatioHistory.shift();
    }
    const smoothed =
      this.faceRatioHistory.reduce((a, b) => a + b, 0) /
      this.faceRatioHistory.length;

    // Zooming in by Z makes the face Z^2 larger, hence the sqrt.
    let desired = Math.sqrt(cfg.targetFaceRatio / smoothed);
    desired = Math.max(cfg.zoomMin, Math.min(cfg.zoomMax, desired));

    if (
      Math.abs(desired - this.targetZoom) / Math.max(this.targetZoom, 1e-6) >
      cfg.zoomDeadzone
    ) {
      this.targetZoom = desired;
    }
  }

  /** Current crop size: zoom applied to the largest aspect-matched fit. */
  private cropDims(frameW: number, frameH: number): [number, number] {
    const [outW, outH] = this.config.outputSize;
    const aspect = outW / outH;

    let baseW: number;
    let baseH: number;
    if (frameW / frameH > aspect) {
      baseH = frameH;
      baseW = baseH * aspect;
    } else {
      baseW = frameW;
      baseH = baseW / aspect;
    }

    return [
      Math.min(baseW / Math.max(this.zoom, 1e-6), frameW),
      Math.min(baseH / Math.max(this.zoom, 1e-6), frameH),
    ];
  }

  private cropRect(
    frameW: number,
    frameH: number,
  ): [number, number, number, number] {
    const [cropW, cropH] = this.cropDims(frameW, frameH);

    let x = (this.cx as number) - cropW / 2;
    let y = (this.cy as number) - cropH / 2;
    x = Math.max(0, Math.min(x, frameW - cropW));
    y = Math.max(0, Math.min(y, frameH - cropH));
    return [Math.round(x), Math.round(y), Math.round(cropW), Math.round(cropH)];
  }
}

/**
 * Convert a per-frame speed at 30 fps into a dt-correct factor. Repeated
 * exponential smoothing composes as (1 - speed)^n, so raising to dt * 30
 * gives identical behavior at any frame rate.
 */
function alphaForDt(speed: number, dt: number): number {
  return 1 - Math.pow(1 - speed, dt * REFERENCE_FPS);
}
