/**
 * Browser pipeline: turns a camera MediaStreamTrack into an auto-framed one.
 *
 * Two entry points:
 *   - createAutoFramedTrack(track): plain WebRTC / getUserMedia use.
 *   - FacestayProcessor: LiveKit TrackProcessor — pass it to
 *     videoTrack.setProcessor(...) and the published track is auto-framed.
 *
 * Uses Insertable Streams (MediaStreamTrackProcessor/Generator) on
 * Chrome/Edge, and falls back to canvas.captureStream() elsewhere.
 */

import { SubjectDetector, DetectorOptions } from "./detector";
import { FramingConfig, FramingEngine, FramingState } from "./framer";

export interface AutoFramerOptions {
  config?: Partial<FramingConfig>;
  detector?: DetectorOptions;
  /** Run detection every Nth frame (smoothing still runs every frame). */
  detectEvery?: number;
}

/** Shared crop-draw core used by both pipelines. */
class FramerCore {
  readonly engine: FramingEngine;
  private detector: SubjectDetector | null = null;
  private readonly detectEvery: number;
  private frameIndex = 0;
  private lastSubjects: ReturnType<SubjectDetector["detect"]> = [];
  private lastTimestampMs: number | null = null;
  private readonly detectorOptions: DetectorOptions;
  readonly canvas: OffscreenCanvas;
  private readonly ctx: OffscreenCanvasRenderingContext2D;

  constructor(options: AutoFramerOptions) {
    this.engine = new FramingEngine(options.config);
    this.detectEvery = Math.max(1, options.detectEvery ?? 2);
    this.detectorOptions = options.detector ?? {};
    const [w, h] = this.engine.config.outputSize;
    this.canvas = new OffscreenCanvas(w, h);
    const ctx = this.canvas.getContext("2d");
    if (!ctx) throw new Error("Could not create 2d canvas context");
    this.ctx = ctx;
  }

  async init(): Promise<void> {
    this.detector = await SubjectDetector.create(this.detectorOptions);
  }

  get state(): FramingState {
    return this.engine.state;
  }

  /** Detect + smooth + draw the crop onto the internal canvas. */
  render(
    source: TexImageSource,
    srcW: number,
    srcH: number,
    timestampMs: number,
  ): OffscreenCanvas {
    if (this.detector && this.frameIndex % this.detectEvery === 0) {
      this.lastSubjects = this.detector.detect(source, srcW, srcH, timestampMs);
    }
    this.frameIndex++;

    let dt = 1 / 30;
    if (this.lastTimestampMs !== null) {
      // Clamp against timestamp glitches and long stalls (tab hidden).
      dt = Math.min(Math.max((timestampMs - this.lastTimestampMs) / 1000, 1 / 120), 0.25);
    }
    this.lastTimestampMs = timestampMs;

    const [x, y, cw, ch] = this.engine.update(
      srcW,
      srcH,
      this.lastSubjects,
      null,
      dt,
    );
    const [outW, outH] = this.engine.config.outputSize;
    this.ctx.drawImage(source as CanvasImageSource, x, y, cw, ch, 0, 0, outW, outH);
    return this.canvas;
  }

  destroy(): void {
    this.detector?.close();
    this.detector = null;
  }
}

// ---------------------------------------------------------------------------
// Plain WebRTC entry point
// ---------------------------------------------------------------------------

export interface AutoFramedTrack {
  track: MediaStreamTrack;
  /** Live framing state for debug UIs (crop rect, zoom, faces). */
  getState: () => FramingState;
  stop: () => void;
}

/**
 * Wrap any camera MediaStreamTrack; the returned track is the auto-framed
 * feed, usable anywhere a MediaStreamTrack is (RTCPeerConnection, <video>).
 */
export async function createAutoFramedTrack(
  inputTrack: MediaStreamTrack,
  options: AutoFramerOptions = {},
): Promise<AutoFramedTrack> {
  const core = new FramerCore(options);
  await core.init();

  if (supportsInsertableStreams()) {
    const track = buildInsertableStreamsPipeline(inputTrack, core);
    return {
      track,
      getState: () => core.state,
      stop: () => {
        track.stop();
        core.destroy();
      },
    };
  }

  const fallback = buildCanvasPipeline(inputTrack, core);
  return {
    track: fallback.track,
    getState: () => core.state,
    stop: () => {
      fallback.stop();
      core.destroy();
    },
  };
}

function supportsInsertableStreams(): boolean {
  return (
    typeof MediaStreamTrackProcessor !== "undefined" &&
    typeof MediaStreamTrackGenerator !== "undefined"
  );
}

function buildInsertableStreamsPipeline(
  inputTrack: MediaStreamTrack,
  core: FramerCore,
): MediaStreamTrack {
  const processor = new MediaStreamTrackProcessor({
    track: inputTrack as MediaStreamVideoTrack,
  });
  const generator = new MediaStreamTrackGenerator({ kind: "video" });

  const transform = new TransformStream<VideoFrame, VideoFrame>({
    transform: (frame, controller) => {
      const w = frame.displayWidth;
      const h = frame.displayHeight;
      const ts = frame.timestamp / 1000;
      const canvas = core.render(frame, w, h, ts);
      const outFrame = new VideoFrame(canvas, { timestamp: frame.timestamp });
      frame.close();
      controller.enqueue(outFrame);
    },
  });

  processor.readable
    .pipeThrough(transform)
    .pipeTo(generator.writable)
    .catch(() => {
      /* pipeline torn down (track stopped) */
    });
  return generator as unknown as MediaStreamTrack;
}

function buildCanvasPipeline(
  inputTrack: MediaStreamTrack,
  core: FramerCore,
  fps = 30,
): { track: MediaStreamTrack; stop: () => void } {
  // Safari/Firefox fallback: play the track in a hidden <video>, draw crops
  // to a visible-DOM canvas, and capture that canvas as a stream.
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.srcObject = new MediaStream([inputTrack]);
  void video.play();

  const [outW, outH] = core.engine.config.outputSize;
  const outCanvas = document.createElement("canvas");
  outCanvas.width = outW;
  outCanvas.height = outH;
  const outCtx = outCanvas.getContext("2d");
  if (!outCtx) throw new Error("Could not create 2d canvas context");

  let running = true;
  const draw = () => {
    if (!running) return;
    if (video.readyState >= 2 && video.videoWidth > 0) {
      const rendered = core.render(
        video,
        video.videoWidth,
        video.videoHeight,
        performance.now(),
      );
      outCtx.drawImage(rendered, 0, 0);
    }
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);

  const stream = outCanvas.captureStream(fps);
  const track = stream.getVideoTracks()[0];
  return {
    track,
    stop: () => {
      running = false;
      track.stop();
      video.srcObject = null;
    },
  };
}

// ---------------------------------------------------------------------------
// LiveKit TrackProcessor
// ---------------------------------------------------------------------------

// Structural types matching livekit-client's TrackProcessor interface, so
// this package compiles without depending on livekit-client.
interface VideoProcessorOptions {
  kind: unknown;
  track: MediaStreamTrack;
  element?: HTMLMediaElement;
}

/**
 * LiveKit track processor. Usage in your app:
 *
 *   import { FacestayProcessor } from "facestay-js";
 *
 *   const processor = new FacestayProcessor();
 *   await videoTrack.setProcessor(processor);
 *   await room.localParticipant.publishTrack(videoTrack);
 *
 * Everyone in the room receives the auto-framed feed. Pass
 * `showProcessedStreamLocally: true` (LiveKit option) so the local
 * self-view shows the framed video too.
 */
export class FacestayProcessor {
  name = "facestay";
  processedTrack?: MediaStreamTrack;

  private options: AutoFramerOptions;
  private framed: AutoFramedTrack | null = null;

  constructor(options: AutoFramerOptions = {}) {
    this.options = options;
  }

  async init(opts: VideoProcessorOptions): Promise<void> {
    this.framed = await createAutoFramedTrack(opts.track, this.options);
    this.processedTrack = this.framed.track;
  }

  async restart(opts: VideoProcessorOptions): Promise<void> {
    this.framed?.stop();
    this.framed = await createAutoFramedTrack(opts.track, this.options);
    this.processedTrack = this.framed.track;
  }

  async destroy(): Promise<void> {
    this.framed?.stop();
    this.framed = null;
    this.processedTrack = undefined;
  }

  /** Live framing state (crop rect, zoom, detected faces) for debug UIs. */
  getState(): FramingState | null {
    return this.framed?.getState() ?? null;
  }
}
