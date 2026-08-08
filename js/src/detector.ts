/**
 * Face detection via MediaPipe Tasks (WASM). Isolated from the framing math
 * so the model or provider can be swapped without touching framer.ts.
 */

import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";
import type { Subject } from "./framer";

const WASM_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

export interface DetectorOptions {
  /** Override to self-host the WASM bundle instead of the CDN. */
  wasmBasePath?: string;
  /** Override to self-host the model file. */
  modelAssetPath?: string;
  minDetectionConfidence?: number;
}

export class SubjectDetector {
  private detector: FaceDetector;

  private constructor(detector: FaceDetector) {
    this.detector = detector;
  }

  static async create(options: DetectorOptions = {}): Promise<SubjectDetector> {
    const fileset = await FilesetResolver.forVisionTasks(
      options.wasmBasePath ?? WASM_CDN,
    );
    const detector = await FaceDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: options.modelAssetPath ?? MODEL_URL },
      runningMode: "VIDEO",
      minDetectionConfidence: options.minDetectionConfidence ?? 0.5,
    });
    return new SubjectDetector(detector);
  }

  /** Detect faces in a video frame. Coordinates are returned in pixels. */
  detect(
    frame: TexImageSource,
    frameW: number,
    frameH: number,
    timestampMs: number,
  ): Subject[] {
    const result = this.detector.detectForVideo(frame, timestampMs);
    const subjects: Subject[] = [];
    for (const det of result.detections) {
      const box = det.boundingBox;
      if (!box || box.width <= 0 || box.height <= 0) continue;
      subjects.push({
        cx: box.originX + box.width / 2,
        cy: box.originY + box.height / 2,
        bbox: [box.originX, box.originY, box.width, box.height],
        areaRatio: (box.width * box.height) / (frameW * frameH),
      });
    }
    return subjects;
  }

  close(): void {
    this.detector.close();
  }
}
