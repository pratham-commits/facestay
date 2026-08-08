# Facestay

Your face stays. The camera does the work.

Smooth digital pan-tilt-zoom auto-framing for any video pipeline. The
physical webcam never moves — Facestay detects people, then smoothly pans
and zooms a crop window to keep them perfectly framed, exactly like Apple's
Center Stage (but without the Apple tax).

Built for video conferencing and interview platforms. One framing algorithm,
three ways to use it:

| Surface | Where it runs | Use case |
| --- | --- | --- |
| `python/` — `AutoFramer` | Your backend / any Python pipeline | Frame-in, frame-out processing; recordings; LiveKit agents |
| `js/` — `FacestayProcessor` | The user's browser | LiveKit / plain WebRTC apps; zero-latency, saves bandwidth |
| `python/vcam.py` | User's machine, as a virtual webcam | Zoom, Meet, any third-party app — no integration needed |

## How it works

1. **Detect**: MediaPipe face detection (BlazeFace) finds faces; body pose is
   the fallback when a face is not visible. Pose head keypoints fill in when
   the face detector misses.
2. **Target**: face centers are averaged; auto-zoom aims for a head-and-shoulders
   composition (~10% face area) with a 15-frame moving average and a
   dead-zone to prevent zoom "breathing". A headroom bias keeps faces at
   the photographic upper third, and losing the subject eases the shot
   wide instead of freezing.
3. **Smooth**: position and zoom ease toward their targets with exponential
   smoothing (`current += (target - current) * speed`, converted with the
   measured frame time so the feel is frame-rate independent) — this is
   what makes the motion feel like a camera operator instead of jumpy
   cropping.
4. **Crop**: the window is clamped to frame bounds, cropped, and resized
   (Lanczos) to the output resolution.

The framing math lives in one pure module per language
(`python/facestay/framer.py`, `js/src/framer.ts`) with no camera or ML
imports, so the two stay in lockstep and porting elsewhere is trivial.

## Python

```bash
cd python
pip install -r requirements.txt   # or: pip install -e .
python demo.py                    # webcam demo with debug window
```

Library use — frames in, frames out, bring your own I/O:

```python
from facestay import AutoFramer

framer = AutoFramer(output_size=(1280, 720))
while True:
    frame = ...                  # BGR numpy array from anywhere
    out = framer.process(frame)  # smoothly framed output
```

Tuning (all optional):

```python
from facestay import AutoFramer, FramingConfig

config = FramingConfig(
    position_speed=0.08,     # lower = heavier/slower camera
    zoom_speed=0.05,
    target_face_ratio=0.10,  # how much of the frame a face should fill
    vertical_bias=0.12,      # headroom: face sits this far above center
    zoom_min=0.6, zoom_max=3.0,
    max_subjects=None,       # or N to track only the N largest faces
)
framer = AutoFramer(output_size=(1280, 720), config=config,
                    detect_every=2)   # detect every 2nd frame to halve CPU
```

Demo keyboard controls: `Q` quit, `+`/`-` position speed, `[`/`]` zoom
speed, `R` reset, `D` debug window, `F` fullscreen, `1-9`/`0` face limit.

### Virtual camera (Zoom / Meet / anything)

```bash
pip install pyvirtualcam        # macOS: install OBS once for the backend
python vcam.py
```

Select the virtual camera as your webcam in any app.

### LiveKit server-side agent

```python
from facestay.livekit_agent import FramedRepublisher

async def entrypoint(ctx):
    await ctx.connect()
    FramedRepublisher(ctx.room, output_size=(1280, 720)).start()
```

## Browser (LiveKit / WebRTC)

```bash
cd js
npm install
npm run demo     # side-by-side raw vs framed demo (Vite)
```

LiveKit integration — two lines in your publish flow:

```ts
import { FacestayProcessor } from "facestay-js";

const processor = new FacestayProcessor();
await videoTrack.setProcessor(processor);
// publish as usual; everyone receives the auto-framed feed
```

Plain WebRTC (no LiveKit):

```ts
import { createAutoFramedTrack } from "facestay-js";

const stream = await navigator.mediaDevices.getUserMedia({ video: true });
const framed = await createAutoFramedTrack(stream.getVideoTracks()[0]);
peerConnection.addTrack(framed.track);   // or attach to a <video>
```

Uses Insertable Streams on Chrome/Edge (zero-copy) and a canvas fallback on
Safari/Firefox. Face detection runs on MediaPipe's official WASM build.

## Notes

- Capture at the highest resolution you can (1080p+) and output smaller
  (720p): the zoom headroom is what makes framing look good.
- Pan range is limited by the physical camera's field of view — if someone
  walks past the edge of the sensor's view, no software can follow them.
- The pose model adds CPU cost; pass `use_pose_fallback=False` to
  `AutoFramer` if your subjects always face the camera.
