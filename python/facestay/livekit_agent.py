"""LiveKit server-side integration: republish a participant's video, framed.

Subscribes to a video track, runs each frame through AutoFramer, and
publishes the result as a new track ("<identity>-framed"). Useful for
recordings, AI-analyzed feeds, or when you don't want client-side processing.

Requires:  pip install livekit livekit-agents

Typical use inside a LiveKit agent entrypoint:

    from facestay.livekit_agent import FramedRepublisher

    async def entrypoint(ctx: JobContext):
        await ctx.connect()
        republisher = FramedRepublisher(ctx.room, output_size=(1280, 720))
        republisher.start()   # follows video tracks as they appear

The client-side (browser) integration in js/ is usually preferable for live
calls: it saves a server round-trip and scales with zero server CPU.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Tuple

import numpy as np

try:
    from livekit import rtc
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The LiveKit SDK is required for this module: pip install livekit"
    ) from exc

from .autoframer import AutoFramer


class FramedRepublisher:
    """Follows remote video tracks and republishes auto-framed versions."""

    def __init__(
        self,
        room: "rtc.Room",
        output_size: Tuple[int, int] = (1280, 720),
        fps: int = 30,
    ) -> None:
        self._room = room
        self._output_size = output_size
        self._fps = fps
        self._tasks: list[asyncio.Task] = []
        self._source: Optional[rtc.VideoSource] = None

    def start(self) -> None:
        @self._room.on("track_subscribed")
        def _on_track(track: rtc.Track, pub, participant) -> None:
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                task = asyncio.create_task(self._process_track(track, participant))
                self._tasks.append(task)
                task.add_done_callback(self._tasks.remove)

        # Pick up tracks that were published before we connected.
        for participant in self._room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track and pub.track.kind == rtc.TrackKind.KIND_VIDEO:
                    task = asyncio.create_task(
                        self._process_track(pub.track, participant)
                    )
                    self._tasks.append(task)
                    task.add_done_callback(self._tasks.remove)

    async def stop(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    async def _process_track(self, track: "rtc.Track", participant) -> None:
        w, h = self._output_size
        framer = AutoFramer(output_size=self._output_size)
        source = rtc.VideoSource(w, h)
        out_track = rtc.LocalVideoTrack.create_video_track(
            f"{participant.identity}-framed", source
        )
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_CAMERA
        await self._room.local_participant.publish_track(out_track, options)

        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.BGRA)
        try:
            async for event in stream:
                frame = event.frame
                bgra = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                    frame.height, frame.width, 4
                )
                out_bgr = framer.process(bgra[:, :, :3])

                out_bgra = np.empty((h, w, 4), dtype=np.uint8)
                out_bgra[:, :, :3] = out_bgr
                out_bgra[:, :, 3] = 255
                source.capture_frame(
                    rtc.VideoFrame(w, h, rtc.VideoBufferType.BGRA, out_bgra.tobytes())
                )
        except asyncio.CancelledError:
            pass
        finally:
            framer.release()
            await stream.aclose()
