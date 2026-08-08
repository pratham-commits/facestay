import { createAutoFramedTrack } from "../src/index";

const statusEl = document.getElementById("status") as HTMLDivElement;
const statsEl = document.getElementById("stats") as HTMLDivElement;
const rawVideo = document.getElementById("raw") as HTMLVideoElement;
const framedVideo = document.getElementById("framed") as HTMLVideoElement;

async function start() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1920 }, height: { ideal: 1080 } },
  });
  rawVideo.srcObject = stream;

  const framed = await createAutoFramedTrack(stream.getVideoTracks()[0], {
    config: { outputSize: [1280, 720] },
    detectEvery: 2,
  });
  framedVideo.srcObject = new MediaStream([framed.track]);
  statusEl.textContent = "Running";

  setInterval(() => {
    const s = framed.getState();
    statsEl.textContent =
      `zoom ${s.zoom.toFixed(2)}x | faces ${s.trackedSubjects.length} | ` +
      `crop [${s.crop.join(", ")}]`;
  }, 250);
}

start().catch((err) => {
  statusEl.textContent = `Failed to start: ${err.message ?? err}`;
  console.error(err);
});
