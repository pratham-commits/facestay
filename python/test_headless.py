"""Headless smoke test: runs the full pipeline without a camera or display.

Feeds synthetic frames (with a real photo-like face pattern absent, so it
exercises detection, the drift-to-center path, cropping, and resizing) plus
a direct unit check of the framing math.

Run:  python test_headless.py
"""

import numpy as np

from facestay import AutoFramer, FramingEngine, Subject


def test_engine_math() -> None:
    engine = FramingEngine()
    # A face left-of-center, ~2% of frame area -> engine should move toward
    # it and zoom in over time.
    face = Subject(cx=400.0, cy=300.0, bbox=(350, 250, 100, 100), area_ratio=0.02)
    for _ in range(300):
        x, y, w, h = engine.update(1920, 1080, [face])
        assert 0 <= x and 0 <= y and x + w <= 1920 and y + h <= 1080, "crop out of bounds"

    state = engine.state
    cx = state.crop[0] + state.crop[2] / 2
    cy = state.crop[1] + state.crop[3] / 2
    assert abs(cx - 400) < 30, f"crop center x did not converge: {cx}"
    # Headroom: the crop center sits below the face by vertical_bias * crop_h,
    # so the face lands in the photographic upper third.
    expected_cy = 300 + engine.config.vertical_bias * state.crop[3]
    assert abs(cy - expected_cy) < 30, f"crop center y off: {cy} vs {expected_cy}"
    assert cy > 300, "face should sit above the crop center (headroom)"
    assert state.zoom > 1.5, f"expected zoom-in on small face, got {state.zoom}"

    # Face disappears -> drift back toward frame center and ease the zoom
    # out wide instead of freezing at the stale zoom level.
    for _ in range(600):
        engine.update(1920, 1080, [])
    cx = engine.state.crop[0] + engine.state.crop[2] / 2
    assert abs(cx - 960) < 40, f"did not recenter: {cx}"
    assert engine.state.zoom < 1.1, f"did not zoom out on loss: {engine.state.zoom}"

    # A subject with unknown size (pose head proxy) anchors position but
    # must not drive zoom: the aim point follows it while the zoom target
    # eases wide.
    engine.reset()
    proxy = Subject(cx=1500.0, cy=500.0, bbox=(1500, 500, 0, 0), area_ratio=0.0)
    for _ in range(300):
        engine.update(1920, 1080, [proxy])
    tx, _ = engine.state.target
    assert abs(tx - 1500) < 1, f"proxy should anchor the target x: {tx}"
    assert engine.state.zoom < 1.1, f"size-less proxy should not hold zoom: {engine.state.zoom}"
    print("engine math: OK")


def test_full_pipeline() -> None:
    framer = AutoFramer(output_size=(1280, 720))
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    for _ in range(10):
        out = framer.process(frame)
    assert out.shape == (720, 1280, 3), f"bad output shape: {out.shape}"
    framer.release()
    print("full pipeline (detectors + crop + resize): OK")


if __name__ == "__main__":
    test_engine_math()
    test_full_pipeline()
    print("All headless tests passed.")
