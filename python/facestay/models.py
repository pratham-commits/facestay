"""Download-and-cache helper for MediaPipe task model files.

MediaPipe's Tasks API loads models from disk. On first use we fetch the
official Google-hosted models into a local cache (override the location with
the FACESTAY_CACHE environment variable).
"""

from __future__ import annotations

import os
import shutil
import ssl
import tempfile
import urllib.request
from pathlib import Path

FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def _cache_dir() -> Path:
    override = os.environ.get("FACESTAY_CACHE")
    if override:
        path = Path(override)
    else:
        path = Path.home() / ".cache" / "facestay"
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "facestay-models"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _ssl_context() -> ssl.SSLContext:
    # macOS python.org installs often lack system CA certs; prefer certifi's
    # bundle when available (it is a transitive dependency of mediapipe).
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def get_model(url: str) -> str:
    """Return a local path to the model, downloading it on first use."""
    target = _cache_dir() / url.rsplit("/", 1)[-1]
    if not target.exists():
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, context=_ssl_context()) as resp:
            with open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh)
        tmp.replace(target)
    return str(target)
