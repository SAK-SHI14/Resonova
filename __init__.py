"""Babel pipeline — top-level package."""

from babel.pipeline import dub_video, unload_all_models

__version__ = "0.1.0"
__author__ = "Sakshi Verma"
__description__ = (
    "Babel: Emotion-preserving AI dubbing and voice-cloned translation pipeline. "
    "English → Hindi (extensible) using Whisper, IndicTrans2, XTTS-v2, and Wav2Lip."
)

__all__ = ["dub_video", "unload_all_models"]

