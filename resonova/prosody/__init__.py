"""
Resonova — Prosody and Emotion Preservation Sub-package.
"""

from resonova.prosody.extract import extract_prosody
from resonova.prosody.conditioning import apply_prosody_conditioning

__all__ = ["extract_prosody", "apply_prosody_conditioning"]
