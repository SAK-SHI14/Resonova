"""
Mimi — Custom exception hierarchy.

All pipeline-specific errors inherit from MimiError so callers can catch
the entire pipeline's errors with a single except clause if needed, while
still being able to catch stage-specific errors individually.

Usage:
    from mimi.exceptions import TranscriptionError, TranslationError, ...
    raise TranscriptionError("Whisper returned empty transcript for: path/to/file.wav")
"""


class MimiError(Exception):
    """Base class for all Mimi pipeline errors."""


class TranscriptionError(MimiError):
    """Raised when Whisper fails to transcribe audio."""


class TranslationError(MimiError):
    """Raised when IndicTrans2 fails to produce a valid translation."""


class VoiceCloningError(MimiError):
    """Raised when XTTS-v2 fails to synthesize cloned speech."""


class LipSyncError(MimiError):
    """Raised when Wav2Lip fails to produce a lip-synced video."""


class AudioExtractionError(MimiError):
    """Raised when audio cannot be extracted from a video file."""


class ProsodyError(MimiError):
    """Raised when prosody feature extraction fails."""


class EvaluationError(MimiError):
    """Raised when an evaluation metric cannot be computed."""
