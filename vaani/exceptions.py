"""
Vaani — Custom exception hierarchy.

All pipeline-specific errors inherit from VaaniError so callers can catch
the entire pipeline's errors with a single except clause if needed, while
still being able to catch stage-specific errors individually.

Usage:
    from vaani.exceptions import TranscriptionError, TranslationError, ...
    raise TranscriptionError("Whisper returned empty transcript for: path/to/file.wav")
"""


class VaaniError(Exception):
    """Base class for all Vaani pipeline errors."""


class TranscriptionError(VaaniError):
    """Raised when Whisper fails to transcribe audio."""


class TranslationError(VaaniError):
    """Raised when IndicTrans2 fails to produce a valid translation."""


class VoiceCloningError(VaaniError):
    """Raised when XTTS-v2 fails to synthesize cloned speech."""


class LipSyncError(VaaniError):
    """Raised when Wav2Lip fails to produce a lip-synced video."""


class AudioExtractionError(VaaniError):
    """Raised when audio cannot be extracted from a video file."""


class ProsodyError(VaaniError):
    """Raised when prosody feature extraction fails."""


class EvaluationError(VaaniError):
    """Raised when an evaluation metric cannot be computed."""
