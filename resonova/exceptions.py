"""
Resonova — Custom exception hierarchy.

All pipeline-specific errors inherit from ResonovaError so callers can catch
the entire pipeline's errors with a single except clause if needed, while
still being able to catch stage-specific errors individually.

Usage:
    from resonova.exceptions import TranscriptionError, TranslationError, ...
    raise TranscriptionError("Whisper returned empty transcript for: path/to/file.wav")
"""


class ResonovaError(Exception):
    """Base class for all Resonova pipeline errors."""


class TranscriptionError(ResonovaError):
    """Raised when Whisper fails to transcribe audio."""


class TranslationError(ResonovaError):
    """Raised when IndicTrans2 fails to produce a valid translation."""


class VoiceCloningError(ResonovaError):
    """Raised when XTTS-v2 fails to synthesize cloned speech."""


class LipSyncError(ResonovaError):
    """Raised when Wav2Lip fails to produce a lip-synced video."""


class AudioExtractionError(ResonovaError):
    """Raised when audio cannot be extracted from a video file."""


class ProsodyError(ResonovaError):
    """Raised when prosody feature extraction fails."""


class EvaluationError(ResonovaError):
    """Raised when an evaluation metric cannot be computed."""
