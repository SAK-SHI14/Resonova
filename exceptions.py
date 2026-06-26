"""
Babel — Custom exception hierarchy.

All pipeline-specific errors inherit from BabelError so callers can catch
the entire pipeline's errors with a single except clause if needed, while
still being able to catch stage-specific errors individually.

Usage:
    from babel.exceptions import TranscriptionError, TranslationError, ...
    raise TranscriptionError("Whisper returned empty transcript for: path/to/file.wav")
"""


class BabelError(Exception):
    """Base class for all Babel pipeline errors."""


class TranscriptionError(BabelError):
    """Raised when Whisper fails to transcribe audio."""


class TranslationError(BabelError):
    """Raised when IndicTrans2 fails to produce a valid translation."""


class VoiceCloningError(BabelError):
    """Raised when XTTS-v2 fails to synthesize cloned speech."""


class LipSyncError(BabelError):
    """Raised when Wav2Lip fails to produce a lip-synced video."""


class AudioExtractionError(BabelError):
    """Raised when audio cannot be extracted from a video file."""


class ProsodyError(BabelError):
    """Raised when prosody feature extraction fails."""


class EvaluationError(BabelError):
    """Raised when an evaluation metric cannot be computed."""
