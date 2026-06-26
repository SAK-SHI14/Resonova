"""
Unit tests for mimi.asr.transcribe
====================================
Run with:  pytest mimi/asr/tests/test_transcribe.py -v

These are smoke tests:
  - Verify transcribe() runs without crashing on known-good input
  - Verify output is non-empty
  - Verify typed exceptions are raised on bad input (not bare Exception)
  - Verify timestamped variant also works

NOTE ON TEST AUDIO:
  Tests use a tiny programmatically-generated WAV (1 kHz tone + silence).
  This produces garbage transcription content, but that is intentional —
  the smoke test goal is "does the pipeline run without crashing", not
  "does it produce a specific string".

  For a real quality check, run manually against one of your recorded clips.
"""

import os
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mimi.exceptions import TranscriptionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_silent_wav(path: Path, duration_s: float = 1.0, sample_rate: int = 16000) -> Path:
    """Write a silent mono WAV file to *path* and return the path."""
    n_samples = int(duration_s * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """A 2-second silent WAV file for testing."""
    return _create_silent_wav(tmp_path / "silent.wav", duration_s=2.0)


@pytest.fixture
def mock_whisper_model():
    """A mock whisper model that returns a fixed transcript."""
    model = MagicMock()
    model.transcribe.return_value = {
        "text": "This is a mock transcription result.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.5,
                "text": "This is a mock transcription result.",
                "tokens": [1, 2, 3],
                "avg_logprob": -0.3,
                "no_speech_prob": 0.01,
            }
        ],
    }
    return model


# ---------------------------------------------------------------------------
# Tests: transcribe()
# ---------------------------------------------------------------------------

class TestTranscribe:

    def test_returns_non_empty_string(self, silent_wav: Path, mock_whisper_model):
        """transcribe() must return a non-empty string on valid audio."""
        with patch("mimi.asr.transcribe._load_model", return_value=mock_whisper_model):
            from mimi.asr.transcribe import transcribe
            result = transcribe(str(silent_wav))

        assert isinstance(result, str), "Return type must be str"
        assert len(result) > 0, "Returned transcript must be non-empty"

    def test_file_not_found_raises_correct_exception(self):
        """transcribe() must raise FileNotFoundError — not bare Exception — for missing files."""
        from mimi.asr.transcribe import transcribe
        with pytest.raises(FileNotFoundError):
            transcribe("/nonexistent/path/audio.wav")

    def test_empty_transcript_raises_transcription_error(self, silent_wav: Path):
        """transcribe() must raise TranscriptionError if Whisper returns empty text."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "   ", "segments": []}

        with patch("mimi.asr.transcribe._load_model", return_value=mock_model):
            from mimi.asr.transcribe import transcribe
            with pytest.raises(TranscriptionError, match="empty transcript"):
                transcribe(str(silent_wav))

    def test_whisper_runtime_error_raises_transcription_error(self, silent_wav: Path):
        """transcribe() must wrap Whisper runtime errors in TranscriptionError."""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("CUDA out of memory")

        with patch("mimi.asr.transcribe._load_model", return_value=mock_model):
            from mimi.asr.transcribe import transcribe
            with pytest.raises(TranscriptionError, match="failed"):
                transcribe(str(silent_wav))

    def test_directory_path_raises_transcription_error(self, tmp_path: Path):
        """transcribe() must raise TranscriptionError when given a directory path."""
        from mimi.asr.transcribe import transcribe
        with pytest.raises(TranscriptionError, match="directory"):
            transcribe(str(tmp_path))

    def test_result_is_stripped(self, silent_wav: Path):
        """transcribe() must return a stripped string (no leading/trailing whitespace)."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  hello world  ", "segments": []}

        with patch("mimi.asr.transcribe._load_model", return_value=mock_model):
            from mimi.asr.transcribe import transcribe
            result = transcribe(str(silent_wav))

        assert result == "hello world", f"Expected stripped result, got: '{result}'"

    def test_model_size_param_passed_to_load_model(self, silent_wav: Path, mock_whisper_model):
        """transcribe() must pass the model_size argument to _load_model."""
        with patch("mimi.asr.transcribe._load_model", return_value=mock_whisper_model) as mock_load:
            from mimi.asr.transcribe import transcribe
            transcribe(str(silent_wav), model_size="large-v3")

        mock_load.assert_called_once_with("large-v3")


# ---------------------------------------------------------------------------
# Tests: transcribe_with_timestamps()
# ---------------------------------------------------------------------------

class TestTranscribeWithTimestamps:

    def test_returns_list(self, silent_wav: Path, mock_whisper_model):
        """transcribe_with_timestamps() must return a list."""
        with patch("mimi.asr.transcribe._load_model", return_value=mock_whisper_model):
            from mimi.asr.transcribe import transcribe_with_timestamps
            result = transcribe_with_timestamps(str(silent_wav))

        assert isinstance(result, list), "Return type must be list"

    def test_segments_have_required_keys(self, silent_wav: Path, mock_whisper_model):
        """Each segment dict must contain 'start', 'end', and 'text' keys."""
        with patch("mimi.asr.transcribe._load_model", return_value=mock_whisper_model):
            from mimi.asr.transcribe import transcribe_with_timestamps
            segments = transcribe_with_timestamps(str(silent_wav))

        for seg in segments:
            assert "start" in seg, f"Segment missing 'start': {seg}"
            assert "end" in seg, f"Segment missing 'end': {seg}"
            assert "text" in seg, f"Segment missing 'text': {seg}"

    def test_file_not_found_raises_file_not_found_error(self):
        """transcribe_with_timestamps() must raise FileNotFoundError for missing files."""
        from mimi.asr.transcribe import transcribe_with_timestamps
        with pytest.raises(FileNotFoundError):
            transcribe_with_timestamps("/nonexistent/path/audio.wav")
