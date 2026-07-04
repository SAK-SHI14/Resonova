"""
Unit tests for vaani.voice_cloning.clone_voice
===============================================
Run with:  pytest vaani/voice_cloning/tests/test_clone_voice.py -v

Smoke tests:
  - clone_voice() creates a non-empty output WAV file
  - Correct typed exceptions on bad inputs
  - Reference clip length check works
  - Output duration validation catches zero-duration files
"""

import wave
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from vaani.exceptions import VoiceCloningError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_wav(path: Path, duration_s: float, sample_rate: int = 16000) -> Path:
    """Write a mono WAV file of given duration and return the path."""
    n_samples = int(duration_s * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Simple sine wave data (audible, not silent)
        import math
        samples = [
            int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
            for i in range(n_samples)
        ]
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def long_reference_wav(tmp_path: Path) -> Path:
    """An 8-second WAV file — meets XTTS-v2 minimum requirement (≥6 s)."""
    return _create_wav(tmp_path / "reference_long.wav", duration_s=8.0)


@pytest.fixture
def short_reference_wav(tmp_path: Path) -> Path:
    """A 3-second WAV file — below XTTS-v2 minimum requirement."""
    return _create_wav(tmp_path / "reference_short.wav", duration_s=3.0)


@pytest.fixture
def mock_tts_model(tmp_path):
    """
    Mock TTS model that writes a real WAV file when tts_to_file() is called.
    Simulates successful synthesis.
    """
    def _fake_tts_to_file(text, speaker_wav, language, file_path):
        # Write a 2-second WAV to the output path to simulate real synthesis
        _create_wav(Path(file_path), duration_s=2.0)

    model = MagicMock()
    model.tts_to_file.side_effect = _fake_tts_to_file
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCloneVoice:

    def test_creates_output_wav_file(
        self, long_reference_wav: Path, mock_tts_model, tmp_path: Path
    ):
        """clone_voice() must create a non-empty WAV file at output_path."""
        output_path = str(tmp_path / "output" / "cloned.wav")

        with patch("vaani.voice_cloning.clone_voice._load_tts_model", return_value=mock_tts_model):
            from vaani.voice_cloning.clone_voice import clone_voice
            result = clone_voice(
                reference_audio_path=str(long_reference_wav),
                text="This is a test sentence.",
                language="en",
                output_path=output_path,
            )

        assert Path(result).exists(), f"Output file must exist at: {result}"
        assert Path(result).stat().st_size > 0, "Output file must be non-empty"

    def test_returns_absolute_path(
        self, long_reference_wav: Path, mock_tts_model, tmp_path: Path
    ):
        """clone_voice() must return an absolute path string."""
        output_path = str(tmp_path / "cloned.wav")

        with patch("vaani.voice_cloning.clone_voice._load_tts_model", return_value=mock_tts_model):
            from vaani.voice_cloning.clone_voice import clone_voice
            result = clone_voice(
                reference_audio_path=str(long_reference_wav),
                text="Test.",
                language="en",
                output_path=output_path,
            )

        assert Path(result).is_absolute(), f"Return value must be absolute path, got: {result}"

    def test_missing_reference_raises_file_not_found_error(self, tmp_path: Path):
        """clone_voice() must raise FileNotFoundError for missing reference."""
        from vaani.voice_cloning.clone_voice import clone_voice
        with pytest.raises(FileNotFoundError):
            clone_voice(
                reference_audio_path="/nonexistent/reference.wav",
                text="Test.",
                language="en",
                output_path=str(tmp_path / "out.wav"),
            )

    def test_short_reference_raises_value_error(self, short_reference_wav: Path, tmp_path: Path):
        """clone_voice() must raise ValueError when reference clip is < 6 seconds."""
        from vaani.voice_cloning.clone_voice import clone_voice
        with pytest.raises(ValueError, match="too short"):
            clone_voice(
                reference_audio_path=str(short_reference_wav),
                text="Test.",
                language="en",
                output_path=str(tmp_path / "out.wav"),
            )

    def test_empty_text_raises_value_error(self, long_reference_wav: Path, tmp_path: Path):
        """clone_voice() must raise ValueError for empty text input."""
        from vaani.voice_cloning.clone_voice import clone_voice
        with pytest.raises(ValueError):
            clone_voice(
                reference_audio_path=str(long_reference_wav),
                text="",
                language="en",
                output_path=str(tmp_path / "out.wav"),
            )

    def test_tts_runtime_error_raises_voice_cloning_error(
        self, long_reference_wav: Path, tmp_path: Path
    ):
        """clone_voice() must wrap TTS runtime errors in VoiceCloningError."""
        mock_model = MagicMock()
        mock_model.tts_to_file.side_effect = RuntimeError("CUDA out of memory")

        with patch("vaani.voice_cloning.clone_voice._load_tts_model", return_value=mock_model):
            from vaani.voice_cloning.clone_voice import clone_voice
            with pytest.raises(VoiceCloningError, match="synthesis failed"):
                clone_voice(
                    reference_audio_path=str(long_reference_wav),
                    text="Test.",
                    language="en",
                    output_path=str(tmp_path / "out.wav"),
                )

    def test_creates_parent_directories_automatically(
        self, long_reference_wav: Path, mock_tts_model, tmp_path: Path
    ):
        """clone_voice() must create nested output directories if they don't exist."""
        nested_output = str(tmp_path / "level1" / "level2" / "cloned.wav")

        with patch("vaani.voice_cloning.clone_voice._load_tts_model", return_value=mock_tts_model):
            from vaani.voice_cloning.clone_voice import clone_voice
            result = clone_voice(
                reference_audio_path=str(long_reference_wav),
                text="Test.",
                language="en",
                output_path=nested_output,
            )

        assert Path(result).exists(), "Output file must exist even in nested directories"


# ---------------------------------------------------------------------------
# Tests: _get_audio_duration()
# ---------------------------------------------------------------------------

class TestGetAudioDuration:

    def test_correct_duration_for_known_wav(self, tmp_path: Path):
        """_get_audio_duration() must return correct duration for a known WAV file."""
        wav_path = _create_wav(tmp_path / "test.wav", duration_s=5.0)

        from vaani.voice_cloning.clone_voice import _get_audio_duration
        duration = _get_audio_duration(wav_path)

        # Allow ±100ms tolerance due to integer sample count rounding
        assert abs(duration - 5.0) < 0.1, f"Expected ~5.0 s, got {duration:.2f} s"
