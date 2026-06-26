"""
Unit tests for mimi.prosody
=============================
Run with: pytest tests/test_prosody.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock librosa in sys.modules BEFORE importing code that uses it
mock_librosa_obj = MagicMock()
sys.modules['librosa'] = mock_librosa_obj
sys.modules['librosa.onset'] = mock_librosa_obj.onset
sys.modules['librosa.feature'] = mock_librosa_obj.feature

from mimi.exceptions import ProsodyError
from mimi.prosody.conditioning import apply_prosody_conditioning
from mimi.prosody.extract import extract_prosody


@pytest.fixture(autouse=True)
def clean_librosa_mock():
    """Reset the mock librosa object before each test."""
    mock_librosa_obj.reset_mock()
    yield mock_librosa_obj


@pytest.fixture
def dummy_audio(tmp_path: Path) -> Path:
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"\x00" * 1020)
    return audio_file


class TestProsody:

    def test_extract_prosody_returns_valid_features(self, dummy_audio: Path):
        """extract_prosody must return a dictionary with all required metrics."""
        mock_librosa_obj.load.return_value = (np.zeros(16000), 16000)
        mock_librosa_obj.get_duration.return_value = 1.0
        mock_librosa_obj.yin.return_value = np.array([100.0, 110.0, 120.0])
        mock_librosa_obj.feature.rms.return_value = np.array([[0.1, 0.2, 0.1]])
        mock_librosa_obj.amplitude_to_db.return_value = np.array([-10.0, -5.0, -10.0])
        mock_librosa_obj.onset.onset_strength.return_value = np.array([0.1, 0.2])
        mock_librosa_obj.onset.onset_detect.return_value = [0.1, 0.5]

        features = extract_prosody(str(dummy_audio))

        assert isinstance(features, dict)
        assert "mean_pitch" in features
        assert "std_pitch" in features
        assert "mean_energy" in features
        assert "speaking_rate" in features
        assert "pause_ratio" in features
        assert "pitch_contour" in features
        assert "energy_contour" in features

        assert features["mean_pitch"] == 110.0
        assert features["speaking_rate"] == 2.0  # 2 onsets / 1.0s duration

    def test_extract_prosody_missing_file_raises_error(self):
        """extract_prosody must raise FileNotFoundError if file is missing."""
        with pytest.raises(FileNotFoundError):
            extract_prosody("nonexistent_file.wav")

    def test_extract_prosody_short_file_raises_error(self, dummy_audio: Path):
        """extract_prosody must raise ProsodyError if audio is under 0.1 seconds."""
        mock_librosa_obj.load.return_value = (np.zeros(100), 16000)
        mock_librosa_obj.get_duration.return_value = 0.05
        
        with pytest.raises(ProsodyError, match="too short"):
            extract_prosody(str(dummy_audio))

    @patch("mimi.prosody.conditioning.extract_prosody")
    @patch("subprocess.run")
    def test_apply_prosody_conditioning_runs_ffmpeg_volume(self, mock_run, mock_extract, tmp_path: Path):
        """apply_prosody_conditioning must scale volume using ffmpeg when energy levels mismatch."""
        orig_audio = tmp_path / "orig.wav"
        synth_audio = tmp_path / "synth.wav"
        out_audio = tmp_path / "out.wav"

        for f in [orig_audio, synth_audio]:
            f.write_bytes(b"\x00" * 10)

        # Mock RMS mismatch: original is 2x louder than synthesized (rms: 0.2 vs 0.1)
        mock_extract.side_effect = [
            {"mean_energy": 0.2, "pitch_contour": [], "energy_contour": []},  # original
            {"mean_energy": 0.1, "pitch_contour": [], "energy_contour": []},  # synthesized
        ]

        mock_run.return_value = MagicMock(returncode=0)

        # Simulate that ffmpeg successfully creates output
        def fake_run(*args, **kwargs):
            out_audio.write_bytes(b"\x00" * 10)
            return MagicMock(returncode=0)
        mock_run.side_effect = fake_run

        res = apply_prosody_conditioning(str(synth_audio), str(orig_audio), str(out_audio))

        assert Path(res).exists()
        # Verify scaling factor of 2.0 was passed to ffmpeg volume filter
        cmd = mock_run.call_args[0][0]
        filter_idx = cmd.index("-filter:a")
        assert cmd[filter_idx + 1] == "volume=2.0000"

    @patch("mimi.prosody.conditioning.extract_prosody")
    @patch("shutil.copy")
    def test_apply_prosody_conditioning_copies_if_rms_matched(self, mock_copy, mock_extract, tmp_path: Path):
        """apply_prosody_conditioning must copy file directly if energy levels match within 5%."""
        orig_audio = tmp_path / "orig.wav"
        synth_audio = tmp_path / "synth.wav"
        out_audio = tmp_path / "out.wav"

        for f in [orig_audio, synth_audio]:
            f.write_bytes(b"\x00" * 10)

        # Mismatched by less than 5% (rms: 0.102 vs 0.100)
        mock_extract.side_effect = [
            {"mean_energy": 0.102, "pitch_contour": [], "energy_contour": []},
            {"mean_energy": 0.100, "pitch_contour": [], "energy_contour": []},
        ]

        apply_prosody_conditioning(str(synth_audio), str(orig_audio), str(out_audio))
        mock_copy.assert_called_once_with(str(synth_audio), str(out_audio))
