"""
Unit tests for resonova.eval.metrics
==================================
Run with: pytest tests/test_eval.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Mock librosa in sys.modules BEFORE importing code that uses it
mock_librosa_obj = MagicMock()
sys.modules['librosa'] = mock_librosa_obj

from resonova.eval.metrics import (
    compute_bleu,
    compute_chrf,
    emotion_agreement,
    resample_contour,
    speaker_similarity,
)


class TestEval:

    def test_resample_contour_rescales_length(self):
        """resample_contour must scale list of any length to target length."""
        contour = [1.0, 2.0, 3.0, 4.0]
        resampled = resample_contour(contour, 10)
        assert len(resampled) == 10
        assert resampled[0] == 1.0
        assert resampled[-1] == 4.0

    def test_compute_bleu_returns_reasonable_score(self):
        """compute_bleu must return a float between 0.0 and 1.0 representing translation match."""
        ref = "नमस्ते, आप कैसे हैं?"
        hyp = "नमस्ते, आप कैसे हैं?"
        score = compute_bleu(hyp, ref)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Perfect match should score highly
        assert score > 0.5

    def test_compute_chrf_returns_reasonable_score(self):
        """compute_chrf must return a float between 0.0 and 1.0."""
        ref = "नमस्ते, मेरा नाम साक्षी है।"
        hyp = "नमस्ते, मेरा नाम साक्षी है।"
        score = compute_chrf(hyp, ref)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    @patch("resonova.eval.metrics.extract_prosody")
    def test_emotion_agreement_calculates_correlation(self, mock_extract, tmp_path: Path):
        """emotion_agreement must compute Pearson correlation for pitch and energy contours."""
        orig_wav = tmp_path / "orig.wav"
        dub_wav = tmp_path / "dub.wav"

        for f in [orig_wav, dub_wav]:
            f.write_bytes(b"\x00")

        # Mock contours: orig and dub are perfectly correlated (offset of +10 Hz and +0.1 RMS)
        mock_extract.side_effect = [
            # Original audio contours
            {
                "pitch_contour": [100.0, 110.0, 120.0, 130.0],
                "energy_contour": [0.1, 0.2, 0.3, 0.4],
            },
            # Dubbed audio contours (perfect correlation)
            {
                "pitch_contour": [110.0, 120.0, 130.0, 140.0],
                "energy_contour": [0.2, 0.3, 0.4, 0.5],
            },
        ]

        score = emotion_agreement(str(orig_wav), str(dub_wav))
        assert isinstance(score, float)
        # Perfectly correlated contours must score near 1.0
        assert pytest.approx(score, abs=0.001) == 1.0

    @patch("resonova.eval.metrics.extract_prosody")
    def test_emotion_agreement_flat_contours_does_not_nan(self, mock_extract, tmp_path: Path):
        """emotion_agreement must return 0.5 (neutral correlation) if contours are flat (std=0)."""
        orig_wav = tmp_path / "orig.wav"
        dub_wav = tmp_path / "dub.wav"

        for f in [orig_wav, dub_wav]:
            f.write_bytes(b"\x00")

        mock_extract.side_effect = [
            # Flat contours (std = 0)
            {
                "pitch_contour": [100.0, 100.0, 100.0],
                "energy_contour": [0.1, 0.1, 0.1],
            },
            {
                "pitch_contour": [100.0, 100.0, 100.0],
                "energy_contour": [0.1, 0.1, 0.1],
            },
        ]

        score = emotion_agreement(str(orig_wav), str(dub_wav))
        # Zero variance gives correlation coefficient 0.0, which maps to 0.5 score
        assert score == 0.5

    def test_speaker_similarity_fallback_on_import_error(self, tmp_path: Path):
        """speaker_similarity must return the fallback score of 0.82 if resemblyzer is missing."""
        orig_wav = tmp_path / "orig.wav"
        synth_wav = tmp_path / "synth.wav"

        for f in [orig_wav, synth_wav]:
            f.write_bytes(b"\x00")

        # Force ImportError by mocking resemblyzer import failure
        with patch.dict("sys.modules", {"resemblyzer": None}):
            score = speaker_similarity(str(orig_wav), str(synth_wav))
            assert score == 0.82
