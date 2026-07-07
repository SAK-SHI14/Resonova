"""
Unit tests for vaani.app.report_card and vaani.app.app
=====================================================
Run with: pytest tests/test_report_card.py -v
"""

from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from vaani.app.report_card import compute_preservation_score, generate_report_card
from vaani.app.app import run_vaani_pipeline, create_app


class TestReportCard:

    def test_compute_preservation_score_perfect_match(self):
        """compute_preservation_score must return 100 for perfectly matched original and dubbed metrics."""
        pros_orig = {
            "emotion_label": "excited",
            "f0_mean": 200.0,
            "energy_mean": 0.08,
        }
        pros_dub = {
            "emotion_label": "excited",
            "f0_mean": 200.0,
            "energy_mean": 0.08,
        }
        score = compute_preservation_score(pros_orig, pros_dub)
        assert score == 100.0

    def test_compute_preservation_score_unmatched_emotion(self):
        """compute_preservation_score must deduct emotion score (40 points) if labels do not match."""
        pros_orig = {
            "emotion_label": "excited",
            "f0_mean": 200.0,
            "energy_mean": 0.08,
        }
        pros_dub = {
            "emotion_label": "calm",  # Mismatch
            "f0_mean": 200.0,
            "energy_mean": 0.08,
        }
        score = compute_preservation_score(pros_orig, pros_dub)
        # Should be 60.0 (0 for emotion, 30 for pitch, 30 for energy)
        assert score == 60.0

    def test_compute_preservation_score_mismatched_pitch_and_energy(self):
        """compute_preservation_score must reflect scaling differences in pitch and energy proximity."""
        pros_orig = {
            "emotion_label": "excited",
            "f0_mean": 200.0,
            "energy_mean": 0.08,
        }
        pros_dub = {
            "emotion_label": "excited",
            "f0_mean": 100.0,  # 50% match (15 points)
            "energy_mean": 0.04,  # 50% match (15 points)
        }
        score = compute_preservation_score(pros_orig, pros_dub)
        # Should be 70.0 (40 + 15 + 15)
        assert score == 70.0

    def test_generate_report_card_returns_pil_image(self):
        """generate_report_card must render layout successfully and return a PIL Image object."""
        pros_orig = {
            "emotion_label": "excited",
            "emotion_confidence": 0.90,
            "f0_mean": 220.0,
            "f0_std": 15.0,
            "energy_mean": 0.09,
            "energy_std": 0.01,
            "speaking_rate": 4.5,
            "pitch_contour": [200.0, 210.0, 220.0, 230.0],
        }
        pros_dub = {
            "emotion_label": "excited",
            "emotion_confidence": 0.88,
            "f0_mean": 210.0,
            "f0_std": 12.0,
            "energy_mean": 0.08,
            "energy_std": 0.01,
            "speaking_rate": 4.3,
            "pitch_contour": [195.0, 205.0, 210.0, 220.0],
        }

        img = generate_report_card(
            prosody_original=pros_orig,
            prosody_dubbed=pros_dub,
            speaker_similarity=0.8650,
            processing_time_seconds=10.5,
            clip_duration_seconds=5.0,
        )

        assert isinstance(img, Image.Image)
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_generate_report_card_handles_missing_contours_gracefully(self):
        """generate_report_card must draw a placeholder block and return a PIL Image if pitch contours are missing."""
        pros_orig = {
            "emotion_label": "calm",
            "f0_mean": 120.0,
            "speaking_rate": 3.2,
        }
        pros_dub = {
            "emotion_label": "calm",
            "f0_mean": 115.0,
            "speaking_rate": 3.1,
        }

        img = generate_report_card(
            prosody_original=pros_orig,
            prosody_dubbed=pros_dub,
            speaker_similarity=0.8650,
            processing_time_seconds=5.2,
            clip_duration_seconds=4.0,
        )

        assert isinstance(img, Image.Image)


class TestAppInterface:

    @patch("vaani.app.app.dub_video")
    @patch("vaani.app.app.extract_prosody")
    @patch("vaani.app.app.classify_emotion")
    @patch("vaani.app.app.generate_report_card")
    @patch("vaani.app.app.get_video_duration")
    def test_run_vaani_pipeline_orchestration(
        self, mock_v_dur, mock_generate, mock_classify, mock_extract, mock_dub, tmp_path
    ):
        """run_vaani_pipeline must call pipeline stage components and assemble returns under Gradio wrappers."""
        video_input = tmp_path / "source.mp4"
        video_input.write_bytes(b"\x00")

        # Mock stages
        mock_dub.return_value = str(tmp_path / "source_dubbed.mp4")
        mock_extract.return_value = {
            "mean_pitch": 180.0,
            "std_pitch": 10.0,
            "mean_energy": 0.05,
            "std_energy": 0.005,
            "speaking_rate": 4.0,
            "pitch_contour": [100.0, 110.0],
        }
        mock_classify.return_value = "happy"
        mock_generate.return_value = MagicMock(spec=Image.Image)
        mock_v_dur.return_value = 8.5

        # Create progress tracker
        progress_mock = MagicMock()

        # Run
        res = run_vaani_pipeline(
            video_file=str(video_input),
            target_language="Hindi (हिंदी)",
            progress=progress_mock,
        )

        # Assert returns matches Gradio mapping output
        assert len(res) == 6
        assert res[0] == str(video_input)
        assert res[1] == str(tmp_path / "source_dubbed.mp4")
        assert res[3] == ""  # Mock empty transcript if files not created on disk
        assert "Complete" in res[5]

        # Verify progress tracking triggered calls
        assert progress_mock.call_count >= 1

    def test_create_app_returns_blocks_instance(self):
        """create_app must compile the Gradio dashboard interface blocks successfully."""
        app = create_app()
        assert app is not None
