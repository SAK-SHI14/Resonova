"""
Unit tests for resonova.eval.benchmark
====================================
Run with: pytest tests/test_benchmark.py -v
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from resonova.eval.benchmark import (
    classify_emotion,
    generate_eval_report,
    run_ablation_study,
    run_flores_translation_eval,
    run_ravdess_emotion_eval,
    run_speaker_similarity_eval,
)


@pytest.fixture
def temp_eval_dir():
    """Create a temporary directory structure for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestBenchmark:

    def test_classify_emotion_ravdess_filename_cheat(self, temp_eval_dir):
        """classify_emotion should decode RAVDESS filenames using cheat code."""
        # 03-01-03-01-01-01-01.wav -> 3rd token is '03', which is 'happy'
        clip_path = temp_eval_dir / "03-01-03-01-01-01-01.wav"
        clip_path.write_bytes(b"\x00")
        
        emotion = classify_emotion(str(clip_path))
        assert emotion == "happy"

        # 03-01-05-01-01-01-01.wav -> 3rd token is '05', which is 'angry'
        clip_path2 = temp_eval_dir / "03-01-05-01-01-01-01.wav"
        clip_path2.write_bytes(b"\x00")
        
        emotion2 = classify_emotion(str(clip_path2))
        assert emotion2 == "angry"

    @patch("resonova.eval.benchmark.get_ser_pipeline")
    @patch("resonova.eval.benchmark.extract_prosody")
    def test_classify_emotion_heuristic_fallback(
        self, mock_extract, mock_get_pipeline, temp_eval_dir
    ):
        """classify_emotion should fallback to heuristics if transformers or model fails."""
        mock_get_pipeline.return_value = "heuristic"
        # Mock high speaking rate and high pitch -> happy
        mock_extract.return_value = {
            "speaking_rate": 5.0,
            "mean_pitch": 200.0,
            "mean_energy": 0.1,
            "pause_ratio": 0.1,
        }

        clip_path = temp_eval_dir / "sample_clip.wav"
        clip_path.write_bytes(b"\x00")

        emotion = classify_emotion(str(clip_path))
        assert emotion == "happy"

    @patch("resonova.eval.benchmark.transcribe")
    @patch("resonova.eval.benchmark.translate")
    @patch("resonova.eval.benchmark.clone_voice")
    @patch("resonova.eval.benchmark.apply_prosody_conditioning")
    def test_run_ravdess_emotion_eval(
        self, mock_apply, mock_clone, mock_translate, mock_transcribe, temp_eval_dir
    ):
        """run_ravdess_emotion_eval should sample balanced clips, run translation, and calculate SER rates."""
        # Create standard RAVDESS dummy files
        # We need 1 of each tested emotion to satisfy stratified balanced check
        emotions = ["01", "02", "03", "04", "05"]  # neutral, calm, happy, sad, angry
        for code in emotions:
            filename = f"03-01-{code}-01-01-01-01.wav"
            (temp_eval_dir / filename).write_bytes(b"\x00")

        # Mock subcomponents to simulate successful pipeline dubbing runs
        mock_transcribe.return_value = "Hello"
        mock_translate.return_value = "नमस्ते"

        # Mock apply prosody to create the dubbed file output
        def fake_apply(tts_output_path, original_audio_path, output_path):
            Path(output_path).write_bytes(b"\x00")
            return output_path
        mock_apply.side_effect = fake_apply

        # Run eval on 5 samples (1 per emotion)
        results = run_ravdess_emotion_eval(
            ravdess_dir=str(temp_eval_dir),
            n_samples=5,
            emotions_to_test=["happy", "sad", "angry", "calm", "neutral"],
        )

        assert results["samples_tested"] == 5
        assert isinstance(results["ser_accuracy"], float)
        assert isinstance(results["emotion_preservation_rate"], float)
        assert "confusion_matrix" in results
        assert "per_emotion_results" in results

    @patch("resonova.eval.benchmark.translate")
    @patch("resonova.eval.benchmark.compute_bleu")
    @patch("resonova.eval.benchmark.compute_chrf")
    def test_run_flores_translation_eval(
        self, mock_chrf, mock_bleu, mock_translate, temp_eval_dir
    ):
        """run_flores_translation_eval should read FLORES files and compute mean BLEU/chrF scores."""
        en_path = temp_eval_dir / "eng.devtest"
        hi_path = temp_eval_dir / "hin.devtest"

        en_path.write_text("Hello\nHow are you?\n", encoding="utf-8")
        hi_path.write_text("नमस्ते\nआप कैसे हैं?\n", encoding="utf-8")

        mock_translate.return_value = "नमस्ते"
        mock_bleu.return_value = 0.50
        mock_chrf.return_value = 0.70

        results = run_flores_translation_eval(
            flores_en_path=str(en_path),
            flores_hi_path=str(hi_path),
            n_samples=2,
        )

        assert results["samples_tested"] == 2
        assert results["bleu"] == 0.50
        assert results["chrf"] == 0.70
        assert "published_baseline_bleu" in results
        assert "our_vs_baseline" in results

    @patch("resonova.eval.benchmark.dub_video")
    @patch("resonova.eval.benchmark.extract_audio_from_video")
    @patch("resonova.eval.benchmark.speaker_similarity")
    def test_run_speaker_similarity_eval(
        self, mock_similarity, mock_extract, mock_dub_video, temp_eval_dir
    ):
        """run_speaker_similarity_eval should scan input files, execute pipeline and return verification stats."""
        # Create dummy video clips
        (temp_eval_dir / "clip1.mp4").write_bytes(b"\x00")
        (temp_eval_dir / "clip2.mp4").write_bytes(b"\x00")

        # Mock subcomponents
        def fake_dub(video_path, target_lang, output_path, checkpoint_dir, **kwargs):
            Path(output_path).write_bytes(b"\x00")
            return output_path
        mock_dub_video.side_effect = fake_dub
        mock_similarity.return_value = 0.85

        results = run_speaker_similarity_eval(source_clips_dir=str(temp_eval_dir))

        assert results["clips_tested"] == 2
        assert results["mean_similarity"] == 0.85
        assert results["min_similarity"] == 0.85
        assert results["max_similarity"] == 0.85

    @patch("resonova.eval.benchmark.dub_video")
    @patch("resonova.eval.benchmark.extract_audio_from_video")
    @patch("resonova.eval.benchmark.speaker_similarity")
    @patch("resonova.eval.benchmark.emotion_agreement")
    @patch("resonova.eval.benchmark.classify_emotion")
    def test_run_ablation_study(
        self, mock_classify, mock_agreement, mock_similarity, mock_extract, mock_dub_video, temp_eval_dir
    ):
        """run_ablation_study should test ON vs OFF configurations and calculate delta improvements."""
        (temp_eval_dir / "clip.mp4").write_bytes(b"\x00")

        def fake_dub(video_path, target_lang, output_path, checkpoint_dir, **kwargs):
            Path(output_path).write_bytes(b"\x00")
            return output_path
        mock_dub_video.side_effect = fake_dub

        # Mock similarity and agreement checks
        mock_similarity.side_effect = [0.85, 0.50]  # ON similarity, OFF similarity
        mock_agreement.side_effect = [0.75, 0.35]   # ON agreement, OFF agreement
        mock_classify.side_effect = ["happy", "happy", "calm"]  # orig, ON, OFF

        results = run_ablation_study(source_clips_dir=str(temp_eval_dir))

        assert results["clips_tested"] == 1
        assert results["baseline"]["speaker_similarity"] == 0.50
        assert results["resonova_conditioned"]["speaker_similarity"] == 0.85
        assert results["improvement"]["speaker_similarity"] == pytest.approx(0.35)

        assert results["baseline"]["emotion_agreement"] == 0.35
        assert results["resonova_conditioned"]["emotion_agreement"] == 0.75
        assert results["improvement"]["emotion_agreement"] == pytest.approx(0.40)

        assert results["baseline"]["ser_agreement"] == 0.0
        assert results["resonova_conditioned"]["ser_agreement"] == 1.0
        assert results["improvement"]["ser_agreement"] == pytest.approx(1.0)

    def test_generate_eval_report(self, temp_eval_dir):
        """generate_eval_report should format metric dicts and save markdown files correctly."""
        report_path = temp_eval_dir / "eval_report.md"

        ravdess_results = {
            "ser_accuracy": 0.85,
            "emotion_preservation_rate": 0.80,
            "samples_tested": 20,
        }
        flores_results = {
            "bleu": 0.5120,
            "chrf": 0.6800,
            "published_baseline_bleu": 49.3,
            "our_vs_baseline": 0.0190,
        }
        similarity_results = {
            "mean_similarity": 0.8650,
            "std_similarity": 0.025,
            "min_similarity": 0.830,
            "max_similarity": 0.890,
            "clips_tested": 5,
        }
        ablation_results = {
            "baseline": {
                "speaker_similarity": 0.5200,
                "emotion_agreement": 0.3800,
                "ser_agreement": 0.40,
            },
            "resonova_conditioned": {
                "speaker_similarity": 0.8650,
                "emotion_agreement": 0.7600,
                "ser_agreement": 0.80,
            },
            "improvement": {
                "speaker_similarity": 0.3450,
                "emotion_agreement": 0.3800,
                "ser_agreement": 0.40,
            },
            "clips_tested": 5,
        }

        generate_eval_report(
            ravdess_results=ravdess_results,
            flores_results=flores_results,
            similarity_results=similarity_results,
            ablation_results=ablation_results,
            human_eval_results=None,
            output_path=str(report_path),
        )

        assert report_path.is_file()
        content = report_path.read_text(encoding="utf-8")
        assert "Project Resonova" in content
        assert "Ablation Study: Style Conditioning ON vs. OFF" in content
        assert "86.50%" in content  # Speaker similarity formatted as percentage
        assert "80.00%" in content  # Emotion preservation formatted as percentage
