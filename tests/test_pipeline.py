"""
Unit tests for resonova.pipeline.dub_video
========================================
Run with: pytest tests/test_pipeline.py -v
"""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from resonova.exceptions import (
    AudioExtractionError,
    ResonovaError,
    LipSyncError,
    TranscriptionError,
    TranslationError,
    VoiceCloningError,
)
from resonova.pipeline import dub_video, get_video_duration, time_stretch_audio


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    video_file = tmp_path / "input_video.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return video_file


@pytest.fixture
def mock_subcomponents():
    """Mock all model-related functions in the pipeline to prevent real loading/running."""
    with patch("resonova.pipeline.transcribe") as mock_transcribe, \
         patch("resonova.pipeline.translate") as mock_translate, \
         patch("resonova.pipeline.clone_voice") as mock_clone_voice, \
         patch("resonova.pipeline.lipsync") as mock_lipsync, \
         patch("resonova.pipeline.extract_audio_from_video") as mock_extract, \
         patch("resonova.pipeline.time_stretch_audio") as mock_stretch, \
         patch("resonova.pipeline.get_audio_duration") as mock_duration, \
         patch("resonova.pipeline.get_video_duration") as mock_v_duration, \
         patch("resonova.pipeline.unload_all_models") as mock_unload:

        # Default return values
        mock_transcribe.return_value = "Hello, this is a test."
        mock_translate.return_value = "नमस्ते, यह एक परीक्षण है।"
        mock_duration.side_effect = lambda path: 10.0 if "extracted" in str(path) else 12.0
        mock_v_duration.return_value = 10.0

        yield {
            "transcribe": mock_transcribe,
            "translate": mock_translate,
            "clone_voice": mock_clone_voice,
            "lipsync": mock_lipsync,
            "extract_audio": mock_extract,
            "time_stretch": mock_stretch,
            "get_audio_duration": mock_duration,
            "get_video_duration": mock_v_duration,
            "unload_all_models": mock_unload,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipeline:

    def test_dub_video_chains_all_components_successfully(
        self, dummy_video: Path, tmp_path: Path, mock_subcomponents
    ):
        """dub_video must call every component wrapper in the correct order."""
        out_video = tmp_path / "dubbed_output.mp4"
        ckpt_dir = tmp_path / "checkpoints"

        # Mock voice clone & lip-sync outputs to simulate successful generation
        def fake_clone(*args, **kwargs):
            (ckpt_dir / "cloned_audio_raw.wav").write_bytes(b"\x00" * 100)

        def fake_lipsync(video_path, audio_path, output_path, **kwargs):
            Path(output_path).write_bytes(b"\x00" * 200)

        mock_subcomponents["clone_voice"].side_effect = fake_clone
        mock_subcomponents["lipsync"].side_effect = fake_lipsync

        res_path = dub_video(
            video_path=str(dummy_video),
            target_lang="hin_Deva",
            output_path=str(out_video),
            checkpoint_dir=str(ckpt_dir),
            sync_strategy="time_stretch",
        )

        # Assert correct output path
        assert Path(res_path) == out_video.resolve()
        assert out_video.exists()

        # Verify correct sequential calls
        mock_subcomponents["extract_audio"].assert_called_once()
        mock_subcomponents["transcribe"].assert_called_once()
        mock_subcomponents["translate"].assert_called_once()
        mock_subcomponents["clone_voice"].assert_called_once()
        mock_subcomponents["time_stretch"].assert_called_once_with(
            str(ckpt_dir / "cloned_audio_raw.wav"),
            str(ckpt_dir / "cloned_audio_synced.wav"),
            1.2  # 12.0s cloned / 10.0s original = 1.2 speed ratio
        )
        mock_subcomponents["lipsync"].assert_called_once()

    def test_checkpoint_resumption_skips_completed_stages(
        self, dummy_video: Path, tmp_path: Path, mock_subcomponents
    ):
        """dub_video must load existing files from checkpoint_dir and skip finished steps."""
        out_video = tmp_path / "dubbed_output.mp4"
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Simulate that ASR and Translation have already completed in a prior run
        (ckpt_dir / "extracted_audio.wav").write_bytes(b"\x00" * 10)
        (ckpt_dir / "transcript.txt").write_text("Hello, this is a test.", encoding="utf-8")
        (ckpt_dir / "translated_text.txt").write_text("नमस्ते, यह एक परीक्षण है।", encoding="utf-8")

        # Mock voice clone & lip-sync outputs
        def fake_clone(*args, **kwargs):
            (ckpt_dir / "cloned_audio_raw.wav").write_bytes(b"\x00" * 100)

        def fake_lipsync(video_path, audio_path, output_path, **kwargs):
            Path(output_path).write_bytes(b"\x00" * 200)

        mock_subcomponents["clone_voice"].side_effect = fake_clone
        mock_subcomponents["lipsync"].side_effect = fake_lipsync

        dub_video(
            video_path=str(dummy_video),
            target_lang="hin_Deva",
            output_path=str(out_video),
            checkpoint_dir=str(ckpt_dir),
        )

        # Extraction, ASR, and Translation must be skipped
        mock_subcomponents["extract_audio"].assert_not_called()
        mock_subcomponents["transcribe"].assert_not_called()
        mock_subcomponents["translate"].assert_not_called()

        # Clone and subsequent stages must still run
        mock_subcomponents["clone_voice"].assert_called_once()
        mock_subcomponents["time_stretch"].assert_called_once()
        mock_subcomponents["lipsync"].assert_called_once()

    def test_sync_strategy_accept_drift_does_not_time_stretch(
        self, dummy_video: Path, tmp_path: Path, mock_subcomponents
    ):
        """If sync_strategy is 'accept_drift', time_stretch must not be called."""
        out_video = tmp_path / "dubbed_output.mp4"
        ckpt_dir = tmp_path / "checkpoints"

        # Simulate voice clone output
        def fake_clone(*args, **kwargs):
            (ckpt_dir / "cloned_audio_raw.wav").write_bytes(b"\x00" * 100)

        def fake_lipsync(video_path, audio_path, output_path, **kwargs):
            Path(output_path).write_bytes(b"\x00" * 200)

        mock_subcomponents["clone_voice"].side_effect = fake_clone
        mock_subcomponents["lipsync"].side_effect = fake_lipsync

        dub_video(
            video_path=str(dummy_video),
            target_lang="hin_Deva",
            output_path=str(out_video),
            checkpoint_dir=str(ckpt_dir),
            sync_strategy="accept_drift",
        )

        # time_stretch_audio must be bypassed
        mock_subcomponents["time_stretch"].assert_not_called()
        mock_subcomponents["lipsync"].assert_called_once()

    @patch("subprocess.run")
    def test_time_stretch_audio_chains_atempo_for_large_ratios(self, mock_run):
        """For extreme ratios outside [0.65, 1.5], time_stretch_audio uses smart
        padding/trimming instead of distorted extreme atempo chaining."""
        mock_run.return_value = MagicMock(returncode=0)

        # Test ratio = 4.5 (> 1.5 threshold) — should use silence-padding
        time_stretch_audio("in.wav", "out.wav", 4.5)
        cmd_high = mock_run.call_args[0][0]
        assert "-filter_complex" in cmd_high, (
            "Ratio 4.5 should use silence padding (-filter_complex with apad)"
        )
        assert any("apad" in str(arg) for arg in cmd_high), (
            "Expected 'apad' in ffmpeg args for ratio=4.5"
        )

        # Test ratio = 0.15 (< 0.65 threshold) — should use fade-trim
        time_stretch_audio("in.wav", "out.wav", 0.15)
        cmd_low = mock_run.call_args[0][0]
        assert "-filter:a" in cmd_low, (
            "Ratio 0.15 should use atrim+afade trimming"
        )
        filter_val = cmd_low[cmd_low.index("-filter:a") + 1]
        assert "atrim" in filter_val, (
            f"Expected 'atrim' in filter for ratio=0.15, got: {filter_val}"
        )

    @patch("shutil.copy")
    def test_time_stretch_audio_copies_file_directly_for_unit_ratio(self, mock_copy):
        """time_stretch_audio must copy file directly without subprocess call if ratio is close to 1.0."""
        time_stretch_audio("in.wav", "out.wav", 1.0001)
        mock_copy.assert_called_once_with("in.wav", "out.wav")
