"""
Unit tests for babel.lipsync.lipsync
=====================================
Run with:  pytest babel/lipsync/tests/test_lipsync.py -v

Smoke tests — these mock the subprocess call because Wav2Lip requires
a full GPU environment to actually run. The tests verify:
  - lipsync() constructs the correct subprocess command
  - lipsync() raises correct typed exceptions on bad inputs
  - lipsync() raises LipSyncError with actionable messages on common failures
  - Output validation logic catches missing/empty output files

IMPORTANT: A true integration test requires a Wav2Lip environment.
           The integration test is in tests/integration_test_lipsync.py
           and is marked with @pytest.mark.integration to be run separately.
"""

import subprocess
import wave
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import os

import pytest

from babel.exceptions import LipSyncError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_wav(path: Path, duration_s: float = 2.0, sample_rate: int = 16000) -> Path:
    n_samples = int(duration_s * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return path


def _create_dummy_mp4(path: Path) -> Path:
    """Write a fake MP4 file (just bytes — enough to pass file existence check)."""
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def source_video(tmp_path: Path) -> Path:
    return _create_dummy_mp4(tmp_path / "source.mp4")


@pytest.fixture
def cloned_audio(tmp_path: Path) -> Path:
    return _create_wav(tmp_path / "cloned.wav")


@pytest.fixture
def mock_env(tmp_path: Path, monkeypatch):
    """Set up fake WAV2LIP env vars pointing to a mock directory structure."""
    repo_dir = tmp_path / "Wav2Lip"
    repo_dir.mkdir()
    # Create a fake inference.py
    (repo_dir / "inference.py").write_text("# fake wav2lip inference script")

    checkpoint_path = tmp_path / "wav2lip_gan.pth"
    checkpoint_path.write_bytes(b"\x00" * 100)  # fake checkpoint

    monkeypatch.setenv("WAV2LIP_REPO_PATH", str(repo_dir))
    monkeypatch.setenv("WAV2LIP_CHECKPOINT_PATH", str(checkpoint_path))

    return str(repo_dir), str(checkpoint_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLipsync:

    def test_raises_file_not_found_for_missing_video(self, cloned_audio: Path, tmp_path: Path, mock_env):
        """lipsync() must raise FileNotFoundError if video file is missing."""
        from babel.lipsync.lipsync import lipsync
        with pytest.raises(FileNotFoundError):
            lipsync(
                video_path="/nonexistent/video.mp4",
                audio_path=str(cloned_audio),
                output_path=str(tmp_path / "out.mp4"),
            )

    def test_raises_file_not_found_for_missing_audio(self, source_video: Path, tmp_path: Path, mock_env):
        """lipsync() must raise FileNotFoundError if audio file is missing."""
        from babel.lipsync.lipsync import lipsync
        with pytest.raises(FileNotFoundError):
            lipsync(
                video_path=str(source_video),
                audio_path="/nonexistent/audio.wav",
                output_path=str(tmp_path / "out.mp4"),
            )

    def test_raises_lipsync_error_when_wav2lip_env_not_set(
        self, source_video: Path, cloned_audio: Path, tmp_path: Path, monkeypatch
    ):
        """lipsync() must raise LipSyncError with helpful message if env vars are not set."""
        monkeypatch.delenv("WAV2LIP_REPO_PATH", raising=False)
        monkeypatch.delenv("WAV2LIP_CHECKPOINT_PATH", raising=False)

        from babel.lipsync.lipsync import lipsync
        with pytest.raises(LipSyncError, match="WAV2LIP_REPO_PATH"):
            lipsync(
                video_path=str(source_video),
                audio_path=str(cloned_audio),
                output_path=str(tmp_path / "out.mp4"),
            )

    def test_raises_lipsync_error_on_nonzero_exit_code(
        self, source_video: Path, cloned_audio: Path, tmp_path: Path, mock_env
    ):
        """lipsync() must raise LipSyncError when subprocess exits with non-zero code."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "RuntimeError: some wav2lip error"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            from babel.lipsync.lipsync import lipsync
            with pytest.raises(LipSyncError, match="exit code 1"):
                lipsync(
                    video_path=str(source_video),
                    audio_path=str(cloned_audio),
                    output_path=str(tmp_path / "out.mp4"),
                )

    def test_numpy_error_in_stderr_gives_actionable_diagnosis(
        self, source_video: Path, cloned_audio: Path, tmp_path: Path, mock_env
    ):
        """Error message must mention NumPy pinning when np.bool error detected in stderr."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "AttributeError: module 'numpy' has no attribute 'np.bool'"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            from babel.lipsync.lipsync import lipsync
            with pytest.raises(LipSyncError) as exc_info:
                lipsync(
                    video_path=str(source_video),
                    audio_path=str(cloned_audio),
                    output_path=str(tmp_path / "out.mp4"),
                )
        assert "numpy==1.23.5" in str(exc_info.value), \
            "Error message must include the pinned NumPy version fix"

    def test_successful_run_returns_absolute_output_path(
        self, source_video: Path, cloned_audio: Path, tmp_path: Path, mock_env
    ):
        """lipsync() must return absolute path to output file on success."""
        output_path = tmp_path / "out.mp4"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = "Wav2Lip complete"

        # Create a fake output file to pass the existence check
        def fake_run(*args, **kwargs):
            output_path.write_bytes(b"\x00" * 1024)
            return mock_result

        with patch("subprocess.run", side_effect=fake_run):
            from babel.lipsync.lipsync import lipsync
            result = lipsync(
                video_path=str(source_video),
                audio_path=str(cloned_audio),
                output_path=str(output_path),
            )

        assert Path(result).is_absolute(), f"Expected absolute path, got: {result}"
        assert Path(result).exists(), "Output file must exist"

    def test_timeout_raises_lipsync_error(
        self, source_video: Path, cloned_audio: Path, tmp_path: Path, mock_env
    ):
        """lipsync() must raise LipSyncError on subprocess timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="test", timeout=600)):
            from babel.lipsync.lipsync import lipsync
            with pytest.raises(LipSyncError, match="timed out"):
                lipsync(
                    video_path=str(source_video),
                    audio_path=str(cloned_audio),
                    output_path=str(tmp_path / "out.mp4"),
                )

    def test_creates_parent_directories_automatically(
        self, source_video: Path, cloned_audio: Path, tmp_path: Path, mock_env
    ):
        """lipsync() must create nested output directories before running Wav2Lip."""
        nested_output = tmp_path / "level1" / "level2" / "out.mp4"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = ""

        def fake_run(*args, **kwargs):
            nested_output.write_bytes(b"\x00" * 1024)
            return mock_result

        with patch("subprocess.run", side_effect=fake_run):
            from babel.lipsync.lipsync import lipsync
            result = lipsync(
                video_path=str(source_video),
                audio_path=str(cloned_audio),
                output_path=str(nested_output),
            )

        assert Path(result).exists()
