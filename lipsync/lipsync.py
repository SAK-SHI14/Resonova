"""
Babel — Lip-Sync Module
=========================
Wrapper around Wav2Lip for audio-driven lip synchronization.

Model: Wav2Lip GAN (wav2lip_gan.pth)  — open-source, runs fully locally
Repository: https://github.com/Rudrabha/Wav2Lip

⚠️  DEPENDENCY WARNING — READ BEFORE USING ⚠️
=================================================
Wav2Lip is the HIGHEST-RISK component in this pipeline for dependency conflicts.
It was designed for an older ecosystem and does NOT pip-install cleanly.

REQUIRED SETUP STEPS (see notes.md for troubleshooting):
  1. Clone the Wav2Lip repository:
     git clone https://github.com/Rudrabha/Wav2Lip

  2. Download the pretrained checkpoint wav2lip_gan.pth:
     https://github.com/Rudrabha/Wav2Lip/releases

  3. Install the face detection sub-library (specific version!):
     pip install face-alignment==1.3.5
     # If face_detection (batch_face_detection) is also needed:
     pip install git+https://github.com/hhj1897/face_detection.git

  4. Set environment variables:
     WAV2LIP_REPO_PATH=/path/to/Wav2Lip
     WAV2LIP_CHECKPOINT_PATH=/path/to/wav2lip_gan.pth

KNOWN WORKING DEPENDENCY VERSIONS (as of Phase 1 testing):
  torch==1.12.1, torchvision==0.13.1, face-alignment==1.3.5,
  numpy==1.23.5, librosa==0.8.1, opencv-python==4.5.5.64
  See notes.md for full troubleshooting log.

GPU Memory (T4 reference):
  - Wav2Lip : ~2 GB VRAM (fits comfortably on T4)
  - Primary risk: dependency conflicts, NOT VRAM

Integration Pattern:
  This module calls Wav2Lip via subprocess (its inference.py script)
  because Wav2Lip was not designed as a pip-installable library.
  The subprocess approach avoids import-level dependency conflicts
  while keeping Babel's own dependency tree clean.

Known Limitations (documented, not hidden):
  - Wav2Lip produces visible artifacts on fast head movement or profile angles.
    Best results on near-frontal, relatively still video.
  - Lip-sync quality depends on video resolution (≥720p recommended).
  - Very long clips (>90 s) should be chunked — Wav2Lip uses significant
    RAM for frame buffering on long sequences.
  - The GAN checkpoint (wav2lip_gan.pth) produces sharper visuals than
    the non-GAN checkpoint but can introduce occasional texture artifacts.

Usage:
    from babel.lipsync.lipsync import lipsync
    output_video = lipsync(
        video_path="samples/source_video.mp4",
        audio_path="outputs/cloned_hindi.wav",
        output_path="outputs/dubbed_video.mp4",
    )
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from babel.exceptions import LipSyncError
from babel.logger import get_logger

logger = get_logger(__name__)


def _resolve_env_paths() -> tuple[str, str]:
    """
    Resolve Wav2Lip repo and checkpoint paths from environment variables.

    Returns:
        Tuple of (wav2lip_repo_path, wav2lip_checkpoint_path).

    Raises:
        LipSyncError: If required env vars are not set.
    """
    repo_path = os.environ.get("WAV2LIP_REPO_PATH", "")
    checkpoint_path = os.environ.get("WAV2LIP_CHECKPOINT_PATH", "")

    if not repo_path:
        raise LipSyncError(
            "WAV2LIP_REPO_PATH environment variable is not set.\n"
            "Set it to the absolute path of the cloned Wav2Lip repository:\n"
            "  export WAV2LIP_REPO_PATH=/path/to/Wav2Lip\n"
            "See .env.example for configuration details."
        )
    if not Path(repo_path).is_dir():
        raise LipSyncError(
            f"WAV2LIP_REPO_PATH does not point to a valid directory: '{repo_path}'.\n"
            "Clone Wav2Lip with: git clone https://github.com/Rudrabha/Wav2Lip"
        )

    if not checkpoint_path:
        raise LipSyncError(
            "WAV2LIP_CHECKPOINT_PATH environment variable is not set.\n"
            "Download wav2lip_gan.pth from:\n"
            "  https://github.com/Rudrabha/Wav2Lip/releases\n"
            "Then set: export WAV2LIP_CHECKPOINT_PATH=/path/to/wav2lip_gan.pth"
        )
    if not Path(checkpoint_path).is_file():
        raise LipSyncError(
            f"Wav2Lip checkpoint not found at: '{checkpoint_path}'.\n"
            "Download wav2lip_gan.pth and update WAV2LIP_CHECKPOINT_PATH."
        )

    return repo_path, checkpoint_path


def _validate_video(video_path: Path) -> None:
    """
    Validate that the video file exists and has a face in the first frame.

    Args:
        video_path: Path to the video file.

    Raises:
        FileNotFoundError: If the file does not exist.
        LipSyncError: If the file is not a video or has no detectable face.
    """
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: '{video_path}'. "
            "Ensure the source video path is correct."
        )
    if not video_path.is_file():
        raise LipSyncError(f"Expected a file but got a directory: '{video_path}'.")

    # Check for supported video extension
    supported_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if video_path.suffix.lower() not in supported_extensions:
        logger.warning(
            "Video file has unusual extension '%s'. "
            "Supported formats: %s. Proceeding anyway.",
            video_path.suffix,
            supported_extensions,
        )


def _validate_audio(audio_path: Path) -> None:
    """
    Validate that the audio file exists.

    Args:
        audio_path: Path to the audio file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: '{audio_path}'. "
            "Ensure the cloned voice audio has been generated before calling lipsync()."
        )


def lipsync(
    video_path: str,
    audio_path: str,
    output_path: str,
    wav2lip_repo_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    resize_factor: int = 1,
    fps: float = 25.0,
    pads: tuple = (0, 10, 0, 0),
    nosmooth: bool = False,
) -> str:
    """
    Apply Wav2Lip lip synchronization: replace the video's lip movements
    to match the provided audio track.

    Args:
        video_path:         Path to the source video (with the speaker's face).
        audio_path:         Path to the replacement audio (cloned translated speech).
        output_path:        Path where the lip-synced output video will be saved.
        wav2lip_repo_path:  Absolute path to the cloned Wav2Lip repository.
                            Defaults to WAV2LIP_REPO_PATH env var.
        checkpoint_path:    Absolute path to the Wav2Lip pretrained checkpoint.
                            Defaults to WAV2LIP_CHECKPOINT_PATH env var.
        resize_factor:      Resize the video by this factor before processing.
                            Use 2 to halve resolution (speeds up processing,
                            reduces quality). Default 1 = original resolution.
        fps:                Output video frame rate. Must match source video FPS.
        pads:               Padding (top, bottom, left, right) added around the
                            detected face region. Adjust if lip region is clipped.
        nosmooth:           If True, disable temporal smoothing of face detections.
                            Use when face detection is jittery on fast movement.

    Returns:
        Absolute path to the output video file.

    Raises:
        FileNotFoundError:  If video or audio file does not exist.
        LipSyncError:       If Wav2Lip fails (environment issues, dependency errors,
                            no face detected, or output not created).

    ⚠️  First-Time Setup Required:
        Wav2Lip must be separately cloned and configured.
        See module docstring for full setup instructions.

    Example:
        >>> output = lipsync(
        ...     video_path="samples/source.mp4",
        ...     audio_path="outputs/cloned.wav",
        ...     output_path="outputs/dubbed.mp4",
        ... )
    """
    # --- Input validation ---
    video_file = Path(video_path)
    audio_file = Path(audio_path)
    out_file = Path(output_path)

    _validate_video(video_file)
    _validate_audio(audio_file)

    # --- Resolve Wav2Lip paths ---
    if wav2lip_repo_path is None or checkpoint_path is None:
        _repo, _ckpt = _resolve_env_paths()
        wav2lip_repo_path = wav2lip_repo_path or _repo
        checkpoint_path = checkpoint_path or _ckpt

    wav2lip_inference_script = Path(wav2lip_repo_path) / "inference.py"
    if not wav2lip_inference_script.is_file():
        raise LipSyncError(
            f"Wav2Lip inference script not found at: '{wav2lip_inference_script}'.\n"
            "Ensure WAV2LIP_REPO_PATH points to the root of a valid Wav2Lip repository "
            "(the one that contains inference.py)."
        )

    # --- Create output directory ---
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[LipSync] Starting | video='%s' | audio='%s' | out='%s'",
        video_file.name, audio_file.name, out_file.name,
    )
    t_start = time.perf_counter()

    # --- Build Wav2Lip subprocess command ---
    # We call inference.py as a subprocess to avoid importing Wav2Lip's modules
    # directly (which would cause dependency conflicts with Babel's own imports).
    pads_str = " ".join(str(p) for p in pads)
    cmd = [
        sys.executable,            # use the same Python interpreter
        str(wav2lip_inference_script),
        "--checkpoint_path", str(checkpoint_path),
        "--face", str(video_file),
        "--audio", str(audio_file),
        "--outfile", str(out_file),
        "--resize_factor", str(resize_factor),
        "--fps", str(fps),
        "--pads", *[str(p) for p in pads],
    ]
    if nosmooth:
        cmd.append("--nosmooth")

    logger.debug("[LipSync] Running command: %s", " ".join(cmd))

    # --- Execute Wav2Lip ---
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(wav2lip_repo_path),  # run from Wav2Lip's repo root
            capture_output=True,
            text=True,
            timeout=600,               # 10-minute timeout for long clips
        )
    except subprocess.TimeoutExpired:
        raise LipSyncError(
            "Wav2Lip process timed out after 10 minutes. "
            "For clips >90 seconds, split into shorter segments. "
            "If the clip is short, check for GPU hangs or zombie processes."
        )
    except Exception as exc:
        raise LipSyncError(
            f"Failed to launch Wav2Lip subprocess: {exc}\n"
            "Ensure the Python interpreter at '{sys.executable}' has Wav2Lip's "
            "dependencies installed. Check notes.md for version pinning details."
        ) from exc

    # --- Check subprocess exit code ---
    if proc.returncode != 0:
        stderr_excerpt = proc.stderr[-2000:] if proc.stderr else "(no stderr)"
        stdout_excerpt = proc.stdout[-1000:] if proc.stdout else "(no stdout)"

        # Provide actionable diagnosis for common failures
        error_hints = ""
        if "No face detected" in (proc.stderr or "") or "No face detected" in (proc.stdout or ""):
            error_hints = (
                "\nDIAGNOSIS: Wav2Lip could not detect a face in the video. "
                "Ensure the video shows a clear, near-frontal face. "
                "Try resize_factor=2 to reduce resolution if face detection is failing."
            )
        elif "np.bool" in (proc.stderr or "") or "np.int" in (proc.stderr or ""):
            error_hints = (
                "\nDIAGNOSIS: NumPy compatibility error (np.bool/np.int removed in NumPy 1.24+). "
                "Downgrade NumPy: pip install numpy==1.23.5  "
                "See notes.md for the full dependency pinning guide."
            )
        elif "face_alignment" in (proc.stderr or ""):
            error_hints = (
                "\nDIAGNOSIS: face_alignment version conflict. "
                "Install exactly: pip install face-alignment==1.3.5"
            )
        elif "CUDA out of memory" in (proc.stderr or ""):
            error_hints = (
                "\nDIAGNOSIS: GPU out of memory. "
                "Try resize_factor=2 to reduce memory usage. "
                "If running multiple models, unload others before calling lipsync()."
            )

        raise LipSyncError(
            f"Wav2Lip failed with exit code {proc.returncode}.{error_hints}\n"
            f"--- stderr (last 2000 chars) ---\n{stderr_excerpt}\n"
            f"--- stdout (last 1000 chars) ---\n{stdout_excerpt}\n"
            "See notes.md for the troubleshooting log."
        )

    # --- Validate output ---
    if not out_file.exists():
        raise LipSyncError(
            f"Wav2Lip reported success (exit code 0) but output file was not created: "
            f"'{output_path}'.\n"
            "This may indicate a path issue or a silent failure in ffmpeg post-processing."
        )
    if out_file.stat().st_size == 0:
        raise LipSyncError(
            f"Wav2Lip output file is empty: '{output_path}'.\n"
            "This typically indicates Wav2Lip crashed during frame assembly."
        )

    elapsed = time.perf_counter() - t_start
    size_mb = out_file.stat().st_size / (1024 * 1024)
    logger.info(
        "[LipSync] Complete | duration=%.2f s | output='%s' | size=%.1f MB",
        elapsed, out_file.name, size_mb,
    )

    return str(out_file.resolve())
