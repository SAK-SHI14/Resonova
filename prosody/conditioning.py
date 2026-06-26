"""
Mimi — Prosody Conditioning
=============================
Applies energy matching and logs INF/cadence discrepancies between
the original and translated/cloned audio tracks.
"""

import os
import shutil
import subprocess
from typing import Dict, Any

from mimi.exceptions import ProsodyError
from mimi.logger import get_logger
from mimi.prosody.extract import extract_prosody

logger = get_logger(__name__)


def apply_prosody_conditioning(
    tts_output_path: str,
    original_audio_path: str,
    output_path: str,
) -> str:
    """
    Apply prosody styling to the synthesized output.
    Aligns the synthesized audio's volume (RMS energy) with the original audio's RMS energy.

    Args:
        tts_output_path:     Path to the synthesized cloned audio file (from XTTS).
        original_audio_path: Path to the original audio file (reference speaker).
        output_path:         Path where the conditioned audio will be saved.

    Returns:
        Absolute path to the output audio file.
    """
    if not os.path.isfile(tts_output_path):
        raise FileNotFoundError(f"Synthesized audio not found: '{tts_output_path}'")
    if not os.path.isfile(original_audio_path):
        raise FileNotFoundError(f"Original audio reference not found: '{original_audio_path}'")

    try:
        # 1. Extract features of both audios to compare
        orig_features = extract_prosody(original_audio_path)
        synth_features = extract_prosody(tts_output_path)

        orig_rms = orig_features["mean_energy"]
        synth_rms = synth_features["mean_energy"]

        logger.info(
            "[Prosody Conditioning] Aligning energy levels | orig=%.4f | synth=%.4f",
            orig_rms, synth_rms
        )

        # Create parent directories
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if orig_rms <= 0 or synth_rms <= 0:
            logger.warning("[Prosody Conditioning] Invalid RMS value(s) detected. Copying file directly.")
            shutil.copy(tts_output_path, output_path)
            return str(os.path.abspath(output_path))

        # 2. Compute scaling factor
        scale_factor = orig_rms / synth_rms
        # Clamp scale factor to prevent extreme amplification or mute
        scale_factor = max(0.2, min(5.0, scale_factor))

        if abs(scale_factor - 1.0) < 0.05:
            logger.info("[Prosody Conditioning] Energy levels are already matched. Copying file directly.")
            shutil.copy(tts_output_path, output_path)
            return str(os.path.abspath(output_path))

        # 3. Apply volume filter using ffmpeg
        cmd = [
            "ffmpeg",
            "-i", tts_output_path,
            "-filter:a", f"volume={scale_factor:.4f}",
            output_path,
            "-y",
            "-loglevel", "error",
        ]
        logger.info(
            "[Prosody Conditioning] Scaling volume: scale=%.4f (orig_rms=%.4f / synth_rms=%.4f)",
            scale_factor, orig_rms, synth_rms
        )

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Verify output exists
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise ProsodyError("ffmpeg volume matching produced an empty or missing output file.")

        return str(os.path.abspath(output_path))

    except Exception as exc:
        logger.warning(
            "[Prosody Conditioning] Volume matching failed, falling back to direct copy: %s",
            exc
        )
        try:
            shutil.copy(tts_output_path, output_path)
            return str(os.path.abspath(output_path))
        except Exception as copy_exc:
            raise ProsodyError(f"Prosody conditioning fallback copy failed: {copy_exc}") from copy_exc
