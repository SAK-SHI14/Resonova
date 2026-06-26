"""
Mimi — Prosody Feature Extraction
===================================
Extracts F0 (pitch) contours, RMS (energy) contours, speaking rate, and pauses
from speech audio using librosa.
"""

import os
from typing import Dict, Any

import numpy as np

from mimi.exceptions import ProsodyError
from mimi.logger import get_logger

logger = get_logger(__name__)


def extract_prosody(audio_path: str) -> Dict[str, Any]:
    """
    Extract prosody features from an audio file.

    Args:
        audio_path: Path to the input WAV audio file.

    Returns:
        A dictionary containing extracted features:
            - 'mean_pitch' (Hz)
            - 'std_pitch' (Hz)
            - 'mean_energy' (RMS)
            - 'std_energy' (RMS)
            - 'speaking_rate' (syllables/sec)
            - 'pause_ratio' (fraction of silence)
            - 'pitch_contour' (list of F0 float values)
            - 'energy_contour' (list of RMS float values)

    Raises:
        FileNotFoundError: If the file does not exist.
        ProsodyError: If extraction fails due to format or processing issues.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: '{audio_path}'")

    try:
        import librosa  # noqa: PLC0415
        # Load audio (mono, original sample rate)
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))

        if duration < 0.1:
            raise ProsodyError(f"Audio file too short to extract prosody: {duration:.2f}s")

        logger.info(
            "[Prosody] Extracting features | file='%s' | duration=%.2f s | sr=%d Hz",
            os.path.basename(audio_path), duration, sr
        )

        # --- 1. Pitch Contour (F0) ---
        # Human speaking pitch bounds are generally between 50 Hz and 500 Hz
        try:
            f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
            # Replace NaNs/Infs with 0
            f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as exc:
            logger.warning("[Prosody] Yin pitch tracking failed, falling back to zeros: %s", exc)
            f0 = np.zeros(100)

        voiced_pitches = f0[f0 > 0]
        mean_pitch = float(np.mean(voiced_pitches)) if len(voiced_pitches) > 0 else 0.0
        std_pitch = float(np.std(voiced_pitches)) if len(voiced_pitches) > 0 else 0.0

        # --- 2. Energy Contour (RMS) ---
        rms = librosa.feature.rms(y=y)
        # Flatten to 1D
        rms_flat = rms.flatten()
        mean_energy = float(np.mean(rms_flat))
        std_energy = float(np.std(rms_flat))

        # --- 3. Pauses and Silences ---
        # Convert RMS to decibels relative to peak
        rms_db = librosa.amplitude_to_db(rms_flat, ref=np.max)
        # Threshold for silence: -30dB relative to peak RMS
        silence_threshold_db = -30.0
        silent_frames = rms_db < silence_threshold_db
        pause_duration = float(np.sum(silent_frames) * (duration / len(rms_flat)))
        pause_ratio = pause_duration / duration if duration > 0 else 0.0

        # --- 4. Speaking Rate (Syllable Onsets per Second) ---
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            syllable_count = len(onsets)
            speaking_rate = syllable_count / duration if duration > 0 else 0.0
        except Exception as exc:
            logger.warning("[Prosody] Onset detection failed, falling back to 0: %s", exc)
            speaking_rate = 0.0

        logger.debug(
            "[Prosody] Extraction successful: pitch=%.1f Hz (std=%.1f) | energy=%.4f | rate=%.2f syl/s",
            mean_pitch, std_pitch, mean_energy, speaking_rate
        )

        return {
            "mean_pitch": mean_pitch,
            "std_pitch": std_pitch,
            "mean_energy": mean_energy,
            "std_energy": std_energy,
            "speaking_rate": speaking_rate,
            "pause_ratio": pause_ratio,
            "pitch_contour": f0.tolist(),
            "energy_contour": rms_flat.tolist(),
        }

    except Exception as exc:
        if isinstance(exc, ProsodyError):
            raise exc
        raise ProsodyError(f"Failed to analyze prosody features for '{audio_path}': {exc}") from exc
