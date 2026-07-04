"""
Vaani — Voice Cloning Module
==============================
Wrapper around Coqui XTTS-v2 for zero-shot voice cloning and multilingual TTS.

Model: tts_models/multilingual/multi-dataset/xtts_v2  (open-weight, runs fully locally)

ADR reference: docs/adrs/ADR-002-voice-cloning-model.md

GPU Memory (T4 reference):
  - XTTS-v2 : ~4–5 GB VRAM, ~30–60s load time on first run

Reference Clip Requirements:
  - Minimum duration : 6 seconds of clean speech (XTTS-v2 requirement)
  - Recommended      : 10–30 seconds for better voice fidelity
  - Format           : WAV, 22050 Hz or 16000 Hz, mono preferred
  - Quality          : Minimal background noise — this IS the voice identity reference;
                       noisy references produce noisier clones

Supported Languages (partial list — full list in XTTS-v2 model card):
  en (English), hi (Hindi), es (Spanish), fr (French), de (German),
  it (Italian), pt (Portuguese), pl (Polish), tr (Turkish), ru (Russian),
  nl (Dutch), cs (Czech), ar (Arabic), zh-cn (Chinese), ja (Japanese),
  hu (Hungarian), ko (Korean)

Usage:
    from vaani.voice_cloning.clone_voice import clone_voice
    output_path = clone_voice(
        reference_audio_path="samples/my_voice_reference.wav",
        text="नमस्ते, यह एक परीक्षण वाक्य है।",
        language="hi",
        output_path="outputs/cloned_hindi.wav",
    )
"""

import os
import time
from pathlib import Path
from typing import Optional

from vaani.exceptions import VoiceCloningError
from vaani.logger import get_logger

logger = get_logger(__name__)

# Minimum reference clip duration in seconds (XTTS-v2 hard requirement)
MIN_REFERENCE_DURATION_S = 6.0

# Model cache — avoid re-loading within the same session
_tts_model_cache: dict = {}


def _get_audio_duration(audio_path: Path) -> float:
    """
    Return duration of an audio file in seconds.

    Uses soundfile for WAV/FLAC, falls back to pydub for MP3/M4A.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds as a float.

    Raises:
        VoiceCloningError: If duration cannot be determined.
    """
    try:
        import soundfile as sf  # noqa: PLC0415
        info = sf.info(str(audio_path))
        return info.duration
    except Exception:
        pass  # try pydub as fallback

    try:
        from pydub import AudioSegment  # noqa: PLC0415
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0
    except Exception as exc:
        raise VoiceCloningError(
            f"Cannot determine duration of reference audio '{audio_path}': {exc}\n"
            "Ensure soundfile or pydub is installed, and ffmpeg is available on PATH."
        ) from exc


def _load_tts_model(model_name: str):
    """
    Load (or retrieve from cache) the XTTS-v2 TTS model.

    Args:
        model_name: Coqui TTS model identifier string.

    Returns:
        A loaded TTS model object.

    Raises:
        VoiceCloningError: If the model cannot be loaded.
    """
    if model_name in _tts_model_cache:
        logger.debug("XTTS-v2 model '%s' loaded from cache.", model_name)
        return _tts_model_cache[model_name]

    try:
        from TTS.api import TTS  # noqa: PLC0415
    except ImportError as exc:
        raise VoiceCloningError(
            "Coqui TTS is not installed. "
            "Run: pip install TTS==0.22.0"
        ) from exc

    logger.info(
        "Loading XTTS-v2 model '%s' — first run may take 1–2 min for download (~2 GB)...",
        model_name,
    )
    t0 = time.perf_counter()

    try:
        tts = TTS(model_name=model_name, progress_bar=True)
        # Use GPU if available
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            tts = tts.to("cuda")
            logger.info("XTTS-v2 moved to CUDA device.")
        else:
            logger.warning(
                "XTTS-v2 running on CPU — synthesis will be significantly slower. "
                "Expected ~2–5x slower than GPU inference."
            )
    except Exception as exc:
        raise VoiceCloningError(
            f"Failed to load XTTS-v2 model '{model_name}': {exc}\n"
            "Possible causes: insufficient VRAM (~4–5 GB needed), "
            "network error during download, or TTS version mismatch. "
            "See notes.md for version pinning details."
        ) from exc

    elapsed = time.perf_counter() - t0
    logger.info("XTTS-v2 model loaded in %.1f s.", elapsed)
    _tts_model_cache[model_name] = tts
    return tts


def clone_voice(
    reference_audio_path: str,
    text: str,
    language: str,
    output_path: str,
    model_name: Optional[str] = None,
) -> str:
    """
    Synthesize speech in a cloned voice using XTTS-v2 zero-shot voice cloning.

    The cloned voice is derived entirely from the reference clip — no fine-tuning
    or training is required. Quality depends heavily on reference clip quality and length.

    Args:
        reference_audio_path: Path to a reference WAV file from the target speaker.
                              Must be ≥6 seconds of clean speech (XTTS-v2 requirement).
        text:                 The text to synthesize in the cloned voice.
        language:             BCP-47 language code for synthesis (e.g., "en", "hi").
                              Must be one of XTTS-v2's supported languages.
        output_path:          Path where the output WAV will be saved.
                              Parent directories are created automatically.
        model_name:           XTTS-v2 model identifier. Defaults to the value of
                              ``XTTS_MODEL_NAME`` env var, or the standard XTTS-v2 name.

    Returns:
        Absolute path to the generated WAV file (same as output_path).

    Raises:
        FileNotFoundError:  If ``reference_audio_path`` does not exist.
        ValueError:         If reference clip is too short (<6 s) or text is empty.
        VoiceCloningError:  If XTTS-v2 fails to generate audio.

    Known Limitations (documented, not hidden):
        - Voice cloning quality degrades with noisy reference clips.
        - Hindi synthesis quality is lower than English in XTTS-v2 as of v2.0.3
          — the model was trained on less Hindi data than English.
        - Prosody (pitch/rate/emotion) in the output reflects the TEXT content and
          reference clip style, NOT a controllable parameter at this stage.
          Prosody conditioning is handled separately in the prosody module (Phase 3).

    Example:
        >>> path = clone_voice(
        ...     reference_audio_path="samples/reference.wav",
        ...     text="This is a test sentence in English.",
        ...     language="en",
        ...     output_path="outputs/cloned_en.wav",
        ... )
    """
    # --- Input validation ---
    ref_path = Path(reference_audio_path)
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference audio file not found: '{reference_audio_path}'. "
            "Ensure you have recorded and saved a reference clip before calling clone_voice()."
        )
    if not ref_path.is_file():
        raise VoiceCloningError(
            f"Expected a file but got a directory: '{reference_audio_path}'."
        )

    if not text or not text.strip():
        raise ValueError(
            "clone_voice() received empty text. "
            "Provide the text to synthesize in the cloned voice."
        )

    # --- Reference clip duration check ---
    duration = _get_audio_duration(ref_path)
    if duration < MIN_REFERENCE_DURATION_S:
        raise ValueError(
            f"Reference audio is too short: {duration:.1f} s. "
            f"XTTS-v2 requires ≥{MIN_REFERENCE_DURATION_S} s of speech for voice cloning. "
            "Record a longer reference clip or concatenate multiple clips."
        )
    logger.debug("Reference clip duration: %.1f s (OK, ≥ %.0f s).", duration, MIN_REFERENCE_DURATION_S)

    # --- Resolve model name ---
    if model_name is None:
        model_name = os.environ.get(
            "XTTS_MODEL_NAME",
            "tts_models/multilingual/multi-dataset/xtts_v2",
        )

    # --- Create output directory ---
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[VoiceCloning] Starting | language='%s' | ref='%s' | chars=%d | out='%s'",
        language, ref_path.name, len(text), out_path.name,
    )
    t_start = time.perf_counter()

    # --- Load model ---
    tts = _load_tts_model(model_name)

    # --- Synthesize ---
    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=str(ref_path),
            language=language,
            file_path=str(out_path),
        )
    except Exception as exc:
        raise VoiceCloningError(
            f"XTTS-v2 synthesis failed: {exc}\n"
            f"  text='{text[:80]}...'\n"
            f"  language='{language}'\n"
            f"  reference='{reference_audio_path}'\n"
            "Possible causes: unsupported language code, GPU OOM, "
            "corrupted reference clip, or text too long (try shorter chunks). "
            "See notes.md for known fixes."
        ) from exc

    # --- Validate output ---
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise VoiceCloningError(
            f"XTTS-v2 appeared to succeed but output file is missing or empty: '{output_path}'."
        )

    # Verify audio has non-zero duration
    out_duration = _get_audio_duration(out_path)
    if out_duration < 0.1:
        raise VoiceCloningError(
            f"Generated audio file has near-zero duration ({out_duration:.2f} s). "
            "This likely indicates a synthesis failure. "
            f"Output file: '{output_path}'."
        )

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[VoiceCloning] Complete | duration=%.2f s | output_duration=%.2f s | file='%s'",
        elapsed, out_duration, out_path.name,
    )

    return str(out_path.resolve())


def unload_model() -> None:
    """Clear loaded XTTS-v2 models from cache and free CUDA memory."""
    global _tts_model_cache
    if _tts_model_cache:
        logger.info("Unloading XTTS-v2 models and clearing CUDA cache...")
        _tts_model_cache.clear()

        import gc  # noqa: PLC0415
        import torch  # noqa: PLC0415

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("XTTS-v2 models cache cleared.")

