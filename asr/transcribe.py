"""
Babel — ASR Module
==================
Wrapper around OpenAI Whisper for automatic speech recognition.

Model: openai-whisper (open-weight, runs fully locally — no API calls)
Supported model sizes: tiny, base, small, medium (default), large-v3

GPU Memory (T4 reference):
  - medium : ~1.5 GB VRAM  ← default, recommended for speed/quality balance
  - large-v3 : ~3.0 GB VRAM ← higher quality, still fits T4

Usage:
    from babel.asr.transcribe import transcribe
    text = transcribe("path/to/audio.wav")
    text = transcribe("path/to/audio.wav", model_size="large-v3", language="en")
"""

import os
import time
from pathlib import Path
from typing import Optional

from babel.exceptions import TranscriptionError
from babel.logger import get_logger

logger = get_logger(__name__)

# Cache loaded model to avoid re-loading on repeated calls within the same session.
_whisper_model_cache: dict = {}


def _load_model(model_size: str):
    """
    Load (or retrieve from cache) a Whisper model of the given size.

    Args:
        model_size: One of 'tiny', 'base', 'small', 'medium', 'large-v3'.

    Returns:
        A loaded whisper model object.

    Raises:
        TranscriptionError: If the model cannot be loaded.
    """
    if model_size in _whisper_model_cache:
        logger.debug("Whisper model '%s' loaded from cache.", model_size)
        return _whisper_model_cache[model_size]

    try:
        import whisper  # noqa: PLC0415 (import inside function intentional — lazy load)
    except ImportError as exc:
        raise TranscriptionError(
            "openai-whisper is not installed. "
            "Run: pip install openai-whisper"
        ) from exc

    logger.info("Loading Whisper model '%s' — this may take 30–60 s on first run...", model_size)
    t0 = time.perf_counter()
    try:
        model = whisper.load_model(model_size)
    except Exception as exc:
        raise TranscriptionError(
            f"Failed to load Whisper model '{model_size}': {exc}"
        ) from exc

    elapsed = time.perf_counter() - t0
    logger.info("Whisper model '%s' loaded in %.1f s.", model_size, elapsed)
    _whisper_model_cache[model_size] = model
    return model


def transcribe(
    audio_path: str,
    model_size: str = "medium",
    language: Optional[str] = "en",
    task: str = "transcribe",
) -> str:
    """
    Transcribe speech from an audio file using OpenAI Whisper.

    Args:
        audio_path:  Path to the input audio file (WAV, MP3, M4A, FLAC, etc.).
                     The file must exist and be readable.
        model_size:  Whisper model variant to use. Defaults to 'medium'.
                     Options: 'tiny', 'base', 'small', 'medium', 'large-v3'.
                     Larger models are more accurate but slower.
        language:    BCP-47 language code of the source speech (e.g., 'en', 'hi').
                     Pass None to let Whisper auto-detect the language.
        task:        'transcribe' (default) or 'translate' (Whisper's built-in
                     translation to English — not used in Babel's main pipeline,
                     but available for debugging).

    Returns:
        The transcribed text as a stripped string.

    Raises:
        FileNotFoundError:  If ``audio_path`` does not exist.
        TranscriptionError: If Whisper fails or returns an empty transcript.

    Example:
        >>> text = transcribe("samples/my_clip.wav")
        >>> print(text)
        'Hello, this is a sample English sentence.'
    """
    # --- Input validation ---
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(
            f"Audio file not found: '{audio_path}'. "
            "Check the path and ensure the file exists before calling transcribe()."
        )
    if not audio_file.is_file():
        raise TranscriptionError(
            f"Expected a file but got a directory: '{audio_path}'."
        )

    logger.info(
        "[ASR] Starting transcription | file='%s' | model='%s' | language='%s'",
        audio_file.name, model_size, language or "auto-detect",
    )
    t_start = time.perf_counter()

    # --- Load model ---
    model = _load_model(model_size)

    # --- Run inference ---
    try:
        result = model.transcribe(
            str(audio_file),
            language=language,
            task=task,
            verbose=False,
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Whisper transcription failed for '{audio_path}': {exc}\n"
            "Possible causes: corrupted audio file, unsupported format, or GPU OOM. "
            "Try a smaller model_size (e.g., 'small') or check audio integrity with ffprobe."
        ) from exc

    # --- Extract and validate transcript ---
    transcript: str = result.get("text", "").strip()
    if not transcript:
        raise TranscriptionError(
            f"Whisper returned an empty transcript for '{audio_path}'. "
            "Possible causes: audio is silent, too short (<0.5 s), or not in the "
            f"expected language ('{language}'). "
            "Try passing language=None to enable auto-detection."
        )

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[ASR] Transcription complete | duration=%.2f s | chars=%d | preview='%s...'",
        elapsed,
        len(transcript),
        transcript[:60].replace("\n", " "),
    )

    return transcript


def transcribe_with_timestamps(
    audio_path: str,
    model_size: str = "medium",
    language: Optional[str] = "en",
) -> list[dict]:
    """
    Transcribe audio and return word/segment-level timestamps.

    Useful for Phase 3 prosody analysis (speaking rate estimation).

    Args:
        audio_path:  Path to the input audio file.
        model_size:  Whisper model variant. Defaults to 'medium'.
        language:    BCP-47 language code of the source speech.

    Returns:
        List of segment dicts, each with keys:
            'id', 'start', 'end', 'text', 'tokens', 'avg_logprob', 'no_speech_prob'

    Raises:
        FileNotFoundError:  If ``audio_path`` does not exist.
        TranscriptionError: If Whisper fails.
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: '{audio_path}'.")

    logger.info(
        "[ASR] Starting timestamped transcription | file='%s' | model='%s'",
        audio_file.name, model_size,
    )
    t_start = time.perf_counter()

    model = _load_model(model_size)

    try:
        result = model.transcribe(
            str(audio_file),
            language=language,
            task="transcribe",
            verbose=False,
            word_timestamps=True,
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Timestamped transcription failed for '{audio_path}': {exc}"
        ) from exc

    segments = result.get("segments", [])
    elapsed = time.perf_counter() - t_start
    logger.info(
        "[ASR] Timestamped transcription complete | duration=%.2f s | segments=%d",
        elapsed, len(segments),
    )
    return segments


def unload_model() -> None:
    """Clear loaded Whisper models from cache and free CUDA memory."""
    global _whisper_model_cache
    if _whisper_model_cache:
        logger.info("Unloading Whisper models and clearing CUDA cache...")
        _whisper_model_cache.clear()

        import gc  # noqa: PLC0415
        import torch  # noqa: PLC0415

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Whisper models cache cleared.")

