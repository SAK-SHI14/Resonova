"""
Babel — Core Pipeline Orchestrator
==================================
Chains ASR, Translation, Voice Cloning, and Lip-Sync stages into a single pipeline.

Features:
  - Sequence: extract audio -> Whisper -> IndicTrans2 -> XTTS-v2 -> Sync -> Wav2Lip
  - GPU Memory Management: sequentially loads and unloads models to fit T4 free tier.
  - Checkpoint Resumption: skips completed stages if intermediate files exist in checkpoint_dir.
  - Duration Synchronization: supports time-stretching via ffmpeg atempo or accepting drift.
  - Execution Logging: measures per-stage and overall latency.
"""

import gc
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2
import torch

from babel.asr.transcribe import transcribe, unload_model as unload_asr
from babel.exceptions import (
    AudioExtractionError,
    BabelError,
    LipSyncError,
    TranscriptionError,
    TranslationError,
    VoiceCloningError,
)
from babel.logger import get_logger
from babel.lipsync.lipsync import lipsync
from babel.translation.translate import translate, unload_model as unload_translation
from babel.voice_cloning.clone_voice import clone_voice, unload_model as unload_voice

logger = get_logger(__name__)


def get_video_duration(video_path: str) -> float:
    """Return the duration of a video file in seconds using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: '{video_path}'")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps == 0:
        return 0.0
    return frame_count / fps


def get_audio_duration(audio_path: str) -> float:
    """Return duration of an audio file in seconds using soundfile or pydub."""
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        pass

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0
    except Exception as exc:
        raise ValueError(
            f"Cannot determine duration of audio file '{audio_path}': {exc}"
        ) from exc


def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    """Extract audio track from video to a mono 16kHz WAV file using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
        "-y",
        "-loglevel", "error",
    ]
    logger.info("[Pipeline] Extracting audio from video: '%s' -> '%s'", video_path, audio_path)
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise AudioExtractionError(
            f"ffmpeg failed to extract audio from video '{video_path}': {exc.stderr}"
        ) from exc


def time_stretch_audio(input_path: str, output_path: str, ratio: float) -> None:
    """
    Time-stretch or compress audio using ffmpeg's atempo filter.
    Handles arbitrary ratios by chaining filters if ratio is outside [0.5, 2.0].
    """
    if abs(ratio - 1.0) < 0.001:
        logger.info("[Pipeline] Time-stretch ratio is ~1.0. Copying file directly.")
        shutil.copy(input_path, output_path)
        return

    filters = []
    temp_ratio = ratio
    while temp_ratio > 2.0:
        filters.append("atempo=2.0")
        temp_ratio /= 2.0
    while temp_ratio < 0.5:
        filters.append("atempo=0.5")
        temp_ratio /= 0.5
    if temp_ratio != 1.0:
        filters.append(f"atempo={temp_ratio:.4f}")

    filter_str = ",".join(filters)
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-filter:a", filter_str,
        output_path,
        "-y",
        "-loglevel", "error",
    ]
    logger.info("[Pipeline] Time-stretching audio: ratio=%.4f (filter_str='%s')", ratio, filter_str)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"ffmpeg atempo filter failed: {exc.stderr}"
        ) from exc


def unload_all_models() -> None:
    """Unload all cached models and flush CUDA memory."""
    logger.info("[Pipeline] Unloading all models...")
    unload_asr()
    unload_translation()
    unload_voice()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dub_video(
    video_path: str,
    target_lang: str = "hin_Deva",
    output_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    sync_strategy: str = "time_stretch",
    model_size_asr: str = "medium",
    model_name_translation: Optional[str] = None,
    model_name_xtts: Optional[str] = None,
) -> str:
    """
    Translate and dub a video of a person speaking English to a target language (default Hindi)
    in their own cloned voice, with synchronized lip movement.

    Args:
        video_path:             Path to the source English video file.
        target_lang:            Target language code (e.g., "hin_Deva").
        output_path:            Path where the final dubbed video will be saved.
        checkpoint_dir:         Directory where intermediate checkpoints will be saved.
                                If not specified, a folder next to output_path is used.
        sync_strategy:          "time_stretch" to match audio exactly to video length,
                                or "accept_drift" to retain raw synthesized audio length.
        model_size_asr:         Whisper model variant ('tiny', 'base', 'small', 'medium', etc.).
        model_name_translation: HuggingFace translation model identifier.
        model_name_xtts:        XTTS-v2 voice cloning model identifier.

    Returns:
        Absolute path to the final dubbed video file.
    """
    t_pipeline_start = time.perf_counter()
    timings = {}

    # --- Setup directories and paths ---
    src_video = Path(video_path).resolve()
    if not src_video.is_file():
        raise FileNotFoundError(f"Source video file not found: '{video_path}'")

    if output_path is None:
        output_path = str(src_video.parent / f"{src_video.stem}_dubbed{src_video.suffix}")
    dest_video = Path(output_path).resolve()
    dest_video.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir is None:
        checkpoint_dir = str(dest_video.parent / "checkpoints" / src_video.stem)
    ckpt_dir = Path(checkpoint_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTING DUBBING PIPELINE | file='%s' | target='%s'", src_video.name, target_lang)
    logger.info("  Checkpoints directory : %s", ckpt_dir)
    logger.info("  Output video path      : %s", dest_video)
    logger.info("  Sync strategy          : %s", sync_strategy)
    logger.info("=" * 60)

    # Define checkpoint file paths
    original_audio_path = ckpt_dir / "extracted_audio.wav"
    transcript_path = ckpt_dir / "transcript.txt"
    translated_text_path = ckpt_dir / "translated_text.txt"
    cloned_audio_raw_path = ckpt_dir / "cloned_audio_raw.wav"
    cloned_audio_synced_path = ckpt_dir / "cloned_audio_synced.wav"
    final_video_temp_path = ckpt_dir / "dubbed_video_temp.mp4"

    # --- STAGE 1: Audio Extraction ---
    t0 = time.perf_counter()
    if original_audio_path.is_file():
        logger.info("[Pipeline] Checkpoint found: audio already extracted. Skipping.")
    else:
        try:
            extract_audio_from_video(str(src_video), str(original_audio_path))
        except Exception as exc:
            raise AudioExtractionError(f"Audio extraction stage failed: {exc}") from exc
    timings["Audio Extraction"] = time.perf_counter() - t0

    # --- STAGE 2: Speech-to-Text (ASR) ---
    t0 = time.perf_counter()
    if transcript_path.is_file():
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        logger.info("[Pipeline] Checkpoint found: ASR transcript loaded. Skipping ASR.")
    else:
        logger.info("[Pipeline] Running Whisper ASR...")
        try:
            # Unload any other model before loading ASR
            unload_all_models()
            transcript = transcribe(
                str(original_audio_path),
                model_size=model_size_asr,
                language="en",
            )
            transcript_path.write_text(transcript, encoding="utf-8")
            logger.info("[Pipeline] ASR complete. Saved transcript to checkpoint.")
            # Unload ASR to free memory
            unload_asr()
        except Exception as exc:
            raise TranscriptionError(f"ASR stage failed: {exc}") from exc
    timings["ASR (Whisper)"] = time.perf_counter() - t0

    # --- STAGE 3: Translation ---
    t0 = time.perf_counter()
    if translated_text_path.is_file():
        translated_text = translated_text_path.read_text(encoding="utf-8").strip()
        logger.info("[Pipeline] Checkpoint found: translated text loaded. Skipping Translation.")
    else:
        logger.info("[Pipeline] Running Translation English -> Hindi...")
        try:
            unload_all_models()
            translated_text = translate(
                transcript,
                source_lang="eng_Latn",
                target_lang=target_lang,
                model_name=model_name_translation,
            )
            translated_text_path.write_text(translated_text, encoding="utf-8")
            logger.info("[Pipeline] Translation complete. Saved translation to checkpoint.")
            unload_translation()
        except Exception as exc:
            raise TranslationError(f"Translation stage failed: {exc}") from exc
    timings["Translation"] = time.perf_counter() - t0

    # --- STAGE 4: Voice Cloning (multilingual TTS) ---
    t0 = time.perf_counter()
    if cloned_audio_raw_path.is_file():
        logger.info("[Pipeline] Checkpoint found: cloned voice generated. Skipping Voice Cloning.")
    else:
        logger.info("[Pipeline] Running Voice Cloning...")
        try:
            unload_all_models()
            # XTTS requires language code like 'hi' instead of 'hin_Deva'
            lang_short = "hi" if target_lang == "hin_Deva" else target_lang.split("_")[0]
            clone_voice(
                reference_audio_path=str(original_audio_path),
                text=translated_text,
                language=lang_short,
                output_path=str(cloned_audio_raw_path),
                model_name=model_name_xtts,
            )
            logger.info("[Pipeline] Voice Cloning complete. Saved cloned voice to checkpoint.")
            unload_voice()
        except Exception as exc:
            raise VoiceCloningError(f"Voice cloning stage failed: {exc}") from exc
    timings["Voice Cloning (XTTS)"] = time.perf_counter() - t0

    # --- STAGE 5: Audio Synchronization (Duration Match) ---
    t0 = time.perf_counter()
    if cloned_audio_synced_path.is_file():
        logger.info("[Pipeline] Checkpoint found: synced audio file exists. Skipping Sync stage.")
    else:
        try:
            original_duration = get_audio_duration(str(original_audio_path))
            raw_cloned_duration = get_audio_duration(str(cloned_audio_raw_path))
            logger.info(
                "[Pipeline] Duration mismatch check: original=%.2fs | cloned=%.2fs",
                original_duration, raw_cloned_duration
            )

            if sync_strategy == "time_stretch" and original_duration > 0:
                speed_ratio = raw_cloned_duration / original_duration
                # Check for extreme ratio boundaries
                if speed_ratio < 0.6 or speed_ratio > 1.6:
                    logger.warning(
                        "[Pipeline] Speed ratio %.2f is extreme (outside [0.6, 1.6]). "
                        "Time-stretching may sound unnatural.",
                        speed_ratio
                    )
                time_stretch_audio(
                    str(cloned_audio_raw_path),
                    str(cloned_audio_synced_path),
                    speed_ratio
                )
                logger.info("[Pipeline] Sync complete (time-stretch applied).")
            else:
                # Accept drift / no time stretch
                logger.info("[Pipeline] Sync strategy set to accept drift. Copying raw cloned audio.")
                shutil.copy(str(cloned_audio_raw_path), str(cloned_audio_synced_path))

        except Exception as exc:
            raise BabelError(f"Audio synchronization stage failed: {exc}") from exc
    timings["Synchronization"] = time.perf_counter() - t0

    # --- STAGE 6: Lip-Sync (Wav2Lip) ---
    t0 = time.perf_counter()
    # We always run the final lipsync step if the destination video is missing,
    # but we can copy from temp checkpoint if it exists.
    if dest_video.is_file():
        logger.info("[Pipeline] Checkpoint found: final dubbed video already exists. Skipping Lip-Sync.")
    elif final_video_temp_path.is_file():
        logger.info("[Pipeline] Copying dubbed video from temp checkpoint...")
        shutil.copy(str(final_video_temp_path), str(dest_video))
    else:
        logger.info("[Pipeline] Running Wav2Lip Lip-Sync...")
        try:
            # Run Wav2Lip
            lipsync(
                video_path=str(src_video),
                audio_path=str(cloned_audio_synced_path),
                output_path=str(final_video_temp_path),
            )
            shutil.copy(str(final_video_temp_path), str(dest_video))
            logger.info("[Pipeline] Lip-Sync complete. Final dubbed video created.")
        except Exception as exc:
            raise LipSyncError(f"Lip-Sync stage failed: {exc}") from exc
    timings["Lip-Sync (Wav2Lip)"] = time.perf_counter() - t0

    pipeline_duration = time.perf_counter() - t_pipeline_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE | total_time=%.2fs", pipeline_duration)
    for stage, t in timings.items():
        logger.info("  - %-25s : %6.2fs (%5.1f%%)", stage, t, (t / pipeline_duration) * 100)
    logger.info("=" * 60)

    return str(dest_video)
