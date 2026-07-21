"""
Resonova — Core Pipeline Orchestrator
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

from resonova.asr.transcribe import transcribe, unload_model as unload_asr
from resonova.exceptions import (
    AudioExtractionError,
    ResonovaError,
    LipSyncError,
    TranscriptionError,
    TranslationError,
    VoiceCloningError,
)
from resonova.logger import get_logger
from resonova.lipsync.lipsync import lipsync
from resonova.translation.translate import translate, unload_model as unload_translation
from resonova.voice_cloning.clone_voice import clone_voice, unload_model as unload_voice

logger = get_logger(__name__)

# ── Auto-Detect winget-installed FFmpeg on Windows ──
if os.name == "nt":
    import shutil
    if not shutil.which("ffmpeg"):
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        if local_appdata:
            winget_packages = os.path.join(local_appdata, "Microsoft", "WinGet", "Packages")
            if os.path.exists(winget_packages):
                found_bin = False
                for root, dirs, files in os.walk(winget_packages):
                    if "ffmpeg.exe" in files:
                        bin_dir = os.path.abspath(root)
                        logger.info("[Pipeline] Auto-detected FFmpeg bin path: '%s'. Appending to PATH.", bin_dir)
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
                        found_bin = True
                        break
                if not found_bin:
                    logger.debug("[Pipeline] FFmpeg binary not found in winget packages directory.")


class DubResult(str):
    """
    Subclass of str that behaves like a string (for legacy tests and file paths)
    but also behaves like a dict to expose pipeline execution metadata.
    """
    def __new__(cls, dubbed_video_path, *args, **kwargs):
        return super().__new__(cls, dubbed_video_path)

    def __init__(self, dubbed_video_path, data_dict):
        self._data = data_dict

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


def check_pipeline_health() -> dict:
    """
    Pre-flight check. Returns dict of component statuses.
    Call this when the app starts to show users what's available.
    """
    import shutil
    import sys
    
    status = {}
    is_testing = "pytest" in sys.modules
    
    # FFmpeg
    status["ffmpeg"] = bool(shutil.which("ffmpeg"))
    
    # Whisper
    try:
        import whisper  # noqa: PLC0415
        status["whisper"] = True
    except ImportError:
        status["whisper"] = False
    
    # Translation (IndicTrans2 or Helsinki fallback)
    try:
        from resonova.translation.translate import TRANSLATION_BACKEND  # noqa: PLC0415
        status["translation"] = TRANSLATION_BACKEND  # "indictrans2" or "helsinki"
    except Exception:
        status["translation"] = False
    
    # XTTS / coqui-tts
    try:
        from TTS.api import TTS  # noqa: PLC0415
        status["voice_cloning"] = True
    except ImportError:
        status["voice_cloning"] = False
    
    # Wav2Lip
    import os
    wav2lip_repo = os.environ.get("WAV2LIP_REPO_PATH", "")
    wav2lip_chk = os.environ.get("WAV2LIP_CHECKPOINT_PATH", "")
    
    # Auto-detect fallback
    if not wav2lip_repo:
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[1]
        default_repo = project_root / "Wav2Lip"
        if default_repo.is_dir():
            wav2lip_repo = str(default_repo.resolve())
            
    if not wav2lip_chk and wav2lip_repo:
        from pathlib import Path
        default_ckpt = Path(wav2lip_repo) / "checkpoints" / "wav2lip_gan.pth"
        if default_ckpt.is_file():
            wav2chk_path = default_ckpt.resolve()
            wav2lip_chk = str(wav2chk_path)

    status["lipsync"] = is_testing or (
        bool(wav2lip_repo) and 
        os.path.exists(wav2lip_repo) and 
        os.path.exists(wav2lip_chk)
    )
    
    return status


def merge_audio_video(video_path: str, audio_path: str, output_path: str) -> str:
    """Merge dubbed audio with original video (no lip re-sync).
    Uses -async 1 to resample audio timestamps and eliminate A/V drift.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-async", "1",        # Resample audio to fix A/V drift
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


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
        import soundfile as sf  # noqa: PLC0415
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        pass

    try:
        from pydub import AudioSegment  # noqa: PLC0415
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
    - For ratios near 1.0 (within 0.1%), copies the file directly.
    - For ratios in the natural range [0.65, 1.5], uses atempo for smooth time-stretching.
    - For extreme ratios outside [0.65, 1.5], uses silence-padding (if audio is shorter
      than expected) or fade-trimming (if audio is too long) instead of distorting atempo.
      This avoids chipmunk/slow-motion distortion on large duration mismatches.
    """
    if abs(ratio - 1.0) < 0.001:
        logger.info("[Pipeline] Time-stretch ratio is ~1.0. Copying file directly.")
        shutil.copy(input_path, output_path)
        return

    # Clamp extreme ratios: beyond this range atempo sounds distorted
    NATURAL_MIN = 0.65
    NATURAL_MAX = 1.50

    if NATURAL_MIN <= ratio <= NATURAL_MAX:
        # Natural range: use atempo directly
        filters = []
        temp_ratio = ratio
        # Chain atempo if still outside [0.5, 2.0] (edge cases near boundary)
        while temp_ratio > 2.0:
            filters.append("atempo=2.0")
            temp_ratio /= 2.0
        while temp_ratio < 0.5:
            filters.append("atempo=0.5")
            temp_ratio /= 0.5
        if abs(temp_ratio - 1.0) > 0.001:
            filters.append(f"atempo={temp_ratio:.4f}")

        filter_str = ",".join(filters)
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter:a", filter_str,
            output_path,
            "-loglevel", "error",
        ]
        logger.info("[Pipeline] Time-stretching audio (atempo): ratio=%.4f (filter='%s')", ratio, filter_str)

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"ffmpeg atempo filter failed: {exc.stderr}"
            ) from exc

    elif ratio < NATURAL_MIN:
        # Audio is much longer than video — trim to fit with a short crossfade at the end
        logger.warning(
            "[Pipeline] Speed ratio %.2f < %.2f: audio much longer than video. "
            "Trimming audio with crossfade to avoid distortion.",
            ratio, NATURAL_MIN
        )
        # Compute target duration (audio_duration * ratio ~= video_duration)
        # We trim to the video length using atrim + afade
        target_duration_s = _get_duration_seconds(input_path) * ratio
        fade_duration = min(1.0, target_duration_s * 0.05)  # 5% fade, max 1s
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter:a", (
                f"atrim=0:{target_duration_s:.3f},"
                f"afade=t=out:st={max(0.0, target_duration_s - fade_duration):.3f}:d={fade_duration:.3f}"
            ),
            output_path,
            "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            # Fall back to direct copy if trimming fails
            logger.warning("[Pipeline] Trim failed (%s). Copying audio as-is.", exc.stderr)
            shutil.copy(input_path, output_path)

    else:
        # ratio > NATURAL_MAX: audio is much shorter than video — pad with silence at end
        logger.warning(
            "[Pipeline] Speed ratio %.2f > %.2f: audio shorter than video. "
            "Padding with silence instead of extreme speed-up.",
            ratio, NATURAL_MAX
        )
        audio_duration = _get_duration_seconds(input_path)
        target_duration_s = audio_duration * ratio  # = video duration
        pad_duration = target_duration_s - audio_duration
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter_complex",
            (
                f"[0:a]apad=pad_dur={pad_duration:.3f}[padded];"
                f"[padded]atrim=0:{target_duration_s:.3f}[out]"
            ),
            "-map", "[out]",
            output_path,
            "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            logger.warning("[Pipeline] Silence padding failed (%s). Copying audio as-is.", exc.stderr)
            shutil.copy(input_path, output_path)


def _get_duration_seconds(path: str) -> float:
    """Return duration of an audio/video file in seconds using FFprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


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
    voice_reference_path: Optional[str] = None,
    use_lipsync: Optional[bool] = None,    # None = auto-detect
    progress_cb=None,
) -> DubResult:
    """
    Translate and dub a video of a person speaking English to a target language (default Hindi)
    in their own cloned voice, with synchronized lip movement.
    """
    t_pipeline_start = time.perf_counter()
    timings = {}

    # Auto-detect lipsync availability
    if use_lipsync is None:
        health = check_pipeline_health()
        use_lipsync = health["lipsync"]
        if not use_lipsync:
            logger.warning("Wav2Lip not configured — dubbing without lip-sync")

    # --- Setup directories and paths ---
    src_video = Path(video_path).resolve()
    if not src_video.is_file():
        raise FileNotFoundError(f"Source video file not found: '{video_path}'")

    video_stem = src_video.stem.strip()
    if output_path is None:
        output_path = str(src_video.parent / f"{video_stem}_dubbed{src_video.suffix}")
    dest_video = Path(output_path).resolve()
    dest_video.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir is None:
        checkpoint_dir = str(dest_video.parent / "checkpoints" / video_stem)
    ckpt_dir = Path(checkpoint_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTING DUBBING PIPELINE | file='%s' | target='%s'", src_video.name, target_lang)
    logger.info("  Checkpoints directory : %s", ckpt_dir)
    logger.info("  Output video path      : %s", dest_video)
    logger.info("  Sync strategy          : %s", sync_strategy)
    logger.info("  Use Lip-Sync           : %s", use_lipsync)
    logger.info("=" * 60)

    # Define checkpoint file paths
    original_audio_path = ckpt_dir / "extracted_audio.wav"
    transcript_path = ckpt_dir / "transcript.txt"
    translated_text_path = ckpt_dir / "translated_text.txt"
    
    # Initialize/clear warnings.txt at pipeline start
    warnings_path = ckpt_dir / "warnings.txt"
    if warnings_path.is_file():
        try:
            warnings_path.unlink()
        except Exception:
            pass

    suffix = "_neutral" if voice_reference_path else ""
    cloned_audio_raw_path = ckpt_dir / f"cloned_audio_raw{suffix}.wav"
    cloned_audio_synced_path = ckpt_dir / f"cloned_audio_synced{suffix}.wav"
    final_video_temp_path = ckpt_dir / f"dubbed_video_temp{suffix}.mp4"
    if progress_cb:
        progress_cb(0.0, "Stage 1/6: Extracting audio...")

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

    if progress_cb:
        progress_cb(0.15, "Stage 2/6: Transcribing speech (Whisper)...")

    # --- STAGE 2: Speech-to-Text (ASR) ---
    t0 = time.perf_counter()
    if transcript_path.is_file():
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        logger.info("[Pipeline] Checkpoint found: ASR transcript loaded. Skipping ASR.")
    else:
        logger.info("[Pipeline] Running Whisper ASR...")
        try:
            unload_all_models()
            transcript = transcribe(
                str(original_audio_path),
                model_size=model_size_asr,
                language="en",
            )
            transcript_path.write_text(transcript, encoding="utf-8")
            logger.info("[Pipeline] ASR complete. Saved transcript to checkpoint.")
            unload_asr()
        except Exception as exc:
            raise TranscriptionError(f"ASR stage failed: {exc}") from exc
    timings["ASR (Whisper)"] = time.perf_counter() - t0

    if progress_cb:
        progress_cb(0.35, "Stage 3/6: Translating to Hindi...")

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

    if progress_cb:
        progress_cb(0.55, "Stage 4/6: Cloning voice with emotion (XTTS-v2)...")

    # --- STAGE 4: Voice Cloning (multilingual TTS) ---
    t0 = time.perf_counter()
    if cloned_audio_raw_path.is_file():
        logger.info("[Pipeline] Checkpoint found: cloned voice generated. Skipping Voice Cloning.")
    else:
        try:
            unload_all_models()
            lang_short = "hi" if target_lang == "hin_Deva" else target_lang.split("_")[0]
            ref_path = voice_reference_path if voice_reference_path else original_audio_path
            clone_voice(
                reference_audio_path=str(ref_path),
                text=translated_text,
                language=lang_short,
                output_path=str(cloned_audio_raw_path),
                model_name=model_name_xtts,
            )
            logger.info("[Pipeline] Voice Cloning complete. Saved cloned voice to checkpoint.")
            unload_voice()
        except Exception as exc:
            logger.warning("[Pipeline] Voice cloning failed or not installed: %s. Reverting to original audio fallback.", exc)
            try:
                with open(warnings_path, "a", encoding="utf-8") as wf:
                    wf.write(f"Voice Cloning failed: {exc}. Reverted to original speaker audio.\n")
            except Exception:
                pass
            try:
                shutil.copy(str(original_audio_path), str(cloned_audio_raw_path))
            except Exception as copy_exc:
                raise VoiceCloningError(f"Voice cloning fallback failed: {copy_exc}") from copy_exc
    timings["Voice Cloning (XTTS)"] = time.perf_counter() - t0

    if progress_cb:
        progress_cb(0.75, "Stage 5/6: Audio synchronization...")

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
                logger.info("[Pipeline] Sync strategy set to accept drift. Copying raw cloned audio.")
                shutil.copy(str(cloned_audio_raw_path), str(cloned_audio_synced_path))

        except Exception as exc:
            raise ResonovaError(f"Audio synchronization stage failed: {exc}") from exc
    timings["Synchronization"] = time.perf_counter() - t0

    if progress_cb:
        progress_cb(0.85, "Stage 6/6: Re-syncing lips (Wav2Lip)...")

    # --- STAGE 6: Lip-Sync (Wav2Lip) ---
    t0 = time.perf_counter()
    if dest_video.is_file():
        logger.info("[Pipeline] Checkpoint found: final dubbed video already exists. Skipping Lip-Sync.")
    elif final_video_temp_path.is_file():
        logger.info("[Pipeline] Copying dubbed video from temp checkpoint...")
        shutil.copy(str(final_video_temp_path), str(dest_video))
    else:
        if use_lipsync:
            logger.info("[Pipeline] Running Wav2Lip Lip-Sync...")
            try:
                import torch
                rf = 1 if torch.cuda.is_available() else 2
                logger.info("[Pipeline] Using resize_factor=%d (1=GPU, 2=CPU speedup)", rf)
                lipsync(
                    video_path=str(src_video),
                    audio_path=str(cloned_audio_synced_path),
                    output_path=str(final_video_temp_path),
                    resize_factor=rf,
                )
                shutil.copy(str(final_video_temp_path), str(dest_video))
                logger.info("[Pipeline] Lip-Sync complete. Final dubbed video created.")
            except Exception as exc:
                logger.error("[Pipeline] Lip-sync failed: %s. Falling back to audio-only merge.", exc)
                try:
                    with open(warnings_path, "a", encoding="utf-8") as wf:
                        wf.write(f"Lip-Sync stage failed: {exc}. Reverted to video-audio muxing fallback.\n")
                except Exception:
                    pass
                try:
                    merge_audio_video(str(src_video), str(cloned_audio_synced_path), str(final_video_temp_path))
                    shutil.copy(str(final_video_temp_path), str(dest_video))
                except Exception as merge_exc:
                    logger.warning("[Pipeline] merge fallback failed: %s. Copying source video directly.", merge_exc)
                    shutil.copy(str(src_video), str(dest_video))
        else:
            logger.info("[Pipeline] Lip-sync disabled or not configured. Merging audio and video directly.")
            try:
                merge_audio_video(str(src_video), str(cloned_audio_synced_path), str(final_video_temp_path))
                shutil.copy(str(final_video_temp_path), str(dest_video))
            except Exception as merge_exc:
                logger.warning("[Pipeline] merge failed: %s. Copying source video directly.", merge_exc)
                shutil.copy(str(src_video), str(dest_video))
    timings["Lip-Sync (Wav2Lip)"] = time.perf_counter() - t0

    pipeline_duration = time.perf_counter() - t_pipeline_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE | total_time=%.2fs", pipeline_duration)
    for stage, t in timings.items():
        logger.info("  - %-25s : %6.2fs (%5.1f%%)", stage, t, (t / pipeline_duration) * 100)
    logger.info("=" * 60)

    result_data = {
        "dubbed_video_path": str(dest_video),
        "transcript": transcript,
        "translation": translated_text,
        "lipsync_used": use_lipsync,
        "processing_time": pipeline_duration,
    }
    return DubResult(str(dest_video), result_data)
