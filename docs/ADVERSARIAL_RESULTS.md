# Vaani — Adversarial Test Results

**Date:** July 2026 | **Test Suite:** `tests/test_adversarial.py` | **Tests:** 16

We deliberately stress-tested our own system to find every edge case
before a recruiter or interviewer could. The table below documents every
test case, its classification, and what it proved.

---

## Summary Table

| # | Test Case | Classification | Notes |
|---|-----------|---------------|-------|
| 1 | Very short reference clip (5s) | **GRACEFUL FAIL** ✅ | `VoiceCloningError` with minimum duration in message |
| 2 | Silent audio — no speech | **GRACEFUL FAIL** ✅ | `TranscriptionError("empty transcript")` raised |
| 3 | Empty string passed to translate() | **GRACEFUL FAIL** ✅ | `TranslationError` raised on empty input |
| 4 | Unsupported language code | **GRACEFUL FAIL** ✅ | `ValueError` listing supported codes |
| 5 | Hindi input → English→Hindi pipeline | **PASS** ✅ | Pipeline completes; quality degraded but no crash |
| 6 | Missing audio file for ASR | **GRACEFUL FAIL** ✅ | `FileNotFoundError` with path in message |
| 7 | Missing reference audio for cloning | **GRACEFUL FAIL** ✅ | `FileNotFoundError` raised immediately |
| 8 | Monotone flat delivery | **PASS** ✅ | `extract_prosody()` returns zero std, no NaN |
| 9 | Extreme duration ratio (4.5× or 0.15×) | **PASS** ✅ | `atempo` chain correct: `atempo=2.0,atempo=2.0,atempo=1.1250` |
| 10 | Checkpoint resumption after mid-pipeline failure | **PASS** ✅ | ASR/Translation skipped when checkpoint files exist |
| 11 | Output files are non-empty after pipeline | **PASS** ✅ | Output file existence and size validated |
| 12 | Non-existent input video | **GRACEFUL FAIL** ✅ | `FileNotFoundError` — first validation in `dub_video()` |
| 13 | `unload_all_models()` with empty cache | **PASS** ✅ | No error even when caches are empty and no GPU present |
| 14 | Audio < 100ms passed to prosody extractor | **GRACEFUL FAIL** ✅ | `ProsodyError("too short")` raised |
| 15 | Wav2Lip env vars not set | **GRACEFUL FAIL** ✅ | `LipSyncError` naming `WAV2LIP_REPO_PATH` |
| 16 | `atempo` ratio near 1.0 copies file directly | **PASS** ✅ | `shutil.copy` called; no FFmpeg subprocess |

**Result: 0 BAD FAILs. All 16 cases are PASS or GRACEFUL FAIL.**

---

## What These Results Mean

We deliberately stress-tested our own system across 16 edge cases covering
silent audio, missing files, impossible language codes, extreme duration ratios,
monotone delivery, and mid-pipeline failures. Every failure mode is either handled
correctly (PASS) or surfaces a clear, typed exception with a human-readable message
(GRACEFUL FAIL) — none silently crash or produce garbage output (BAD FAIL).

This is how you build systems that are credible in front of a technical interviewer:
not by hiding failure modes, but by knowing exactly where they are and proving they
are handled.

---

## Fixes Made During Adversarial Testing

No BAD FAILs were found. All failure paths were already implemented:
- `transcribe()` validates its output (`text.strip()`) before returning
- `translate()` validates its input before calling the model
- `clone_voice()` checks reference clip duration vs. `MIN_REFERENCE_DURATION_S = 6.0`
- `extract_prosody()` guards `duration < 0.1` before librosa calls
- `lipsync()` validates `WAV2LIP_REPO_PATH` and `WAV2LIP_CHECKPOINT_PATH` before subprocess
- `dub_video()` validates input path before any stage begins
- `time_stretch_audio()` handles ratios outside `[0.5, 2.0]` by chaining `atempo` filters

---

## Proposed Future Work

The following GRACEFUL FAILs would require significant engineering to convert to PASS:

1. **Silent audio → GRACEFUL FAIL** — converting this to PASS would require
   voice activity detection (VAD) pre-processing (e.g., Silero VAD) to detect
   and reject silence before transcription. Out of scope for this implementation.

2. **Non-frontal face input → degraded Wav2Lip output** — Wav2Lip artifacts on
   profile angles are a fundamental model limitation, not a code bug. Fixing this
   would require a different lip-sync model (e.g., Video Retalking or SadTalker)
   which have different dependency trees and would require re-evaluation.

3. **Hindi input → low-quality output** — automatic language detection and rejection
   of wrong-language source clips would require pre-pipeline language identification
   (e.g., `langdetect` on the Whisper transcript, with a threshold check).

4. **Very short reference clips (5s)** — XTTS-v2's minimum is 6 seconds by design.
   Converting this to PASS would require using a voice enhancement model to extend
   short clips, or accepting lower voice fidelity for short references. Out of scope.
