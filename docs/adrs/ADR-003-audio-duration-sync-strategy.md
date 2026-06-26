# ADR-003: Audio-Video Duration Synchronization Strategy

**Status:** Accepted  
**Date:** Week 2, Phase 2  
**Authors:** Sakshi Verma  
**Context:** Project Mimi — AI dubbing pipeline (English → Hindi)

---

## Context

When dubbing speech from English to Hindi:
1. **Density mismatch**: Hindi text is structurally longer and syllables take more time to pronounce compared to equivalent English meaning (typically 15% to 35% longer).
2. **Zero-shot TTS variation**: Synthesized audio from XTTS-v2 has its own intrinsic pacing which depends on the punctuation, speaker voice style, and model parameters, and rarely matches the original duration.

If this duration mismatch is not resolved, the pipeline suffers from:
- **Audio-Video drift**: Lips finish moving or continue moving after speech has stopped.
- **Video truncation**: The source video ends before the cloned audio completes.
- **Recruiter evaluation issues**: Robotic or unsynchronized pacing directly degrades the "Product Sense" metric.

We evaluated three options to address this duration mismatch.

---

## Options Evaluated

### Option A: Dynamic Global Time-Stretching (Selected)

We compute the speed ratio:
$$\text{ratio} = \frac{\text{raw cloned audio duration}}{\text{original video duration}}$$
And use `ffmpeg`'s native `atempo` filter to stretch or compress the cloned audio file to match the original duration.

* **Pros:**
  - Preserves original video duration exactly.
  - Keeps video and audio in lockstep (frame-perfect lip synchronization).
  - Standard `atempo` filter handles pitch correction automatically (no pitch-shifting artifacts).
  - Simple, robust, and works out of the box in standard Linux/cloud runtimes.
* **Cons:**
  - If the ratio is too far from `1.0` (e.g., `< 0.6` or `> 1.6`), speech sounds unnaturally fast or slow.

### Option B: Accept Drift (Video Padding/Trimming)

Retain the natural pace of the synthesized cloned audio. If the audio is longer than the video, pad the video by freezing its last frame; if the audio is shorter, trim the video.

* **Pros:**
  - Synthesized speech maintains its exact natural pacing.
* **Cons:**
  - Destroys original video pacing. Freezing the final frame looks unnatural and unprofessional on short demo clips.
  - Trimmed clips may clip out important facial expressions or visual context.

### Option C: Word/Segment-Level Time-Stretching (Deferred)

Use Whisper word/segment timestamps to align each translated segment with the original speaker's segments by stretching speech locally.

* **Pros:**
  - High degree of alignment throughout the timeline.
* **Cons:**
  - Highly complex. Requires translation segments to map 1:1 to source segments (which is rare since translation changes word order).
  - Leads to jarring local speech speed fluctuations.

---

## Decision

**Adopt Option A (Dynamic Global Time-Stretching) as the default pipeline strategy.** We also implement Option B (Accept Drift) as a configuration parameter (`sync_strategy="accept_drift"`) for cases where natural speech rate is preferred over frame-lock.

### Rationale

For portfolio purposes and 30-90s clips:
1. **Intelligibility**: In practice, Hindi translation length differences are usually within a 0.8–1.3x speed window. Inside this range, `atempo` processing sounds highly natural and legible.
2. **Visual Consistency**: Frame-lock ensures the video looks like a professional dub rather than a mismatched translation.
3. **Robustness**: Global time-stretching avoids segment-mapping heuristics that fail when translation shifts grammar structures (SVO vs SOV).

We establish an warning threshold: if the speed ratio is outside the `[0.6, 1.6]` range, the pipeline logs a warning advising the user to adjust the translation length, but proceeds to generate the clip anyway to avoid hard failures.

---

## Consequences

- **Implementation**: Written using an `ffmpeg` subprocess in `mimi/pipeline.py` with custom logic to chain `atempo` filters (since a single `atempo` filter in ffmpeg only supports ranges between 0.5 and 2.0).
- **Quality**: Synthesized voices remain pitch-corrected, and the final MP4 video has perfectly aligned lip-sync and audio track length.

---

## References
- FFmpeg filters documentation (atempo): https://ffmpeg.org/ffmpeg-filters.html#atempo
- Rubberband audio stretcher library (alternative): https://breakfastquay.com/rubberband/
