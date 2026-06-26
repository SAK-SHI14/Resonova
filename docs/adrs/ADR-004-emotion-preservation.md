# ADR-004: Emotion & Prosody Preservation Strategy

**Status:** Accepted  
**Date:** Week 3, Phase 3  
**Authors:** Sakshi Verma  
**Context:** Project Babel — emotion-preserving dubbing pipeline (English → Hindi)

---

## Context

Most modern speech-to-speech dubbing pipelines sound flat and robotic. They convert text to speech using generic speaker profiles that lack the emotional cadence (inflections, rate of speech, emphasis, and pauses) of the original speaker in the source video. 

Babel must preserve this emotional color to make dubbing feel natural and engaging. We evaluated two primary methods to capture and apply emotional prosody to the synthesized Hindi speech.

---

## Options Evaluated

### Option A: Zero-Shot Neural Style Transfer (Selected)

We leverage the native latent style conditioning capability of **Coqui XTTS-v2**. 
During the TTS synthesis stage, instead of using a standard flat neutral voice sample as the speaker reference, we slice the **exact segment of original audio** corresponding to the source speech and pass it as the reference speaker WAV.

* **Pros:**
  - Transfers complex speaker style details (breath, vocal tension, emotional coloring, pitch range) directly through the model's neural layers.
  - Zero-shot; requires no speaker fine-tuning or secondary emotional voice models.
  - Retains the exact voice identity of the speaker while transferring the tone.
* **Cons:**
  - XTTS-v2's generation is stochastic and sometimes accentuates noise present in the reference clip.

### Option B: Post-Synthesis Digital Contour Warping

Synthesize flat speech, extract the pitch (F0) and energy (RMS) contours from the original audio using `librosa`, and warp/shift the synthesized audio's pitch and energy dynamically (using vocoders like WORLD or PSOLA) to match the original contours.

* **Pros:**
  - High degree of mathematical matching of pitch rises and falls.
* **Cons:**
  - Digital pitch shifting and time warping introduce severe phase distortion and robotic/metallic artifacts.
  - Extremely difficult to align because grammar structures change word positions (e.g. English is SVO, Hindi is SOV), making direct time-aligned contour warping mathematically incorrect.

---

## Decision

**Chose Option A (Zero-Shot Neural Style Transfer via Original Reference Segment).**

We use the original speaker's active audio segment directly as the XTTS-v2 reference. This transfers voice identity and the speaker's emotional state (happy, sad, angry). To prevent volume mismatches, we also implement a volume alignment utility (`apply_prosody_conditioning`) which matches the RMS energy of the synthesized audio with the original's mean RMS using an `ffmpeg` volume filter.

Digital contour warping is rejected due to grammar-reordering misalignment and audio quality degradation.

---

## Consequences

- **Implementation**: The pipeline uses the extracted `extracted_audio.wav` from Stage 1 as the speaker reference. During ablation testing, passing `voice_reference_path` overrides this with a neutral reference to simulate "No Conditioning."
- **Quality**: Inflections are transferred naturally without phase artifacts.
- **VRAM/GPU footprint**: Zero additional GPU overhead (we reuse the same XTTS-v2 model and just change the reference file argument).

---

## References
- Coqui XTTS-v2 architecture: https://docs.coqui.ai/en/latest/models/xtts.html
- Inflection transfer in speech synthesis: https://arxiv.org/abs/2301.12599
