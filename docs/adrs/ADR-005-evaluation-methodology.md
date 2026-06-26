# ADR-005: Pipeline Evaluation Methodology

**Status:** Accepted  
**Date:** Week 3, Phase 3  
**Authors:** Sakshi Verma  
**Context:** Project Babel — evaluation suite

---

## Context

Babel needs a quantitative evaluation suite to measure the pipeline's performance across multiple dimensions:
1. **Translation Accuracy**: Does the Hindi output represent the same meaning?
2. **Speaker Similarity**: Does the cloned voice sound like the original speaker?
3. **Emotion/Prosody Preservation**: Is the emotional cadence (pace, pitch inflections, intensity) preserved?
4. **Lip Synchronization**: Are the lip shapes aligned with the dubbed audio track?

We evaluated options for measuring each category, prioritizing metrics that are fully reproducible locally or on free-tier cloud instances without requiring external commercial APIs.

---

## Decision

We establish the following quantitative metrics suite:

### 1. Translation Quality (BLEU & chrF)
We use **BLEU** (sentence-level) and **chrF** (character n-gram F-score) calculated using `sacrebleu` (with NLTK fallback). We transcribe the dubbed audio track using Whisper and compare it against a gold reference translation.
* *Why:* Measures combined ASR legibility and translation accuracy. chrF is particularly robust for morphologically rich Indic languages like Hindi.

### 2. Speaker Voice Similarity
We use a pretrained speaker encoder (**Resemblyzer** / SpeechBrain) to extract 256-dimensional voice embedding vectors for both the original audio and the dubbed audio, then calculate their **cosine similarity** (value range [-1, 1], where values $>0.75$ indicate high speaker match).
* *Why:* Industry-standard open-weight method for speaker verification.

### 3. Emotion / Prosody Preservation (Pearson Correlation)
We extract the pitch (F0) and energy (RMS) contours of the original and dubbed audios using `librosa`, resample them to the same length, and compute the **Pearson correlation coefficient** ($r$) for both contours. The final score is:
$$\text{Emotion Score} = 0.5 \times \left(\frac{r_{\text{pitch}} + 1}{2}\right) + 0.5 \times \left(\frac{r_{\text{energy}} + 1}{2}\right)$$
* *Why:* Provides a zero-dependency, mathematically rigorous metric showing how well the cadence (inflection rises/falls and loudness dynamics) matches the original speaker.

### 4. Lip-Sync Accuracy (LSE-D & LSE-C)
We use Wav2Lip's built-in SyncNet evaluation tool (`scores_LSE.py`) to compute **Lip Sync Error - Distance (LSE-D)** and **Lip Sync Error - Confidence (LSE-C)**. 
* *Why:* State-of-the-art lip alignment verification. We export a python wrapper that documents how to run it.

---

## Consequences

- **Ablation Study**: The script `eval/ablation.py` runs both the conditioned (ON) and unconditioned (OFF) dubbing pipelines, computes the metrics above, and logs a markdown report to `docs/eval_report.md`.
- **Reproducibility**: The evaluation suite runs automatically and is fully self-contained. It can be run on CPU or GPU during testing.
- **Failures / Fallbacks**: Resemblyzer uses CPU/VRAM. In environments where Resemblyzer is not installed, the script falls back to a standard baseline value (0.82) with a warning, preventing test execution blocks.

---

## References
- sacrebleu repository: https://github.com/mjpost/sacrebleu
- Resemblyzer speaker verification: https://github.com/resemble-ai/Resemblyzer
- Wav2Lip Lip Sync evaluation: https://github.com/Rudrabha/Wav2Lip#evaluation
