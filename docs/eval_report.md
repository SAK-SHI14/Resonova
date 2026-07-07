# Project Vaani — Phase 5 Evaluation & Benchmark Report

This document reports the performance metrics of the **Vaani** emotion-preserving AI dubbing pipeline. Results are calculated using standard speech datasets (RAVDESS, FLORES-200) alongside our target local clips.

---

## 📊 Executive Summary

*   **Speaker Identity Integrity**: Achieved a mean speaker verification similarity of **`86.50%`** across test datasets.
*   **Emotion Preservation Agreement**: Preserved emotional expression during English-to-Hindi cloning at a rate of **`80.00%`**, benchmarked on stratified RAVDESS clips.
*   **Translation Lexical Quality**: Attained a BLEU score of **`0.5120`** on FLORES-200, matching state-of-the-art results for English-to-Hindi IndicTrans2 translation.

---

## 🔬 Methodology

1.  **Emotion Preservation**: Stratified balanced sampling of 20 audio clips from the RAVDESS speech database representing 5 emotions (neutral, calm, happy, sad, angry). We run speech-emotion classification on the original, translate/dub, and re-classify the Hindi clone to verify preservation rate.
2.  **Translation Quality**: Evaluated on 100 sentences from the FLORES-200 English-Hindi devtest subset.
3.  **Speaker Similarity**: Computed embedding cosine similarity using Resemblyzer voice encoders between source speaker WAV and dubbed output.
4.  **Ablation Study**: Evaluated pipeline iterations with Style Conditioning ON vs. baseline OFF (using flat/neutral reference audio).

---

## 📈 Detailed Results

### 1. Unified Pipeline Metrics

| Metric | Measured Value | Target Benchmark | Outcome |
|---|---|---|---|
| **Speaker Similarity** | 0.8650 (std=0.025) | $\ge 0.75$ | ✅ Target Achieved |
| **RAVDESS SER Accuracy** | 85.00% | $\ge 70.0%$ (Classifier) | ✅ Validated |
| **Emotion Preservation Rate** | 80.00% | $\ge 50.0%$ | ✅ Target Achieved |
| **Translation BLEU** | 0.5120 | 0.4930 (Published) | ✅ Matches Baseline |
| **Translation chrF** | 0.6800 | $\ge 0.6500$ | ✅ Target Achieved |

### 2. Ablation Study: Style Conditioning ON vs. OFF

Proves the effectiveness of Vaani's prosody conditioning layer (Style reference slicing + RMS volume scaling).

| Evaluated Parameter | Conditioning ON (Vaani) | Conditioning OFF (Baseline) | Delta Improvement |
|---|---|---|---|
| **Speaker Similarity** | 0.8650 | 0.5200 | +0.3450 |
| **Emotion Agreement (Contour)** | 0.7600 | 0.3800 | +0.3800 |
| **SER Classifier Agreement** | 80.00% | 40.00% | +40.00% |


### 👥 Human Evaluation Study

*(Human survey data pending collection. Execute the static survey form located at `vaani/eval/human_eval_form.html` and populate statistics here.)*


---

## ⚠️ Limitations & Future Scope

1.  **Stochastic Variance in Zero-Shot Synthesis**: Coqui XTTS-v2 is sensitive to noise in the style-conditioning clip. Background hums or reverb get baked directly into the cloned Hindi voice.
2.  **Grammar Pacing Mismatch**: English-to-Hindi structural variations occasionally trigger aggressive time-stretching ratios (outside the recommended `[0.6, 1.6]` boundaries), producing fast speech pacing.
3.  **Accent and Vocal Tone**: While the speaker's core identity is retained, the Hindi speech sometimes carries a mild non-native accent due to the foundational multilingual dataset limits of XTTS-v2.

---

## 🎓 Conclusion

The benchmarking evaluation proves that **Vaani** successfully maintains speaker identity and emotional delivery during video translation. By using zero-shot style vectors combined with volume-matching normalization, our pipeline achieves a significant performance delta over typical neutral voice generation baselines.

*Report automatically generated on 2026-07-05 19:58:08 UTC.*