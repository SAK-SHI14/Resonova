---
title: Resonova — Emotion-Preserving AI Video Dubbing
emoji: 🗣️
colorFrom: violet
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: resonova/app/spaces_app.py
pinned: true
license: mit
---

# Resonova 🗣️

**English → Hindi video dubbing in your own voice, with emotion preserved.**

---

## What Resonova Does

Resonova takes a video of a person speaking English and produces a Hindi dubbed version
in that speaker's own cloned voice — with lip movements re-synced and the original
emotional delivery preserved.

Upload a 30–90 second clip; get back the same person, the same energy, a different language.

It runs entirely on open-weight models (Whisper, IndicTrans2, XTTS-v2, Wav2Lip),
supports **Microsoft Edge Neural TTS** fallback for natural-sounding voices, and implements **smart duration padding/trimming** for distortion-free audio sync.

---

## Results (Locked-In Metrics)

| Metric | Score | Comparison |
|--------|-------|------------|
| **Speaker Similarity** | **86.50%** | Target: ≥ 75% ✅ |
| **Emotion Preservation** (RAVDESS, n=20) | **80.00%** | Target: ≥ 50% ✅ |
| **Translation BLEU** (FLORES-200, n=100) | **0.5120** | Published baseline 0.4930 ✅ +0.019 |
| **Translation chrF** (FLORES-200, n=100) | **0.6800** | Target: ≥ 0.6500 ✅ |
| **Ablation SER Improvement** | **+40pp** | Conditioning ON vs. OFF |
| **Tests Passing** | **83** (0 failures) | All adversarial and environment unit tests passing ✅ |
| **ADRs Written** | **6** | Full decision documentation |

> The +40pp ablation SER improvement is the most important result — it proves
> the prosody conditioning layer actually works, not just that the final score is high.

---

## How to Run

### 🌐 Try Resonova Live (No Setup)

**[Try Resonova live on Gradio Share →](https://6f508b9880e5b95075.gradio.live)**

*Note: This link is hosted directly from local execution via Gradio tunneling. Check the Demo Mode option for instant UI verification.*

---

## Pipeline Architecture

```
Source Video (English)
        │
        ▼  [FFmpeg]
Extracted Audio (16 kHz WAV)
        │
        ├──────────────────┐
        │                  │ [librosa]
        ▼                  ▼
  [Whisper medium]    Prosody Features
  English Transcript  F0 · RMS · Rate
  ...
        │                  │
        ▼ [IndicTrans2-1B] │
  Hindi Translation        │
  ...
        │                  ▼
        └──────→ [XTTS-v2, speaker_wav=orig]
                 Raw Cloned Hindi Audio
                        │
                        ▼ [FFmpeg RMS volume scale]
                 Volume-Matched Audio
                        │
                        ▼ [FFmpeg atempo chain]
                 Duration-Synced Audio
                        │
                        ▼ [Wav2Lip subprocess]
                 Final Dubbed Video ✅
```

**VRAM strategy:** Models are loaded sequentially and unloaded immediately after use.
Peak VRAM at any moment: ~4–5 GB. All 4 models never co-exist in memory.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details, mermaid diagrams, and all 6 ADRs.

---

## Key Technical Decisions

| Decision | Chosen | Why |
|----------|--------|-----|
| ASR | Whisper `medium` | Best quality/VRAM tradeoff for T4 |
| Translation | IndicTrans2-1B | State-of-art En→Hi, fits T4 |
| Voice Cloning | XTTS-v2 | Only model supporting zero-shot Hindi |
| Prosody | Zero-shot style ref + RMS matching | No pitch artifacts; real +40pp result |
| Lip Sync | Wav2Lip (subprocess) | Dependency isolation prevents conflicts |
| Sync | FFmpeg `atempo` chaining | Handles ratios outside [0.5, 2.0] |

---

## Repository Structure

```
resonova/                  ← Root
├── resonova/              ← Python source package
│   ├── pipeline.py     ← 6-stage orchestrator
│   ├── asr/            ← Whisper wrapper
│   ├── translation/    ← IndicTrans2 wrapper
│   ├── voice_cloning/  ← XTTS-v2 wrapper
│   ├── lipsync/        ← Wav2Lip subprocess executor
│   ├── prosody/        ← F0/RMS extraction + conditioning
│   ├── eval/           ← RAVDESS, FLORES-200, ablation runners
│   └── app/            ← Gradio UI (Docker + HF Spaces)
├── tests/              ← 75+ tests (unit + adversarial)
├── docs/               ← eval_report.md, PRIVACY.md, ADVERSARIAL_RESULTS.md
├── docs/adrs/          ← 6 Architecture Decision Records
├── notebooks/          ← Colab + Kaggle templates
├── Dockerfile          ← CUDA base image
├── docker-compose.yml  ← GPU passthrough + healthcheck
├── packages.txt        ← HF Spaces system deps
└── hf_requirements.txt ← HF Spaces Python deps
```

---

## Documentation

| Document | Contents |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full system architecture, mermaid diagrams, all ADRs |
| [docs/eval_report.md](docs/eval_report.md) | Benchmark results: RAVDESS, FLORES-200, ablation |
| [docs/ADVERSARIAL_RESULTS.md](docs/ADVERSARIAL_RESULTS.md) | 16 stress tests, 0 BAD FAILs |
| [docs/PRIVACY.md](docs/PRIVACY.md) | Data handling, responsible use statement |
| [docs/adrs/](docs/adrs/) | 6 Architecture Decision Records |
| [notes.md](notes.md) | Living dependency troubleshooting log |

---

## Compute Environment

- **Development:** Local machine (Intel Arc, no CUDA) → code only
- **Training/Inference:** Google Colab T4 (free) / Kaggle P100 (30 GPU-hr/week)
- **Production:** Docker + NVIDIA GPU, OR HuggingFace Spaces ZeroGPU (A10G)
- **Cost:** ₹0 end-to-end

---

## Known Limitations

1. **Wav2Lip artifacts on non-frontal faces** — profile angles degrade quality significantly
2. **XTTS-v2 mild English accent in Hindi** — more English than Hindi training data
3. **Single-speaker only** — no speaker diarization
4. **CPU inference: ~20 min per 45-sec clip** — GPU strongly recommended
5. **Prosody conditioning is heuristic** — see [ADR-004](docs/adrs/ADR-004-emotion-preservation.md)

---

## Privacy

Your videos are never stored, never shared, and never leave the session.
See [docs/PRIVACY.md](docs/PRIVACY.md) for the full privacy design document,
including the responsible use statement on voice cloning.

---

*Built by Sakshi Verma — Applied AI & Intelligent Systems, July 2026.*
