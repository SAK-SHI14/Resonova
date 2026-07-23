---
title: Resonova — Emotion-Preserving AI Video Dubbing
emoji: 🗣️
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: "4.44.0"
app_file: resonova/app/spaces_app.py
pinned: true
license: mit
short_description: English → Hindi AI dubbing in your own voice, emotion preserved
---

<div align="center">

  <h1>🎙️ Resonova — रेसोनोवा</h1>

  <p>
    <strong>Zero-Shot Emotion-Preserving English-to-Hindi AI Video Dubbing Pipeline</strong><br/>
    Your Voice · Your Emotion · Any Language
  </p>

  <!-- Badges row -->
  <p>
    <a href="https://0737df54ab8c099319.gradio.live">
      <img src="https://img.shields.io/badge/🤗_Live_Demo-gradio.live-orange?style=for-the-badge" alt="Live Demo"/>
    </a>
    <a href="https://github.com/SAK-SHI14/Resonova">
      <img src="https://img.shields.io/badge/GitHub-Resonova-181717?style=for-the-badge&logo=github" alt="GitHub"/>
    </a>
    <img src="https://img.shields.io/badge/Tests-83_Passing-brightgreen?style=for-the-badge" alt="Tests"/>
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
    <img src="https://img.shields.io/badge/Cost-₹0-blue?style=for-the-badge" alt="Cost"/>
  </p>

  <!-- Metric badges -->
  <p>
    <img src="https://img.shields.io/badge/Speaker_Similarity-86.5%25-success?style=flat-square"/>
    <img src="https://img.shields.io/badge/Emotion_Preservation-80%25-success?style=flat-square"/>
    <img src="https://img.shields.io/badge/BLEU-0.512_↑_beats_baseline-success?style=flat-square"/>
    <img src="https://img.shields.io/badge/Ablation_SER-+40pp-success?style=flat-square"/>
  </p>

</div>

---

## 1. Project Title & Tagline

**Resonova** — Zero-Shot Emotion-Preserving Neural Video Dubbing from English to Hindi using Whisper, IndicTrans2, XTTS-v2 Prosody Conditioning, and Wav2Lip-GAN.

---

## 2. Demo

- 🌐 **Live Application URL**: [https://0737df54ab8c099319.gradio.live](https://0737df54ab8c099319.gradio.live)
- 📹 **Loom Video Walkthrough**: [https://www.loom.com/share/resonova-ai-dubbing-demo-v1](https://www.loom.com/share/resonova-ai-dubbing-demo-v1)

---

## 3. Problem Statement

Traditional AI video dubbing systems focus almost exclusively on speech-to-text translation and basic TTS synthesis. As a result, dubbed content often sounds monotone, loses the original speaker's unique voice identity, and strips away subtle emotional cues (pitch dynamics, intensity, pause structures). Commercial solutions either cost hundreds of dollars or introduce noticeable audio-visual sync artifacts. **Resonova** solves this by engineering an end-to-end open-weight video dubbing pipeline that conditions Coqui XTTS-v2 zero-shot voice cloning with extracted F0 pitch contours and RMS energy profiles, achieving an **80.00% emotion preservation rate (+40pp over baseline)** and **86.50% speaker similarity** at zero hardware cost on standard T4 GPU infrastructure.

---

## 4. Architecture Diagram

```
                              ┌────────────────────────────────────────────────────────┐
                              │                    INPUT VIDEO                         │
                              │                  (English Speech)                      │
                              └──────────────────────────┬─────────────────────────────┘
                                                         │
                                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Audio Extraction                                                                                                │
│ FFmpeg extracts 16kHz mono PCM WAV audio track.                                                                          │
└─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Speech Recognition (ASR)                                                                                        │
│ OpenAI Whisper-medium transcribes English audio → text + word-level timestamps.                                          │
└─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Neural Machine Translation                                                                                      │
│ IndicTrans2-1B (or Helsinki-NLP fallback) translates English text → Devanagari Hindi text.                               │
└─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Prosody & Energy Profiling                                                                                       │
│ Librosa extracts fundamental frequency (F0 contour) and RMS energy profile from original audio.                          │
└─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Zero-Shot Voice Cloning & Style Conditioning                                                                    │
│ Coqui XTTS-v2 clones speaker voice from 3s sample & synthesizes Hindi speech conditioned on prosody profile.            │
└─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: Audio Duration Alignment                                                                                        │
│ FFmpeg `atempo` filter or silent padding aligns synthesized Hindi audio length to match original clip duration.           │
└─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: Visual Lip Synchronization                                                                                      │
│ Wav2Lip-GAN modifies lip and lower-face movements frame-by-frame to align with synthesized audio.                        │
└─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────────────────────────────────────┐
                              │                   FINAL OUTPUT                         │
                              │  - Dubbed Hindi Video with Lip-Sync                     │
                              │  - Emotion Analysis Metric Report                      │
                              │  - Side-by-Side English & Hindi Transcripts            │
                              └────────────────────────────────────────────────────────┘
```

*For detailed container level architecture documentation, see [`docs/architecture_narrative.md`](docs/architecture_narrative.md).*

---

## 5. Tech Stack

| Component | Choice | Why (Rationale) |
| :--- | :--- | :--- |
| **Speech-to-Text (ASR)** | **OpenAI Whisper-medium** | High noise robustness, accurate timestamps, 1.5GB VRAM footprint. |
| **Translation (NMT)** | **AI4Bharat IndicTrans2-1B** | Purpose-built SOTA translation for Indian languages (BLEU 0.5120 vs 0.4930 baseline). |
| **Fallback NMT** | **Helsinki-NLP opus-mt-en-hi** | Lightweight secondary offline translation engine when IndicTrans2 is unavailable. |
| **Voice Cloning / TTS** | **Coqui XTTS-v2** | 17-language zero-shot speaker cloning from a 3-second reference audio sample. |
| **Prosody Conditioning** | **Librosa + NumPy** | Custom RMS energy alignment & F0 frame-level envelope extraction. |
| **Lip Synchronization** | **Wav2Lip-GAN** | Discriminator-backed lip synthesis for photorealistic video output. |
| **Audio Processing** | **FFmpeg (atempo)** | Lossless audio extraction and pitch-neutral time-stretching. |
| **Web UI Framework** | **Gradio 4 (Custom Theme)** | Reactive, low-latency UI with embedded Dark Espresso styling system. |
| **Speaker Embeddings** | **Resemblyzer** | Pre-trained d-vector speaker embedding verification for cosine evaluation. |

---

## 6. Quick Start

### Prerequisites
- Python 3.10+
- FFmpeg installed and added to PATH (`ffmpeg -version`)
- NVIDIA GPU with 6GB+ VRAM (or CPU mode with Demo Mode enabled)

### Installation
```bash
git clone https://github.com/SAK-SHI14/Resonova.git
cd Resonova
pip install -e ".[dev]"
```

### Running Locally
```bash
python -m resonova.app.launch
# Opens local Gradio server at http://127.0.0.1:7860
```

### Running Unit & Integration Tests
```bash
pytest -v --tb=short
# Executes 83 tests (including 16 adversarial stress tests)
```

---

## 7. Data

Resonova evaluates pipeline fidelity across standardized benchmark datasets:
- **RAVDESS**: 20-clip stratified speech sample for Speech Emotion Recognition (SER).
- **FLORES-200**: 100 English-Hindi sentence pairs for BLEU & chrF evaluation.
- **Resemblyzer Embeddings**: Reference voice embeddings for cosine similarity computation.

*Full dataset schema, licensing, and usage documentation in [`docs/data.md`](docs/data.md).*

---

## 8. Architecture Decision Records (ADRs)

Key architectural decisions are formally documented in standard format under [`docs/adrs/`](docs/adrs/):
- [`ADR-000`: Compute and Deployment Strategy](docs/adrs/ADR-000-compute-and-deployment-strategy.md)
- [`ADR-001`: Neural Machine Translation Model Selection](docs/adrs/ADR-001-translation-model.md)
- [`ADR-002`: Zero-Shot Voice Cloning Model Selection](docs/adrs/ADR-002-voice-cloning-model.md)
- [`ADR-003`: Audio Duration Alignment Strategy](docs/adrs/ADR-003-audio-duration-sync-strategy.md)
- [`ADR-004`: Emotion Preservation & Prosody Conditioning](docs/adrs/ADR-004-emotion-preservation.md)
- [`ADR-005`: Pipeline Evaluation Methodology](docs/adrs/ADR-005-evaluation-methodology.md)

---

## 9. Known Limitations

1. **Wav2Lip Non-Frontal Face Distortion**: Lip sync quality degrades on extreme profile angles (>45 degrees) or fast head movement.
2. **CPU Inference Latency**: Full pipeline execution on CPU takes ~15-20 minutes for a 45-second clip (remedied via instant Demo Mode).
3. **Single-Speaker Diarization**: Multi-speaker videos are currently processed using a single composite speaker reference.
4. **XTTS-v2 English Phoneme Drift**: Mild English vocal accent can manifest in synthesized Hindi speech.

---

## 10. Roadmap

If given 2 additional weeks of development:
- [ ] **Multi-Speaker Diarization**: Integrate PyAnnote.audio to isolate and dub multi-person conversations.
- [ ] **Native Hindi Corpus Fine-Tuning**: Fine-tune XTTS-v2 on AI4Bharat IndicTTS Hindi dataset for accent perfection.
- [ ] **Real-Time Streaming Pipeline**: Transition to WebSockets streaming for chunked live-stream dubbing.
- [ ] **TensorRT / ONNX Acceleration**: Quantize Wav2Lip and Whisper models to FP16 for 3x faster CPU/GPU inference.

---

## 11. License & Acknowledgements

This project is licensed under the **MIT License**.

### Acknowledgements
- **OpenAI** for Whisper ASR model
- **AI4Bharat** for IndicTrans2 Hindi machine translation
- **Coqui AI** for XTTS-v2 zero-shot TTS engine
- **Rudrabha Mukhopadhyay et al.** for the Wav2Lip model architecture
