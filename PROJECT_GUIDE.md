# 🎙️ Resonova — Complete Project Breakdown & Guide

**Resonova** is an **Emotion-Preserving AI Video Dubbing System** that translates English videos into Hindi while keeping the speaker's original voice, emotion, pitch, and realistic lip synchronization.

---

## 🌟 Part 1: Simple Explainable Project Breakdown

Imagine taking an English video of someone speaking and dubbing it into Hindi. Usually, dubbed videos sound robotic, lose the original speaker's unique voice tone, sound flat (losing emotion), and look awkward because the mouth movements don't match the new Hindi words.

**Resonova fixes all of these problems using a 6-stage AI pipeline:**

```
                               ┌────────────────────────────────────────────────────────┐
                               │                    INPUT VIDEO                         │
                               │                  (English Speech)                      │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Audio Extraction                                                                                                │
│ FFmpeg extracts the clean audio track (.wav) from the uploaded video.                                                     │
└─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Speech-to-Text Transcription (ASR)                                                                              │
│ OpenAI Whisper-base listens to the English audio and converts it into exact English text + timestamped segments.          │
└─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Machine Translation                                                                                             │
│ IndicTrans2 (or Helsinki-NLP fallback) translates English text into natural Hindi (Devanagari script).                   │
└─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Zero-Shot Voice Cloning & Text-to-Speech (TTS)                                                                 │
│ Coqui XTTS-v2 takes a 3-second sample of the original speaker's voice and speaks the translated Hindi text in THEIR     │
│ exact voice tone, pitch, and timbre.                                                                                    │
└─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Emotion Preservation & Audio-Video Synchronization                                                              │
│ - Energy Matching: Aligns RMS volume contours between original English and synthetic Hindi.                              │
│ - Time Stretching: Adjusts Hindi speech speed using FFmpeg `atempo` or smart padding so audio matches original length.    │
└─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: Visual Lip Synchronization                                                                                      │
│ Wav2Lip-GAN modifies the lip and lower-face movements in the original video frame-by-frame so the speaker's lips move   │
│ seamlessly to match the new Hindi speech!                                                                                │
└─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │                   FINAL OUTPUT                         │
                               │  - Dubbed Hindi Video with Lip-Sync                     │
                               │  - Emotion Analysis Report Card (Similarity Score)      │
                               │  - Side-by-Side English & Hindi Transcripts            │
                               └────────────────────────────────────────────────────────┘
```

---

## 🤖 Part 2: AI Models, Core Functions & Technical Features

### 1. Key AI Models Used
| Pipeline Stage | AI Model / Tool | Role in Project | Why Selected |
| :--- | :--- | :--- | :--- |
| **Speech Recognition** | **OpenAI Whisper-base** | Converts spoken English audio into accurate text. | High robustness against background noise and accents. |
| **Translation** | **AI4Bharat IndicTrans2** | Translates English text into natural Hindi (`hin_Deva`). | State-of-the-art BLEU score (0.5120) for Indian languages (beats Helsinki-NLP baseline 0.4930). |
| **Voice Cloning** | **Coqui XTTS-v2** | Clones the original speaker's voice from 3 sec of audio and speaks Hindi. | Supports 17 languages with zero-shot speaker cloning. |
| **Lip Synchronization** | **Wav2Lip-GAN** | Modifies lip movements in the video to match new Hindi audio. | Uses GAN discriminator for realistic face synthesis without artifacts. |
| **Speaker Evaluation** | **Resemblyzer** | Extracts voice embeddings to calculate cosine similarity. | Achieves **86.50% speaker similarity** score. |
| **Emotion Recognition** | **RAVDESS Stratified Model** | Classifies emotion across 8 states (happy, sad, neutral, anger, etc.). | Achieves **80.00% emotion preservation** score (+40pp SER improvement). |

---

### 2. Core Audio & Signal Processing Functions
- **RMS Volume Energy Alignment**: Measures original audio energy profile and scales the cloned voice's volume to match high/low emotional dynamics.
- **FFmpeg Intelligent Time-Stretching**:
  - `ratio ≈ 1.0`: Direct copy.
  - `0.65 ≤ ratio ≤ 1.50`: Smooth time-stretch using FFmpeg `atempo` filter.
  - `ratio < 0.65` (audio too long): Trims audio with smooth 100ms crossfade to prevent overlap.
  - `ratio > 1.50` (audio too short): Appends subtle natural silence instead of unnatural fast speech.
- **Wav2Lip CPU Speedup (`resize_factor=2`)**: Scales video frames down during inference on CPU to achieve a 4x speedup during testing.

---

### 3. Frontend & User Interface Features
- **Gradio 4 Web Application**: Modern interactive web interface.
- **Permanent Dark Espresso Theme**: Deep rich dark mode (`#1C1917` background, `#262322` cards, `#F4A261` amber accents).
- **Handwritten Display Title**: Custom Google Font (`Dancing Script` / `Great Vibes`) for visual polish.
- **⚡ DEMO MODE (Mock Output)**: Checkbox feature allowing instant 3-second UI demonstration during live presentations without waiting for CPU inference.
- **Interactive Emotion Report Card**: Generates a side-by-side visual chart comparing pitch, energy contours, and overall emotion preservation scores.

---

## 📁 Part 3: Folder Structure & File-by-File Role Guide

Here is the exact explanation of every folder and file in the codebase:

```
resonova/
├── resonova/                     # Core Python Package
│   ├── app/                      # Web UI & Application Interface
│   │   ├── app.py                # Main Gradio application definition & layout
│   │   ├── launch.py             # Script to launch local web app server
│   │   ├── report_card.py        # Generates visual emotion evaluation chart
│   │   ├── spaces_app.py         # Entry point for cloud deployment
│   │   └── static/
│   │       └── resonova.css      # Custom Dark Espresso design tokens & styling
│   ├── asr/                      # Automatic Speech Recognition (Stage 2)
│   │   └── transcribe.py         # Runs OpenAI Whisper to extract English text
│   ├── translation/              # Machine Translation (Stage 3)
│   │   └── translate.py          # Translates English to Hindi (IndicTrans2 / Helsinki)
│   ├── voice_cloning/            # Zero-Shot TTS (Stage 4)
│   │   └── clone_voice.py        # Synthesizes Hindi audio in speaker's cloned voice
│   ├── lipsync/                  # Visual Synchronization (Stage 6)
│   │   └── lipsync.py            # Executes Wav2Lip-GAN to align lip movements
│   ├── prosody/                  # Emotion & Energy Conditioning (Stage 5)
│   │   ├── extract.py            # Measures pitch & energy contours (RMS)
│   │   └── conditioning.py       # Aligns audio volume & applies FFmpeg time-stretching
│   ├── eval/                     # Evaluation & Metrics System
│   │   ├── metrics.py            # Calculates Speaker Similarity & BLEU scores
│   │   ├── ablation.py           # Evaluates Emotion Preservation with/without conditioning
│   │   ├── benchmark.py          # Automated benchmark suite runner
│   │   └── human_eval_form.html  # Interactive human evaluation survey
│   ├── pipeline.py               # Master Pipeline Orchestrator (Chains Stage 1-6)
│   ├── exceptions.py             # Custom error handling classes (e.g. PipelineError)
│   └── logger.py                 # Structured logging utility with colored output
├── Wav2Lip/                      # Submodule for Wav2Lip Lip-Sync Neural Network
│   ├── inference.py              # Wav2Lip deep learning inference script
│   ├── models/                   # Neural network architectures (Wav2Lip GAN)
│   └── face_detection/           # Face detection & bounding box tracker (S3FD)
├── docs/                         # Architecture Documentation & Research Reports
│   ├── ARCHITECTURE.md           # Deep-dive system architecture breakdown
│   ├── PRIVACY.md                # Zero data-retention privacy policy
│   ├── eval_report.md            # Experimental benchmark results report
│   └── adrs/                     # Architecture Decision Records (ADRs 000-005)
├── tests/                        # Pytest Automated Test Suite (46 Unit Tests)
├── setup.py                      # Package installation script
├── requirements.txt              # Complete Python library dependency list
└── run_checks.ps1                # System diagnostic & health check script for Windows
```

---

### Detailed File Roles:

#### 1. Core Package (`resonova/`)
- **`resonova/pipeline.py`**: The "brain" of the project. Connects all 6 stages into a single function `run_dubbing_pipeline()`. Manages temporary directories, checkpoints, and progress reporting.
- **`resonova/logger.py`**: Formats console logs with timestamps and clear severity levels (`INFO`, `WARNING`, `ERROR`).
- **`resonova/exceptions.py`**: Defines custom exception classes like `ASRError`, `TranslationError`, and `LipSyncError` to prevent app crashes when an error occurs.

#### 2. User Interface (`resonova/app/`)
- **`resonova/app/app.py`**: Builds the Gradio web UI. Contains video upload boxes, target language dropdown, mock mode toggle, output tabs, and status cards.
- **`resonova/app/launch.py`**: Command-line entry point to launch the web server locally (`python -m resonova.app.launch`).
- **`resonova/app/report_card.py`**: Uses PIL (Python Imaging Library) to generate a graphic report showing emotion preservation percentage, speaker similarity score, and pitch/energy graphs.
- **`resonova/app/static/resonova.css`**: CSS stylesheet implementing the Dark Espresso aesthetic theme.

#### 3. AI Modules (`resonova/asr/`, `translation/`, `voice_cloning/`, `lipsync/`, `prosody/`)
- **`transcribe.py`**: Loads Whisper model, extracts audio from MP4, and outputs transcribed text.
- **`translate.py`**: Loads IndicTrans2 tokenizer and model to convert English text into fluent Hindi (`hin_Deva`). Falls back to Helsinki-NLP if IndicTrans2 is missing.
- **`clone_voice.py`**: Loads Coqui XTTS-v2, extracts speaker embedding from input `.wav`, and generates synthesized Hindi audio.
- **`lipsync.py`**: Interfaces with the `Wav2Lip` neural network to generate lip-synced video output.
- **`extract.py` & `conditioning.py`**: Extracts pitch (F0) and root-mean-square (RMS) energy, scaling cloned voice energy so quiet/loud emotions sound natural.

#### 4. Evaluation Suite (`resonova/eval/`)
- **`metrics.py`**: Computes BLEU and chrF translation scores, plus Resemblyzer speaker cosine similarity.
- **`ablation.py`**: Compares emotion preservation accuracy when prosody conditioning is turned ON vs OFF (demonstrating +40pp improvement).
- **`benchmark.py`**: Runs benchmark tests across test datasets.

#### 5. Configuration & Tests
- **`requirements.txt`**: Lists required libraries (`torch`, `gradio`, `transformers`, `librosa`, `opencv-python`, etc.).
- **`tests/`**: Contains 46 automated unit tests validating pipeline components, error handlers, prosody conditioning, and UI outputs.
- **`run_checks.ps1`**: PowerShell script to inspect system environment, Python version, FFmpeg installation, and CUDA GPU status.

---

### 📊 Summary Metrics to Remember for Faculty Presentations:
- **Speaker Similarity**: `86.50%` (Resemblyzer embedding cosine similarity)
- **Emotion Preservation**: `80.00%` (RAVDESS 20-clip evaluation)
- **Translation BLEU**: `0.5120` (FLORES-200, outperforms Helsinki-NLP baseline 0.4930)
- **Ablation SER Gain**: `+40pp` (Speech Emotion Recognition improves from 40% → 80% when conditioning is ON)
- **Unit Test Coverage**: `46/46 Passed (100%)`
