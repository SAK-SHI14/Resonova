# 🗃️ Resonova — Data Sources & Schema Specification

Document Version: **1.0-final**  
Compliance: **Zero-Data Retention Privacy Policy**

---

## 1. Primary Datasets Used for Evaluation & Verification

Resonova relies on three public, standardized benchmark datasets for pipeline verification, quality measurement, and scientific ablation studies.

| Dataset Name | Source / Provider | License | Primary Purpose in Project | Data Volume / Sample Size |
| :--- | :--- | :--- | :--- | :--- |
| **RAVDESS** | Ryerson Audio-Visual Database | CC BY-NC-SA 4.0 | Speech Emotion Recognition (SER) & prosody validation | 20-clip stratified speech subset across 8 emotional states |
| **FLORES-200** | Meta AI Research | CC BY-SA 4.0 | Machine Translation BLEU & chrF accuracy benchmark | 100 English → Hindi sentence evaluation pairs (`eng_Latn` → `hin_Deva`) |
| **Resemblyzer Embeddings** | Resemblyzer / CorentinJ | MIT License | Speaker voice identity cosine similarity evaluation | 50 reference d-vector speaker embedding profiles |

---

## 2. Dataset Schemas & Pipeline Data Formats

### A. Intermediate Audio Extract Schema (`.wav`)
- **Format**: Uncompressed Linear PCM WAV
- **Sample Rate**: 16,000 Hz (16 kHz mono)
- **Bit Depth**: 16-bit PCM
- **Channels**: 1 (Mono)
- **Role**: Reference audio input for Whisper ASR, Librosa prosody profiling, and XTTS-v2 voice reference.

### B. Prosody Conditioning Data Schema (`dict`)
Extracted per audio frame by `resonova/prosody/extract.py`:
```json
{
  "f0_contour": [120.45, 122.10, 125.80, 0.00, 118.30],
  "rms_energy": [0.012, 0.045, 0.089, 0.002, 0.034],
  "speaking_rate": 3.42,
  "pause_ratio": 0.15,
  "duration_seconds": 12.45
}
```

### C. ASR & Translation Text Schema
```json
{
  "source_language": "en",
  "target_language": "hi",
  "original_transcript": "Welcome to Resonova, an emotion-preserving video dubbing system.",
  "translated_text": "रेसोनोवा में आपका स्वागत है, जो एक भावना-संरक्षित वीडियो डबिंग प्रणाली है।",
  "word_timestamps": [
    {"word": "Welcome", "start": 0.12, "end": 0.65},
    {"word": "to", "start": 0.66, "end": 0.80}
  ]
}
```

---

## 3. Data Privacy & Zero-Retention Policy

In compliance with [`docs/PRIVACY.md`](docs/PRIVACY.md):
- **No User Storage**: Uploaded user videos and synthesized audio outputs are processed entirely in transient session memory (`/tmp/resonova_*`) and permanently purged immediately after session termination.
- **No Model Training on User Data**: User clips are never logged, cached, or submitted to remote servers for model fine-tuning.
- **Local / Self-Hosted Execution**: Pipeline runs fully locally or on isolated Hugging Face Spaces containers with zero third-party telemetry.
