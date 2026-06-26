# Mimi 🗣️ → 🌐
### Emotion-Preserving AI Dubbing & Voice-Cloned Translation Pipeline

> **Before you read anything else — watch the demo.**  
> *[Insert before/after clip here — record this as your very first Loom section]*

> ⚠️ **Compute note:** This project requires a CUDA GPU. No local GPU was available during
> development — all model training and inference was done via free-tier **Google Colab T4**
> and **Kaggle T4/P100** notebooks (see `/notebooks`). See [ADR-000](docs/adrs/ADR-000-compute-and-deployment-strategy.md)
> for the full compute and deployment strategy.

---

## What Mimi Does

Mimi takes a video of a person speaking in English and produces a dubbed version in Hindi, where:

1. **The same speaker's voice** is used — not a generic TTS robot voice
2. **Lip movements are re-synced** to match the new audio
3. **Emotional tone is preserved** — if you spoke the original with excitement, the Hindi output sounds excited too

Most open-source dubbing tools produce flat, robotic-sounding translations. Mimi's differentiator is a prosody-preservation layer that extracts pitch, energy, speaking rate, and emotional tone from the original delivery and conditions the translated speech to match.

---

## Pipeline Overview

```
Source Video (English)
        │
        ▼
┌───────────────┐
│  Whisper ASR  │  → English transcript
└───────────────┘
        │
        ▼
┌───────────────────┐
│  IndicTrans2      │  → Hindi translation
│  (AI4Bharat)      │
└───────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Prosody Extraction     │  → F0 contour, energy, rate, emotion label
│  (librosa + SER model)  │
└─────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  XTTS-v2 Voice Cloning        │  → Hindi speech in original speaker's voice
│  + Prosody Conditioning       │    with emotion preserved
└───────────────────────────────┘
        │
        ▼
┌──────────────┐
│  Wav2Lip     │  → Final dubbed video with re-synced lip movements
└──────────────┘
        │
        ▼
Dubbed Video (Hindi, cloned voice, preserved emotion)
```

---

## Quick Start

### Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | T4 (free Colab/Kaggle) | A100 / V100 |
| VRAM | 6 GB | 16 GB |
| Python | 3.9 | 3.9 |
| Disk | 10 GB (models) | 20 GB |

> ⚠️ **CPU-only runtime:** The pipeline will run but will be extremely slow (~15–30 min per 30-second clip). Expected latency is clearly stated — do not use CPU for real-time or demo purposes.

### 1. Clone and set up

```bash
git clone https://github.com/SAK_SHI14/mimi.git
cd mimi

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate mimi

# OR use pip directly (ensure Python 3.9)
pip install -r requirements.txt
```

### 2. Install IndicTrans2 (required — not pip-installable)

```bash
git clone https://github.com/AI4Bharat/IndicTrans2
pip install -e IndicTrans2/
```

### 3. Install Wav2Lip (required — not pip-installable)

```bash
git clone https://github.com/Rudrabha/Wav2Lip

# Download pretrained checkpoint:
# wav2lip_gan.pth from https://github.com/Rudrabha/Wav2Lip/releases
# Place it at: ./Wav2Lip/checkpoints/wav2lip_gan.pth

# Install Wav2Lip's specific dependencies (use pinned versions from requirements.txt!)
pip install face-alignment==1.3.5 librosa==0.8.1 numpy==1.23.5 opencv-python==4.5.5.64
```

> ⚠️ **Wav2Lip is the most dependency-sensitive component.** If you upgrade numpy, librosa, or face-alignment past the pinned versions, Wav2Lip will break. See `notes.md` for the full troubleshooting log.

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and set:
#   WAV2LIP_REPO_PATH=/absolute/path/to/Wav2Lip
#   WAV2LIP_CHECKPOINT_PATH=/absolute/path/to/wav2lip_gan.pth
```

### 5. Run the pipeline

```bash
# Full pipeline: English video → Hindi dubbed video
python -m mimi.pipeline dub \
  --input samples/my_clip.mp4 \
  --target-lang hin_Deva \
  --output outputs/dubbed_hindi.mp4
```

### 6. Run the Gradio app

```bash
python -m mimi.app.launch
# Open http://localhost:7860
```

---

## Running via Docker (Tier-1 Deployment)

Docker is the **primary deployment artifact** — fully reproducible on any machine with an NVIDIA GPU.

### Prerequisites
- Docker + Docker Compose installed
- NVIDIA GPU with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Wav2Lip checkpoint downloaded (see below)

### Steps

```bash
# 1. Clone and enter the repo
git clone https://github.com/SAK_SHI14/mimi.git
cd mimi

# 2. Download Wav2Lip checkpoint (~400 MB)
mkdir -p wav2lip_checkpoints
# Download wav2lip_gan.pth from: https://github.com/Rudrabha/Wav2Lip/releases
# Place it at: ./wav2lip_checkpoints/wav2lip_gan.pth
# Then uncomment the volume mount in docker-compose.yml

# 3. Build and run
docker compose up --build
# Open http://localhost:7860
```

> **No local GPU?** The container will start and the Gradio UI will load, but model
> inference will fail with a CUDA error. This is expected — use the Colab/Kaggle path
> for actual inference if you don't have a CUDA GPU locally.

---

## HuggingFace Spaces (Tier-2 Public Demo)

A public demo is deployed at:  
🔗 **[mimi — HuggingFace Space](https://huggingface.co/spaces/SAK_SHI14/mimi)**
*(Replace with your actual Space URL)*

### ZeroGPU mode
The Space uses HuggingFace's ZeroGPU shared-GPU pool via the `@spaces.GPU` decorator
pattern. Each request receives a temporary GPU allocation (~5 min max).

**Expected latency:** 1–3 minutes per 30-second clip (includes GPU queue wait + inference).

### CPU fallback
If ZeroGPU is unavailable or quota is exhausted, the app falls back to CPU inference.
The UI displays an explicit warning: *"Running on CPU — expect 15–30 minutes per clip."*
This is not a bug — it's physics. GPU is required for practical inference speed.

### Deploy your own Space
```bash
# Install HuggingFace Hub CLI
pip install huggingface_hub

# Login and push
huggingface-cli login
huggingface-cli repo create mimi --type space --space_sdk gradio
git remote add space https://huggingface.co/spaces/SAK_SHI14/mimi
git push space main
```

---

## Running on Google Colab / Kaggle

Since development has no local GPU, **Colab and Kaggle are the primary development and
testing environments** — not just optional alternatives.

### Session startup (every session)
1. Open `notebooks/mimi_colab_template.ipynb` in Colab (or `mimi_kaggle_template.ipynb` on Kaggle)
2. Run all cells — takes ~5 min to restore the environment
3. Then open `notebooks/phase1_setup.ipynb` for model installation and testing

**Expected model download time (first run, T4):**
| Model | Size | Download Time |
|---|---|---|
| Whisper medium | ~1.4 GB | ~2 min |
| IndicTrans2-1B | ~4 GB | ~5 min |
| XTTS-v2 | ~2 GB | ~3 min |
| Wav2Lip checkpoint | ~400 MB | ~1 min |

After first download, models are cached to Google Drive. Subsequent sessions start in ~60 seconds.

**Kaggle GPU budget:** 30 GPU-hr/week. See the Kaggle template notebook for per-phase usage estimates.

---

## Project Structure

```
mimi/
├── asr/                  # Whisper transcription wrapper
├── translation/          # IndicTrans2 translation wrapper
├── voice_cloning/        # XTTS-v2 voice cloning wrapper
├── lipsync/              # Wav2Lip lip-sync wrapper
├── prosody/              # Prosody extraction & conditioning (Phase 3)
├── eval/                 # Evaluation metrics suite (Phase 3)
├── app/
│   ├── app.py            # Gradio UI + @spaces.GPU stub (Phase 0→4)
│   └── launch.py         # Local/Docker launcher
├── tests/                # Integration tests
├── docs/
│   └── adrs/             # Architecture Decision Records
├── notebooks/
│   ├── mimi_colab_template.ipynb   # Session starter — Colab
│   ├── mimi_kaggle_template.ipynb  # Session starter — Kaggle
│   └── phase1_setup.ipynb           # Phase 1 model installation
├── samples/              # Your source video clips (not committed)
├── outputs/              # Pipeline outputs (not committed)
├── notes.md              # Dependency troubleshooting log
├── requirements.txt      # Pinned dependencies
├── environment.yml       # Conda environment
└── .env.example          # Environment variable template
```

---

## Running Tests

```bash
# Unit tests only (no GPU required — all models are mocked)
pytest -m "not integration" -v

# Individual module tests
pytest asr/tests/ -v
pytest translation/tests/ -v
pytest voice_cloning/tests/ -v
pytest lipsync/tests/ -v

# Integration tests (requires full GPU environment + model weights)
pytest -m integration -v
```

---

## Evaluation Metrics & Ablation Results

To run the automated ablation study comparison (comparing default Emotion Conditioning **ON** vs. a neutral voice baseline **OFF**), execute:
```bash
python -m mimi.eval.ablation --input samples/sample_clip.mp4 --gold "नमस्ते, मेरा नाम साक्षी है और मैं यहाँ हूँ।"
```
This generates a detailed comparison report in [docs/eval_report.md](docs/eval_report.md). 

### Expected Ablation Metric Profile

Based on our calibration runs, here are the expected score bounds:

| Metric | Description | Conditioning ON | Conditioning OFF | Rationale / Target |
|---|---|---|---|---|
| **Speaker Similarity** | Cosine similarity of Resemblyzer embeddings | **> 0.80** | ~ 0.50 | ON uses original speaker's clip; OFF uses a neutral voice reference |
| **Translation BLEU** | Sentence-level BLEU of Whisper transcript vs. Gold | **> 0.50** | > 0.50 | Measures legibility and translation accuracy (higher is better) |
| **Translation chrF** | Character F-score of Whisper transcript vs. Gold | **> 0.65** | > 0.65 | Standard metric for morphologically rich Indic languages |
| **Lip-sync (LSE-D)** | SyncNet Landmark Distance (Wav2Lip output) | **< 7.5** | < 7.5 | Wav2Lip baseline lip sync error (lower distance is better) |
| **Emotion Agreement** | Pitch/Energy Pearson contour correlation | **> 0.70** | ~ 0.35 | Pearson correlation coefficient mapped to [0.0, 1.0]; higher indicates better emotional alignment |

---

## Known Limitations

These are named limitations — not hidden. Naming them is how this project is credible.

1. **Wav2Lip artifacts on fast movement**: Wav2Lip produces visible artifacts on fast head movement, profile angles, or low-resolution source video. Best results on near-frontal, relatively still video. This is a fundamental limitation of the model, not a bug.

2. **Emotion preservation is an applied engineering approximation**: The prosody-conditioning layer adjusts pitch, rate, and energy in the cloned voice to match the original — it is NOT a validated research-grade emotion transfer system. It is an engineering heuristic that measurably improves naturalness (see ablation in `eval_report.md`), but it cannot guarantee perfect emotional fidelity.

3. **Hindi voice quality lower than English**: XTTS-v2 was trained on significantly more English than Hindi data. Hindi synthesis sounds like the speaker but with less natural prosody than the English baseline.

4. **Single-speaker only**: Multi-speaker clips are not supported. The pipeline assumes one speaker throughout the clip.

5. **Clip length limit (30–90 seconds)**: Longer clips are not supported and not tested. Wav2Lip's frame buffering and TTS inference both have practical limits on free-tier GPU.

6. **CPU inference latency**: On CPU-only (e.g., HuggingFace Spaces free tier), expect 15–30 minutes per 30-second clip. This is not a bug — it's physics. GPU is required for practical use.

---

## Architecture Decisions

| ADR | Decision | Rationale |
|---|---|---|
| [ADR-000](docs/adrs/ADR-000-compute-and-deployment-strategy.md) | No local GPU → Colab/Kaggle dev, Docker+HF Spaces deploy | Free-tier GPU pipeline, zero deployment cost |
| [ADR-001](docs/adrs/ADR-001-translation-model.md) | IndicTrans2 over NLLB-200 | SOTA on English→Hindi, Apache 2.0, Indic-specialized |
| [ADR-002](docs/adrs/ADR-002-voice-cloning-model.md) | XTTS-v2 over Bark/YourTTS | Only open-weight model with both Hindi support and true zero-shot cloning |
| [ADR-003](docs/adrs/ADR-003-audio-duration-sync-strategy.md) | Global time-stretching with chained `atempo` filters | Prevents drift by stretching/compressing audio to match video duration exactly |
| [ADR-004](docs/adrs/ADR-004-emotion-preservation.md) | Latent style conditioning using speaker's original segment | Preserves inflections and emotions without digital warping artifacts |
| [ADR-005](docs/adrs/ADR-005-evaluation-methodology.md) | BLEU, chrF, Resemblyzer similarity, Pearson contour correlation | Quantifiable local metrics for translation, speaker identity, and emotional alignment |

---

## Author

**Sakshi Verma** — Internship portfolio project, 2026  
Built entirely on open-weight models: Whisper, IndicTrans2, XTTS-v2, Wav2Lip.  
No paid APIs used anywhere in this pipeline.
