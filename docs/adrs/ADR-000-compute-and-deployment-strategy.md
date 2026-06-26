# ADR-000: Compute and Deployment Strategy
**Status:** Accepted  
**Date:** Phase 0 — Environment Sprint  
**Authors:** Sakshi Verma  
**Context:** Project Babel — emotion-preserving dubbing pipeline (English → Hindi)

---

## Context

Babel is a GPU-heavy inference pipeline. It requires:
- NVIDIA GPU with CUDA support for all model inference (Whisper, IndicTrans2, XTTS-v2, Wav2Lip)
- Persistent storage for model weights and intermediate outputs
- A reproducible, publicly demonstrable deployment for portfolio evaluation

**Hard constraint:** The development machine has **no NVIDIA GPU** (Intel Arc integrated graphics — does not support CUDA). This is the single most important constraint shaping every other decision in this ADR.

---

## Decision 1: Development Compute — Google Colab / Kaggle Free Tier

### Options Evaluated

| Option | Cost | GPU Access | Session Limit | Verdict |
|---|---|---|---|---|
| Local machine (Intel Arc) | ₹0 | ❌ No CUDA | N/A | ❌ Eliminated — no CUDA |
| Google Colab free tier | ₹0 | ✅ T4 (~15 GB VRAM) | ~12 hr + idle disconnect | ✅ **Primary** |
| Kaggle free tier | ₹0 | ✅ T4/P100 | 30 GPU-hr/week, 12 hr/session | ✅ **Backup** |
| Google Colab Pro | ~₹900/mo | ✅ T4/A100 | Higher limits | ❌ Paid — eliminated |
| AWS/GCP spot GPU | Variable | ✅ Any | On-demand | ❌ Paid — eliminated |
| Vast.ai / RunPod | ~₹2/hr | ✅ Various | On-demand | ❌ Paid — eliminated |

### Decision

**Primary compute: Google Colab T4 free tier.**  
**Backup compute: Kaggle T4/P100 free tier** (30 GPU-hr/week; useful for longer ablation runs in Phase 3).

### Consequences of Session Impermanence

Colab and Kaggle sessions are **temporary**:
- Sessions reset on disconnect; no data persists in `/content/` across sessions
- Free Colab: ~12 hour cap with idle disconnects after ~30–60 min of inactivity
- Kaggle: 12-hour session cap, 30 GPU-hours/week hard limit

**Mitigation strategy (mandatory, not optional):**
1. All code lives in **GitHub** — pull at session start, push at session end
2. All intermediate outputs (transcripts, cloned audio, dubbed video, eval results, fine-tuned weights if any) are checkpointed to **Google Drive** immediately after each pipeline stage
3. Every notebook cell that produces output includes a Drive checkpoint log statement
4. The session-start template notebook (`notebooks/babel_colab_template.ipynb`) restores the full environment in ~5 minutes

### GPU Memory Budget (T4 = ~15 GB VRAM)

| Model | Peak VRAM | Fits T4? |
|---|---|---|
| Whisper medium | ~1.5 GB | ✅ |
| Whisper large-v3 | ~3.0 GB | ✅ |
| IndicTrans2-1B | ~4.0 GB | ✅ |
| XTTS-v2 | ~4–5 GB | ✅ |
| Wav2Lip | ~2.0 GB | ✅ |
| **All simultaneously** | **~13–15 GB** | ⚠️ Marginal |

**Strategy: Never load all models simultaneously.** Load → run → `del model` → `torch.cuda.empty_cache()` between pipeline stages. This keeps peak usage at ~4–5 GB at any moment.

---

## Decision 2: Primary Deployment — Docker + Docker Compose

### Context

The pipeline (XTTS-v2, Wav2Lip) requires a GPU for usable inference speed (~30–60 sec per clip on T4). Free always-on GPU web hosting does not exist. The primary deployment artifact must be:
- Fully reproducible from a fresh clone
- Demonstrable on any machine with an NVIDIA GPU (recruiter's machine, lab machine, cloud VM)
- Not dependent on any paid service

### Decision

**Primary deployment: Dockerized Gradio app via `docker compose up`.**

The Docker image:
- Base: `nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04` (CUDA 11.3 matches pinned PyTorch)
- Python 3.9 + all pinned requirements
- Gradio app exposed on port 7860
- Docker Compose wires NVIDIA GPU passthrough via `deploy.resources.reservations.devices`

**Usage:**
```bash
docker compose up          # builds image + starts Gradio on localhost:7860
docker compose up --build  # force rebuild (use after requirements.txt changes)
```

**Known limitation on no-GPU machines:** `docker compose up` will start the container and the Gradio UI will load, but model inference will fail with a CUDA error. This is expected — the Docker artifact is a deployment artifact for GPU machines, not for development on the no-GPU laptop.

### Why Not Render / Railway / Fly.io

Free tiers on these platforms provide CPU-only containers. Running XTTS-v2 and Wav2Lip on CPU takes 15–30 minutes per 30-second clip — not usable for a demo. Adding GPU on these platforms costs money. Eliminated.

---

## Decision 3: Secondary Deployment — HuggingFace Spaces (ZeroGPU)

### Context

For a publicly accessible demo link (portfolio requirement), a secondary deployment is needed. It must cost ₹0 and must be publicly accessible without the viewer running Docker.

### Options Evaluated

| Option | Cost | GPU? | Verdict |
|---|---|---|---|
| HuggingFace Spaces (free CPU tier) | ₹0 | ❌ CPU only | ⚠️ Viable with caveats |
| HuggingFace Spaces ZeroGPU | ₹0 | ✅ Shared GPU bursts | ✅ **Primary attempt** |
| Streamlit Community Cloud | ₹0 | ❌ CPU only | ❌ Wrong framework (Gradio needed for ZeroGPU) |
| Google Colab + ngrok tunnel | ₹0 | ✅ T4 | ⚠️ Session-limited, not persistent |

### Decision

**Secondary deployment: HuggingFace Spaces, attempting ZeroGPU first, falling back to CPU tier.**

**ZeroGPU approach (preferred):**
- Uses `spaces` Python package and `@spaces.GPU` decorator
- HuggingFace allocates a shared GPU (A10G) per-request for the decorated function
- Free tier: limited queue depth, may have wait times during high usage
- Implementation: wrap the `dub_video()` pipeline function with `@spaces.GPU`

**CPU fallback (if ZeroGPU proves impractical):**
- Deploy same Gradio app without `@spaces.GPU`
- CPU inference takes 15–30 min per 30-sec clip
- UI shows an explicit, honest warning: *"This demo runs on CPU. Expect 15–30 minutes per clip. For fast results, run the Docker version locally."*
- This is documented as a feature (honest about constraints) not hidden

### ZeroGPU Known Risks

1. **Queue wait times**: ZeroGPU is shared across all Spaces users — during peak hours, requests may queue for minutes before getting GPU allocation
2. **Model load time**: On each ZeroGPU burst allocation, models must be loaded fresh (~60–90 sec for Wav2Lip + XTTS-v2). This is architectural — ZeroGPU does not persist model state between requests
3. **Memory limit**: ZeroGPU A10G provides ~24 GB VRAM — more than our T4 baseline, so OOM is not expected
4. **HuggingFace model licenses**: XTTS-v2 uses Coqui Public Model License (non-commercial) — acceptable for a portfolio Space

**If ZeroGPU is not available or too unreliable:** The Loom recording uses the Tier-1 Docker demo as the primary proof. The HuggingFace Space link is a "bonus" — its limitations are stated openly in the README and in the UI itself.

---

## Decision 4: Code Hosting — GitHub

All code lives in GitHub. Neither Colab, Kaggle, nor the Docker container is the authoritative source of code. Session workflow:

```
Session start: git pull (Colab/Kaggle)
Work in session: code changes in /content/babel
Session end: git push (code) + Drive checkpoint (outputs)
```

This means the GitHub repo is always in a runnable state — any reviewer can clone it and reproduce the environment using either the Docker path or the Colab notebook path.

---

## Summary of Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DEVELOPMENT (no local GPU)                                      │
│  ┌─────────────────┐   ┌─────────────────────────────────────┐  │
│  │  Local machine  │   │  Google Colab T4 (free)             │  │
│  │  (Intel Arc)    │   │  ↕ GitHub (code)                    │  │
│  │  write code     │   │  ↕ Google Drive (outputs)           │  │
│  │  view results   │   │  run all model inference             │  │
│  └─────────────────┘   └─────────────────────────────────────┘  │
│                              ↕ backup                            │
│                         ┌─────────────────────────────────────┐  │
│                         │  Kaggle T4/P100 (free, 30hr/week)   │  │
│                         │  Phase 3 ablation runs              │  │
│                         └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  DEPLOYMENT                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  TIER 1 (primary): Docker + Docker Compose               │   │
│  │  docker compose up → Gradio on localhost:7860            │   │
│  │  Requires: NVIDIA GPU on the running machine             │   │
│  │  100% reproducible from fresh clone                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  TIER 2 (secondary): HuggingFace Spaces                  │   │
│  │  ZeroGPU (@spaces.GPU) preferred                         │   │
│  │  CPU fallback with honest latency warning in UI          │   │
│  │  Public URL — anyone can try it                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Consequences

**Positive:**
- Zero cost end-to-end — no paid compute or hosting at any point
- Docker artifact is a real, self-contained, reproducible deployment demo
- HuggingFace Spaces gives a public URL for portfolio/recruiter access
- Two-tier strategy means the demo works even if one tier fails

**Negative / Accepted Tradeoffs:**
- No always-on, fast public demo (fundamental physics — GPU inference costs money)
- Colab/Kaggle session impermanence adds session management overhead (mitigated by checkpoint discipline)
- ZeroGPU queue waits are unpredictable — the Loom recording uses Docker as primary proof

---

## References
- HuggingFace ZeroGPU documentation: https://huggingface.co/docs/hub/spaces-zerogpu
- `spaces` package (Gradio + ZeroGPU): https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub/inference/_generated
- Docker NVIDIA GPU support: https://docs.docker.com/compose/gpu-support/
- Colab free tier limits: https://research.google.com/colaboratory/faq.html
- Kaggle GPU limits: https://www.kaggle.com/docs/notebooks#technical-specifications
