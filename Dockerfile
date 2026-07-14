# syntax=docker/dockerfile:1
# ============================================================
# Resonova — Dockerfile
# ============================================================
# Base: CUDA 11.3 + cuDNN 8 runtime (matches pinned torch==1.12.1+cu113)
# Python: 3.9 (pinned — Wav2Lip requires <3.10 for some deps)
# Exposed: port 7860 (Gradio default)
#
# ⚠️  GPU REQUIREMENT:
#   This image requires an NVIDIA GPU with CUDA support to run
#   any model inference (Whisper, IndicTrans2, XTTS-v2, Wav2Lip).
#   On machines WITHOUT a CUDA GPU, the Gradio UI will load but
#   model inference will fail with a CUDA error.
#   This is expected — see ADR-000 for the full deployment strategy.
#
# BUILD:
#   docker build -t resonova:latest .
#
# RUN (with GPU, recommended):
#   docker compose up
#
# RUN (CPU only — UI will load, inference will be very slow):
#   docker run -p 7860:7860 resonova:latest
# ============================================================

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ─── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update -qq && apt-get install -y -q \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-distutils \
    ffmpeg \
    cmake \
    build-essential \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.9 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && python -m pip install --upgrade pip setuptools wheel

# Verify ffmpeg
RUN ffmpeg -version | head -1

# ─── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Install PyTorch FIRST (before requirements.txt) ─────────────────────────
# Installing PyTorch first prevents other packages from pulling in a newer
# version that breaks Wav2Lip's face_alignment dependency.
RUN pip install --no-cache-dir \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    torchaudio==0.12.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# ─── Copy and install requirements ───────────────────────────────────────────
COPY requirements.txt .

# Install requirements (torch already installed above, pip will skip it)
RUN pip install --no-cache-dir -r requirements.txt

# Protect numpy version (some packages silently upgrade it)
RUN pip install --no-cache-dir numpy==1.23.5

# ─── Install IndicTrans2 from source ─────────────────────────────────────────
# IndicTrans2 is not on PyPI — must clone and install from repo
RUN git clone --depth 1 https://github.com/AI4Bharat/IndicTrans2.git /app/IndicTrans2 \
    && pip install --no-cache-dir -e /app/IndicTrans2/ \
    && pip install --no-cache-dir \
        indic-nlp-library==0.91 \
        mosestokenizer==1.2.1

# ─── Clone Wav2Lip ───────────────────────────────────────────────────────────
# Wav2Lip is run via subprocess from its cloned repo (not pip-installable).
# The checkpoint must be downloaded separately (see docker-compose.yml notes).
RUN git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git /app/Wav2Lip

# face-alignment at exact pinned version (Wav2Lip requirement)
RUN pip install --no-cache-dir face-alignment==1.3.5

# ─── Copy Resonova project code ──────────────────────────────────────────────────
COPY . .

# Install Resonova itself in editable mode
RUN pip install --no-cache-dir -e .

# ─── Environment variables ────────────────────────────────────────────────────
ENV WAV2LIP_REPO_PATH=/app/Wav2Lip
ENV WAV2LIP_CHECKPOINT_PATH=/app/Wav2Lip/checkpoints/wav2lip_gan.pth
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
# Allow HuggingFace Hub downloads inside the container
ENV HF_HOME=/app/.cache/huggingface

# ─── Expose Gradio port ───────────────────────────────────────────────────────
EXPOSE 7860

# ─── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# ─── Entrypoint ───────────────────────────────────────────────────────────────
# Note: Wav2Lip checkpoint is NOT bundled in the image (400 MB).
# Download it once and mount it via docker-compose volumes.
CMD ["python", "-m", "resonova.app.launch"]
