"""
vaani.app.launch
================
Entry point for launching the Vaani Gradio app locally or via Docker.

Usage:
    python -m vaani.app.launch           # local run
    docker compose up                     # Docker run (same entrypoint)

The app is available at http://localhost:7860 after startup.

HuggingFace Spaces deployment:
    For HF Spaces, Gradio picks up `demo` from app.py directly —
    this launch.py is only used for local/Docker runs.
"""

import logging
import os
import sys

from vaani.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Launch the Gradio application."""
    logger.info("=" * 60)
    logger.info("Vaani — Starting Gradio application")
    logger.info("=" * 60)

    # ── Validate environment ───────────────────────────────────────────────
    wav2lip_repo = os.environ.get("WAV2LIP_REPO_PATH", "")
    wav2lip_ckpt = os.environ.get("WAV2LIP_CHECKPOINT_PATH", "")

    if not wav2lip_repo or not os.path.isdir(wav2lip_repo):
        logger.warning(
            "WAV2LIP_REPO_PATH not set or not found: '%s'. "
            "Lip-sync will fail until Wav2Lip is installed. "
            "See README.md for setup instructions.",
            wav2lip_repo,
        )
    else:
        logger.info("Wav2Lip repo found: %s", wav2lip_repo)

    if not wav2lip_ckpt or not os.path.isfile(wav2lip_ckpt):
        logger.warning(
            "WAV2LIP_CHECKPOINT_PATH not found: '%s'. "
            "Download wav2lip_gan.pth and set the env var. "
            "See README.md for download instructions.",
            wav2lip_ckpt,
        )
    else:
        logger.info("Wav2Lip checkpoint found: %s", wav2lip_ckpt)

    # ── GPU status log ────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("GPU detected: %s (%.1f GB VRAM)", gpu_name, total_vram)
        else:
            logger.warning(
                "No CUDA GPU detected. Running in CPU mode. "
                "Inference will be very slow (~15-30 min per 30s clip). "
                "See ADR-000 for GPU requirements."
            )
    except ImportError:
        logger.warning("PyTorch not installed — cannot check GPU status.")

    # ── Launch Gradio ─────────────────────────────────────────────────────
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    logger.info("Starting Gradio server on %s:%d", server_name, server_port)
    logger.info("Open http://localhost:%d in your browser", server_port)

    from vaani.app.app import create_app
    demo = create_app()

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        quiet=False,  # Show Gradio startup logs
    )


if __name__ == "__main__":
    main()
