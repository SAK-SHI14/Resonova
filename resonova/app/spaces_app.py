"""
resonova/app/spaces_app.py — HuggingFace Spaces Entry Point
=========================================================
Uses ZeroGPU wrapper for requests to leverage free GPU bursts.
"""
import logging
import os
import sys

logger = logging.getLogger(__name__)

# ─── Programmatic Setup for HuggingFace Spaces ────────────────────────────────
# Clone and install sub-repositories if we are running in the Spaces environment.
if os.environ.get("SPACES_AUTHOR_NAME") or "spaces" in os.environ.get("PATH", "").lower() or os.path.exists("/home/user/app"):
    import subprocess
    import urllib.request

    # 1. Install IndicTransToolkit if missing
    try:
        from IndicTransToolkit import IndicProcessor
        logger.info("HF Space Startup: IndicTransToolkit already available.")
    except ImportError:
        logger.info("HF Space Startup: Installing IndicTransToolkit from PyPI...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "IndicTransToolkit"])

    # 2. Clone Wav2Lip if missing
    if not os.path.exists("Wav2Lip"):
        logger.info("HF Space Startup: Cloning Wav2Lip...")
        subprocess.run(["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git"])

        # Patch Wav2Lip for numpy 1.24+ compatibility (no np.bool, np.int etc.)
        logger.info("HF Space Startup: Patching Wav2Lip for numpy 1.24+...")
        replacements = [
            ('np.bool,', 'bool,'), ('np.bool)', 'bool)'), ('np.bool ', 'bool '),
            ('np.int,', 'int,'), ('np.int)', 'int)'), ('np.int ', 'int '),
            ('np.float,', 'float,'), ('np.float)', 'float)'), ('np.float ', 'float '),
            ('np.complex,', 'complex,'), ('np.complex)', 'complex)'), ('np.complex ', 'complex '),
            ('np.object,', 'object,'), ('np.object)', 'object)'), ('np.object ', 'object '),
            ('np.str,', 'str,'), ('np.str)', 'str)'), ('np.str ', 'str '),
        ]
        for root, _, files in os.walk("Wav2Lip"):
            for fname in files:
                if fname.endswith(".py"):
                    fpath = os.path.join(root, fname)
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    patched = content
                    for old, new in replacements:
                        patched = patched.replace(old, new)
                    if patched != content:
                        with open(fpath, "w", encoding="utf-8") as f:
                            f.write(patched)

    # 3. Download Wav2Lip GAN checkpoint if missing
    ckpt_dir = "Wav2Lip/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "wav2lip_gan.pth")
    if not os.path.exists(ckpt_path):
        logger.info("HF Space Startup: Downloading Wav2Lip checkpoint...")
        urllib.request.urlretrieve(
            "https://huggingface.co/yzd-v/Wav2Lip/resolve/main/wav2lip_gan.pth",
            ckpt_path
        )

    # 4. Set environmental variables and sys.path
    os.environ["WAV2LIP_REPO_PATH"] = os.path.abspath("Wav2Lip")
    os.environ["WAV2LIP_CHECKPOINT_PATH"] = os.path.abspath(ckpt_path)
    
    sys.path.append(os.path.abspath("Wav2Lip"))
    logger.info("HF Space Startup: Setup completed successfully.")

# ─── Gradio App Import and Setup ─────────────────────────────────────────────
import gradio as gr
import spaces
from resonova.app.app import build_interface, run_resonova_pipeline as _pipeline_fn

@spaces.GPU(duration=240)  # 4 minutes max per request
def run_pipeline_zerogpu(video_file, target_language, mock_mode=False, progress=gr.Progress()):
    """ZeroGPU wrapper — models load inside this decorated function."""
    return _pipeline_fn(video_file, target_language, mock_mode, progress)

# Build the same interface as app.py but wrapped for ZeroGPU
demo = build_interface(pipeline_fn=run_pipeline_zerogpu, for_spaces=True)

if __name__ == "__main__":
    _static = os.path.abspath("resonova/app/static")
    _legacy = os.path.abspath("resonova/app/background.png")
    _allowed = [p for p in [_static, _legacy] if os.path.exists(p)]
    demo.launch(allowed_paths=_allowed)
