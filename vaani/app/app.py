"""
vaani.app.app
=============
Gradio application — Phase 0 stub.

This module defines the Gradio UI and wires the pipeline functions.
In Phase 0, the pipeline function is a stub that returns a placeholder message.
In Phase 4, the stub is replaced with the real `dub_video()` pipeline call.

HuggingFace Spaces ZeroGPU integration:
  The `@spaces.GPU` decorator is applied to the inference function so that
  on HuggingFace Spaces, each request receives a temporary GPU allocation
  via the ZeroGPU shared-GPU pool.

  Locally (Docker or plain Python), `spaces` is imported with a no-op
  fallback, so the decorator does nothing and the function runs normally.

  See ADR-000 for the full deployment strategy.
"""

import logging
import os
import time

import gradio as gr

from vaani.logger import get_logger

logger = get_logger(__name__)

# ─── ZeroGPU / spaces import (graceful fallback) ─────────────────────────────
# On HuggingFace Spaces with ZeroGPU enabled, `import spaces` activates
# the @spaces.GPU decorator, allocating a shared GPU burst per request.
# Locally (Docker, plain Python), spaces is not installed — we provide
# a no-op decorator so the code runs identically without modification.

try:
    import spaces  # HuggingFace spaces package (ZeroGPU)
    _ZEROGPU_AVAILABLE = True
    logger.info("HuggingFace `spaces` package loaded — ZeroGPU mode active")
except ImportError:
    # Local/Docker fallback: create a no-op @spaces.GPU decorator
    class _SpacesFallback:
        @staticmethod
        def GPU(fn=None, duration=None):  # noqa: N802
            """No-op decorator when running outside HuggingFace Spaces."""
            if fn is None:
                # Called as @spaces.GPU(duration=60)
                def decorator(f):
                    return f
                return decorator
            return fn

    spaces = _SpacesFallback()
    _ZEROGPU_AVAILABLE = False
    logger.info(
        "HuggingFace `spaces` package not found — running in local mode "
        "(no ZeroGPU, models will use local GPU or CPU)"
    )


# ─── Inference function ───────────────────────────────────────────────────────

@spaces.GPU(duration=300)  # 300 seconds = 5 min max per request on ZeroGPU
def run_dubbing_pipeline(video_file, target_language: str) -> tuple[str, str]:
    """
    Main inference function — entry point for the Gradio UI.

    In Phase 0: returns a placeholder message confirming the stub works.
    In Phase 4: replaced with the real dub_video() pipeline call.

    Args:
        video_file: Gradio file upload object (path to uploaded video).
        target_language: Target language code (currently only "Hindi").

    Returns:
        Tuple of (output_video_path_or_None, status_message).
    """
    logger.info(
        "Pipeline invoked",
        extra={"target_language": target_language, "zerogpu": _ZEROGPU_AVAILABLE},
    )

    # ── PHASE 0 STUB ──────────────────────────────────────────────────────
    # This is a placeholder. In Phase 4, replace this block with:
    #
    #   from vaani.pipeline import dub_video
    #   output_path = dub_video(video_file, target_lang="hin_Deva")
    #   return output_path, "✅ Dubbing complete!"
    #
    # ─────────────────────────────────────────────────────────────────────

    if video_file is None:
        logger.warning("No video file provided")
        return None, "❌ No video uploaded. Please upload a video file."

    try:
        from vaani.pipeline import dub_video  # noqa: PLC0415
        
        # Map target dropdown selection to standard internal BCP-47 codes
        lang_map = {
            "Hindi": "hin_Deva"
        }
        internal_lang = lang_map.get(target_language, "hin_Deva")

        logger.info("[App UI] Starting dubbing pipeline for video='%s'", os.path.basename(video_file))
        t_start = time.perf_counter()

        # Run pipeline
        output_video_path = dub_video(
            video_path=video_file,
            target_lang=internal_lang,
        )

        elapsed = time.perf_counter() - t_start
        status_msg = (
            f"✅ **Dubbing Complete!**\n\n"
            f"Processed: `{os.path.basename(video_file)}` → `{os.path.basename(output_video_path)}`\n"
            f"Total duration: **{elapsed:.1f}s**"
        )
        logger.info("[App UI] Pipeline completed successfully in %.2fs", elapsed)
        return output_video_path, status_msg

    except Exception as exc:
        err_msg = f"❌ **Dubbing Failed:** {exc}"
        logger.error("[App UI] Pipeline execution failed: %s", exc, exc_info=True)
        return None, err_msg


# ─── Gradio UI ───────────────────────────────────────────────────────────────

def _build_interface() -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""

    with gr.Blocks(
        title="Vaani — Emotion-Preserving AI Dubbing",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        css="""
            .gradio-container { max-width: 860px; margin: auto; }
            .warning-banner {
                background: #fef3c7;
                border: 1px solid #d97706;
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 0.9em;
            }
            footer { display: none !important; }
        """,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🗣️ Vaani
            ### Emotion-Preserving AI Dubbing · English → Hindi · Same Speaker Voice
            """,
        )

        # ── Latency warning banner ─────────────────────────────────────────
        # Honest about inference time regardless of whether ZeroGPU or CPU
        if _ZEROGPU_AVAILABLE:
            latency_note = (
                "⏱️ **ZeroGPU mode:** This demo uses HuggingFace's shared GPU pool. "
                "Expect **1–3 minutes per clip** (includes GPU allocation wait + "
                "model inference). The queue indicator below shows your position."
            )
        else:
            latency_note = (
                "⏱️ **Local/Docker mode:** Running on GPU (if available) or CPU. "
                "On GPU: ~1–2 min per 30-second clip. "
                "On CPU: ~15–30 min per clip (not recommended for demo use)."
            )

        gr.HTML(f'<div class="warning-banner">{latency_note}</div>')

        # ── Main interface ─────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                video_input = gr.Video(
                    label="Upload source video (English, 30–90 sec, face visible)",
                    format="mp4",
                    elem_id="video_input",
                )
                target_lang = gr.Dropdown(
                    choices=["Hindi"],
                    value="Hindi",
                    label="Target language",
                    elem_id="target_language",
                )
                submit_btn = gr.Button(
                    "🚀 Dub Video",
                    variant="primary",
                    elem_id="submit_button",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Output")
                video_output = gr.Video(
                    label="Dubbed video (Hindi, cloned voice, re-synced lips)",
                    elem_id="video_output",
                )
                status_box = gr.Markdown(
                    value="_Output will appear here after processing._",
                    elem_id="status_output",
                )

        # ── Pipeline diagram (collapsible) ────────────────────────────────
        with gr.Accordion("▶ How it works", open=False):
            gr.Markdown(
                """
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
                │  Prosody Extraction     │  → pitch, energy, rate, emotion
                │  (librosa)              │
                └─────────────────────────┘
                        │
                        ▼
                ┌───────────────────────────────┐
                │  XTTS-v2 Voice Cloning        │  → Hindi in your voice
                │  + Prosody Conditioning       │    with emotion preserved
                └───────────────────────────────┘
                        │
                        ▼
                ┌──────────────┐
                │  Wav2Lip     │  → Final dubbed video
                └──────────────┘
                ```

                **What makes Vaani different:** Most dubbing tools produce flat, robotic
                translations. Vaani's prosody layer extracts and preserves the original
                speaker's emotional tone — if you spoke with excitement, the Hindi
                output sounds excited too.
                """
            )

        # ── Known limitations (collapsible) ───────────────────────────────
        with gr.Accordion("⚠️ Known limitations", open=False):
            gr.Markdown(
                """
                1. **Wav2Lip artifacts on fast movement** — best results on frontal, relatively still video
                2. **Emotion preservation is an approximation** — heuristic pitch/rate conditioning, not validated research-grade transfer
                3. **Hindi voice quality lower than English** — XTTS-v2 has less Hindi training data
                4. **Single speaker only** — multi-speaker clips are not supported
                5. **30–90 second clips only** — longer clips are not tested
                6. **CPU inference takes 15–30 min** — GPU is required for practical demo use
                """
            )

        # ── Footer ────────────────────────────────────────────────────────
        gr.Markdown(
            """
            ---
            Built on open-weight models: [Whisper](https://github.com/openai/whisper) ·
            [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) ·
            [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) ·
            [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) · No paid APIs.
            """
        )

        # ── Event handler ─────────────────────────────────────────────────
        submit_btn.click(
            fn=run_dubbing_pipeline,
            inputs=[video_input, target_lang],
            outputs=[video_output, status_box],
            api_name="dub",  # enables programmatic API calls
        )

    return demo


# ─── Public API ──────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """
    Create and return the configured Gradio Blocks app.

    Used by launch.py (local/Docker) and by app.py on HuggingFace Spaces
    (where Gradio picks up the `demo` global automatically).
    """
    return _build_interface()


# ─── HuggingFace Spaces entry point ──────────────────────────────────────────
# When deployed to HuggingFace Spaces, Gradio looks for a `demo` global
# in app.py and launches it automatically.
demo = create_app()

if __name__ == "__main__":
    # Direct execution: python -m vaani.app.app
    # (Normally you'd use launch.py instead)
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
