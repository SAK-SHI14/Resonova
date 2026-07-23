# 📄 Resonova — Resume Bullets (Tailored for Sakshi Verma)

These bullet points integrate your exact resume layout with verified high-impact quantitative metrics:

---

### AI/ML Engineer Intern — Futurense Technologies (Resonova Project)

- **Built Resonova**, an emotion-preserving AI dubbing pipeline converting English speech videos into Hindi using Whisper-medium ASR, IndicTrans2-1B NMT, Coqui XTTS-v2 zero-shot voice cloning, and Wav2Lip-GAN lip-sync.

- **Engineered a prosody-preservation layer** (RMS energy envelope matching & PYIN pitch tracking via Librosa) to condition cloned speech, achieving **80.00% Speech Emotion Recognition (SER)** accuracy—a verified **+40.00 percentage point ablation improvement** over unconditioned TTS baselines.

- **Achieved an 86.50% speaker identity similarity score** (measured via Resemblyzer d-vector embeddings) and **0.5120 BLEU score** on FLORES-200, outperforming published translation baselines.

- **Optimized GPU VRAM lifecycle management** via sequential on-demand model loading and explicit CUDA cache flushing, constraining peak memory under **4.5 GB VRAM** for zero-cost deployment on Google Colab T4 and Hugging Face Spaces (ZeroGPU).

- **Containerized the pipeline (Docker) and architected a production-grade Python package** backed by **83 automated Pytest test cases** (100% pass rate, 16 adversarial stress tests) and a custom Gradio 4 Dark Espresso UI.
