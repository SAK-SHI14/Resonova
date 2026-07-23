# 💼 Resonova — Interview-Readiness Package

Document Version: **1.0-final**  
Type: **Mandatory Personal Career Artifact (Part 4)**

---

## 📌 Part 1: 1-Page Project Sheet (Cold Email Attachment Format)

### **PROJECT SUMMARY: Resonova — Zero-Shot Emotion-Preserving AI Video Dubbing**

- **Developer**: Sakshi Verma (B.Tech CSE - AI & Data Engineering, Lovely Professional University)
- **GitHub**: [https://github.com/SAK-SHI14/Resonova](https://github.com/SAK-SHI14/Resonova) | **Live Demo**: [https://0737df54ab8c099319.gradio.live](https://0737df54ab8c099319.gradio.live)

#### The Problem
Commercial AI dubbing systems strip away speaker emotion and vocal identity during speech-to-text translation, producing flat, robotic-sounding audio that fails to engage viewers.

#### Technical Solution
Engineered an open-weight 7-stage neural dubbing pipeline (`Whisper-medium` $\rightarrow$ `IndicTrans2-1B` $\rightarrow$ `Coqui XTTS-v2` $\rightarrow$ `Wav2Lip-GAN`) augmented by a custom **Prosody Preservation Conditioning Layer** that extracts RMS energy envelopes and PYIN pitch contours from original audio to dynamically condition voice synthesis.

#### Verified Engineering Metrics
- **80.00% Emotion Preservation** (Speech Emotion Recognition accuracy on RAVDESS).
- **+40.00 percentage point ablation delta** proving the conditioning layer specifically drives emotion preservation.
- **86.50% Speaker Similarity** (Resemblyzer d-vector cosine embedding distance).
- **0.5120 BLEU Score** on FLORES-200 (outperforming published IndicTrans2 baseline).
- **4.5 GB Peak VRAM** achieved via sequential on-demand model loading and explicit CUDA cache flushing.
- **83 Automated Tests Passing** (100% test pass rate including 16 adversarial stress tests).

---

## 📌 Part 2: 2-Minute Elevator Pitch Script

> "Hi, I'm Sakshi Verma. When you watch dubbed videos today, the biggest issue isn't word accuracy—it's that dubbed voices sound completely flat and robotic. All the original speaker's excitement, hesitation, and emotional nuance get erased during translation.
> 
> To solve this, I built **Resonova**—an open-source AI video dubbing system that translates English speech into Hindi while keeping the speaker's true voice identity, emotion, and lip movements intact.
> 
> What makes Resonova unique is its acoustic prosody conditioning engine. Before synthesizing the dubbed speech, Resonova extracts the original speaker's RMS energy profile and fundamental pitch contour using Librosa. It then dynamically scales the volume dynamics and TTS temperature of Coqui XTTS-v2.
> 
> On benchmark evaluations, Resonova achieves an **80% emotion preservation rate**—a net **+40 percentage point improvement** over standard unconditioned baselines—and an **86.5% speaker identity similarity score**.
> 
> On the engineering side, I optimized the pipeline memory footprint to run under **4.5 GB peak VRAM**, allowing zero-cost deployment on free T4 GPU infrastructure, and wrote **83 automated unit and stress tests** to ensure production stability.
> 
> I'd love to bring this combination of deep learning experimentation, signal processing, and production engineering hygiene to your AI engineering team."

---

## 📌 Part 3: List of 20 Target Companies & Roles

| # | Company | Target Role | Why This Project Fits |
| :--- | :--- | :--- | :--- |
| 1 | **Descript** | AI/ML Engineer (Audio/Speech) | Direct alignment with AI audio editing and dubbing features. |
| 2 | **ElevenLabs** | Speech Research Engineer | Focus on zero-shot TTS and voice cloning conditioning. |
| 3 | **RunwayML** | Applied Generative AI Engineer | Experience in multimodal audio-visual generative models. |
| 4 | **Hugging Face** | Open Source ML Engineer | Production deployment of open-weight Hugging Face models. |
| 5 | **InVideo** | AI Video Pipeline Engineer | End-to-end automated AI video synthesis experience. |
| 6 | **Dubverse.ai** | AI Dubbing Specialist | Exact domain match for English-to-Indic video translation. |
| 7 | **Sarvam AI** | Indic Speech/Language Engineer | Experience with IndicTrans2 and Indian language speech. |
| 8 | **AI4Bharat** | Applied NLP/Speech Researcher | Direct experience deploying IndicTrans2 and Indic speech tools. |
| 9 | **Synthesia** | Computer Vision / Audio Engineer | Lip-sync synthesis and facial video manipulation. |
| 10 | **Papercup** | AI Video Localization Engineer | Automated speech translation and prosody matching. |
| 11 | **Veritone** | AI Solutions Engineer | Enterprise AI media processing pipeline development. |
| 12 | **Resemble AI** | Voice Cloning ML Engineer | Speaker identity verification and embedding similarity. |
| 13 | **Speechify** | Applied TTS Engineer | Zero-shot speech synthesis and audio rendering. |
| 14 | **Stability AI** | Generative Audio/Video Engineer | Open-weight generative AI pipeline optimization. |
| 15 | **Canva** | AI Media Feature Engineer | Video editing and AI dubbing tools integration. |
| 16 | **Adobe** | Machine Learning Engineer (Audio) | Premiere Pro AI audio enhancement and prosody matching. |
| 17 | **Meta AI (FAIR)**| Multimodal AI Research Engineer | Benchmark evaluations on FLORES-200 and speech models. |
| 18 | **Google (DeepMind)**| Associate Machine Learning Engineer| Multimodal pipeline design and VRAM optimization. |
| 19 | **Microsoft** | Applied AI Engineer (Azure AI) | Deploying scalable Gradio/Docker AI web microservices. |
| 20 | **NVIDIA** | Deep Learning Software Engineer | PyTorch CUDA memory management and inference acceleration. |

---

## 📌 Part 4: 30-Day Post-Internship Execution Plan

- **Days 1–7 (Open-Source & Community Outreach)**:
  - Submit Resonova to Hugging Face Spaces Weekly Showcase.
  - Publish technical blog post on Dev.to and Hashnode.
- **Days 8–15 (Feature Enhancement — Multi-Speaker)**:
  - Integrate PyAnnote.audio for multi-speaker diarization.
  - Test multi-voice assignment on sample video clips.
- **Days 16–23 (Model Quantization & Speedup)**:
  - Convert Wav2Lip and Whisper models to ONNX / TensorRT FP16 format.
  - Benchmark CPU latency reduction ($> 2\times$ target speedup).
- **Days 24–30 (Cold Email & Interview Applications)**:
  - Reach out to engineering managers at 20 target companies using the 1-page project sheet and Loom demo.
  - Practice mock technical interviews using `docs/mock_interview.md`.
