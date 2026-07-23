# 📝 Resonova — Final Internship Self-Evaluation Form

Student: **Sakshi Verma**  
Roll / ID: **3rd Year B.Tech CSE (AI & Data Engineering), LPU**  
Track / Segment: **Applied AI & Intelligent Systems / LLM Systems & Applied GenAI**  
Submission Date: **26 July 2026**

---

## Self-Evaluation Form Responses (10 Questions)

### Q1. Segment chosen, problem chosen, and why?
**Response**:
I chose the **Applied AI & Intelligent Systems** track and selected problem code `APPLIED-AI-DUBBING-01` (Emotion-Preserving AI Video Dubbing). I chose this problem because multi-modal AI systems combining audio, vision, and NLP represent the cutting edge of GenAI applications. Existing video dubbing platforms like HeyGen or Dubverse either charge high subscription fees or erase the speaker's emotional nuance. I wanted to engineer an open-weight solution that delivers high emotional fidelity at zero hardware cost.

---

### Q2. What are you most proud of in this project?
**Response**:
I am most proud of my **Prosody Preservation Conditioning Layer** and the rigorous scientific validation behind it. Designing RMS energy envelope matching and PYIN pitch contour tracking allowed me to achieve an **80.00% Speech Emotion Recognition (SER) agreement rate**. Furthermore, running a formal ablation study proved a net gain of **+40.00 percentage points** over unconditioned baselines—proving that my engineering design specifically solved the emotion erasure problem.

---

### Q3. What would you redo if you started over?
**Response**:
If I started over, I would design the **lip-sync subprocess interface** using gRPC or IPC sockets from Day 1 rather than standard file system I/O handoffs. While file passing worked reliably, streaming frame buffers directly via memory sockets would reduce pipeline disk I/O latency by ~15%.

---

### Q4. What role are you now best positioned for?
**Response**:
I am best positioned for roles such as **AI/Machine Learning Engineer**, **Applied GenAI Engineer**, or **Audio/Vision ML Systems Engineer**. This project demonstrated my competence across PyTorch model orchestration, MLOps memory management, signal processing, and production-grade software testing.

---

### Q5. Rate your comfort level (1–5) on the following technical domains:
- **SQL**: `4 / 5`
- **Python**: `5 / 5`
- **Cloud Infrastructure (Docker, HF Spaces, Colab)**: `4.5 / 5`
- **Docker**: `4.5 / 5`
- **Core Tech of Segment (Audio/Vision ML, PyTorch, Transformers, FFmpeg)**: `5 / 5`
- **Technical Communication & Writing**: `5 / 5`

---

### Q6. Which company would you interview at tomorrow with this portfolio?
**Response**:
I would confidently interview at companies building generative media or multimodal AI platforms, such as **Descript, ElevenLabs, RunwayML, Hugging Face, Meta AI (Fair), or InVideo**.

---

### Q7. What is your next 90 days of self-development plan?
**Response**:
- **Days 1–30**: Extend Resonova by integrating PyAnnote.audio for multi-speaker diarization and quantizing Whisper/Wav2Lip to ONNX FP16 formats.
- **Days 31–60**: Build a second portfolio project focused on real-time streaming LLM agents with RAG and vector database retrieval (Qdrant/Milvus).
- **Days 61–90**: Participate in open-source AI contributions and prepare for technical machine learning system design interviews.

---

### Q8. (Optional) Anything you want the internship lead to know?
**Response**:
Thank you for designing an internship structure that prioritizes engineering hygiene, formal ADR documentation, thorough testing, and production deployment over quick hacky scripts. This experience gave me genuine confidence in building defensible, production-grade AI systems.
