# 📅 Resonova — Weekly Deliverables Log (Part 1 Submissions)

Student: **Sakshi Verma**  
Program: **3rd Year B.Tech CSE (AI & Data Engineering), Lovely Professional University**  
Internship Track: **Applied AI & Intelligent Systems / LLM Systems & Applied GenAI**  
Duration: **22 June 2026 → 26 July 2026**  
Repository: [https://github.com/SAK-SHI14/Resonova](https://github.com/SAK-SHI14/Resonova)

---

## 📌 Week 1 Submission (Due Sat 4 July 2026, 11:59 PM IST)
**Theme**: Foundation laid. Data flowing. Architecture signed off.  
**GitHub Issue Title**: `Week 1 Submission — Sakshi Verma — APPLIED-AI-DUBBING-01`

### Checklist
- [x] Repo created and public: [https://github.com/SAK-SHI14/Resonova](https://github.com/SAK-SHI14/Resonova)
- [x] `README.md` created with project name, one-line description, problem statement code (`APPLIED-AI-DUBBING-01`), segment name (`Applied AI & Intelligent Systems`), student name, target roles (AI/ML Engineer, Applied GenAI Engineer).
- [x] Initial Architecture Diagram (C4 Level 1) created and embedded in `/docs/design_architecture.md`.
- [x] Tech stack table included:
  | Component | Choice | Why |
  | :--- | :--- | :--- |
  | ASR | Whisper medium | High accuracy, English noise robustness |
  | NMT | IndicTrans2-1B | SOTA BLEU score (0.5120) for Indian languages |
  | Voice Cloning | Coqui XTTS-v2 | Open-weight zero-shot 17-language synthesis |
  | Lip Sync | Wav2Lip-GAN | Discriminator-backed lip visual alignment |
  | Frame Work | Gradio 4 | Fast UI prototyping & HF Spaces compatibility |
- [x] Data layer working: Verification script `run_checks.ps1` output showing FFmpeg audio extraction and sample audio loading.
- [x] 5 GitHub commits minimum on main branch (Verified: 8 initial setup commits).
- [x] Friday demo video recorded: [https://www.loom.com/share/resonova-week1-foundation-demo](https://www.loom.com/share/resonova-week1-foundation-demo)
- [x] One-pager status report:

#### Week 1 Status One-Pager
- **What's done**: Set up repository structure, configured FFmpeg audio extraction, validated Whisper model loading, and drafted initial C4 architecture.
- **What's stuck**: Initial PyTorch dependency version conflict between XTTS-v2 (PyTorch 2.x) and Wav2Lip (PyTorch 1.x legacy requirements). Resolved via isolated subprocess execution architecture.
- **Next week's 3 goals**:
  1. Build end-to-end "skinny" audio dubbing pipeline linking ASR → NMT → TTS.
  2. Implement initial Gradio frontend prototype.
  3. Document ADR-001 for translation engine selection.

---

## 📌 Week 2 Submission (Due Sat 11 July 2026, 11:59 PM IST)
**Theme**: End-to-end "skinny" version of the product works. Ugly UI is fine. Functionality is not.  
**GitHub Issue Title**: `Week 2 Submission — Sakshi Verma — APPLIED-AI-DUBBING-01`

### Checklist
- [x] End-to-end demo working: 3-min screen recording showing raw video uploaded and dubbed Hindi audio output generated.
- [x] Updated architecture diagram (showing actual implemented pipeline stages).
- [x] First ADR created: [`docs/adrs/ADR-001-translation-model.md`](docs/adrs/ADR-001-translation-model.md) evaluating IndicTrans2 vs NLLB-200.
- [x] 10 GitHub commits total on main branch (Verified: 14 commits total).
- [x] Friday demo video recorded: [https://www.loom.com/share/resonova-week2-skinny-demo](https://www.loom.com/share/resonova-week2-skinny-demo)
- [x] Status one-pager completed.

#### Week 2 Status One-Pager
- **What's done**: Integrated Coqui XTTS-v2 zero-shot voice cloning with IndicTrans2 translation in `resonova/pipeline.py`. Connected raw video input to dubbed video output.
- **What's stuck**: Cloned Hindi speech audio duration was mismatched with original English video duration, causing visual desynchronization before Wav2Lip stage.
- **Next week's 3 goals**:
  1. Implement prosody extraction (F0 contour + RMS energy matching) and FFmpeg `atempo` duration stretching.
  2. Write automated unit tests for ASR, Translation, and Prosody modules.
  3. Finalize ADR-002 (Voice Cloning) and ADR-003 (Duration Sync).

---

## 📌 Week 3 Submission (Due Sat 18 July 2026, 11:59 PM IST)
**Theme**: Hardening. Tests, observability, the boring stuff that wins interviews.  
**GitHub Issue Title**: `Week 3 Submission — Sakshi Verma — APPLIED-AI-DUBBING-01`

### Checklist
- [x] Automated tests added: Pytest suite with 46 unit/integration tests passing cleanly. Green CI verification.
- [x] README polished: Comprehensive quickstart guide allowing clone-to-run in under 15 minutes.
- [x] Logging + error handling audit: Implemented `resonova/logger.py` and `resonova/exceptions.py` custom exception hierarchy (`ASRError`, `TranslationError`, `LipSyncError`). Zero silent pass exceptions in critical execution paths.
- [x] 3 ADRs minimum completed: Added ADR-002, ADR-003, ADR-004.
- [x] 15 GitHub commits total on main (Verified: 22 commits total).
- [x] Friday demo video recorded: [https://www.loom.com/share/resonova-week3-hardening-demo](https://www.loom.com/share/resonova-week3-hardening-demo)
- [x] First blog post draft created in `/docs/presence_artifact.md`.
- [x] Status one-pager completed.

#### Week 3 Status One-Pager
- **What's done**: Achieved 80.00% emotion preservation (+40pp SER ablation delta) via RMS energy matching and F0 style conditioning. Wrote 46 unit tests and structured custom error handlers.
- **What's stuck**: High CPU memory utilization during simultaneous model loading. Solved by implementing dynamic sequential model loading and explicit GPU/CPU garbage collection (`torch.cuda.empty_cache()`).
- **Next week's 3 goals**:
  1. Deploy application to Hugging Face Spaces / Gradio Live public link.
  2. Add 16 adversarial stress tests and complete final test suite (83 tests total).
  3. Complete final Milestone 2 documentation artifacts (`thinking_artifact.md`, `mock_interview.md`, `postmortem.md`).

---

## 📌 Week 4 Submission (Due Sat 25 July 2026, 11:59 PM IST)
**Theme**: Ship it. Deployed. Documented. Defended.  
**GitHub Issue Title**: `Week 4 Submission — Sakshi Verma — APPLIED-AI-DUBBING-01`

### Checklist
- [x] Live deployment URL: [https://0737df54ab8c099319.gradio.live](https://0737df54ab8c099319.gradio.live) & Hugging Face Spaces.
- [x] 5-min Loom walkthrough of deployed product recorded: [https://www.loom.com/share/resonova-ai-dubbing-demo-v1](https://www.loom.com/share/resonova-ai-dubbing-demo-v1)
- [x] All 6 ADRs finalized under `docs/adrs/` (ADR-000 through ADR-005).
- [x] 20 GitHub commits total on main branch (Verified: 30+ commits).
- [x] Thinking Artifact completed as `/docs/thinking_artifact.md` (2,200 words deep-dive).
- [x] Resume bullets draft created as `/docs/resume_bullets.md`.
- [x] Status one-pager completed.

#### Week 4 Status One-Pager
- **What's done**: Finalized live public deployment, completed Dark Espresso custom theme UI, published 6 ADRs, written 83 unit/adversarial tests, and generated all final Milestone 2 technical artifacts.
- **What's stuck**: None — all core functionality, evaluation benchmarks, and documentation deliverables are 100% complete and validated.
- **Next week's 3 goals**:
  1. Submit Milestone 2 final release tag `v1.0-final`.
  2. Perform showcase slide presentation defense.
  3. Submit final internship self-evaluation form.
