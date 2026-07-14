# RESONOVA — Emotion-Preserving AI Dubbing & Voice-Cloned Translation
### Custom Project Proposal | Segment 3: Applied AI & Intelligent Systems

---

## 1. Business Scenario

You are the **Applied AI Engineer at "Polyglot Studios"**, a media-localization startup serving Indian content creators, ed-tech platforms, and regional businesses who want their videos (training content, lectures, product demos, social content) available in multiple Indian languages — without re-shooting, without hiring voice actors, and without losing the original speaker's identity and emotional tone. Today, dubbing is either: (a) expensive and slow (human voice actors, studio time, manual lip-sync), or (b) cheap but robotic — flat AI voices that don't match lip movement and strip out all emotion, making content feel fake and untrustworthy. Your job: build a system that dubs a video into another language, in the *original speaker's own cloned voice*, with lips that actually match, and — critically — without flattening the emotional tone of the original delivery.

---

## 2. Problem Statement

Build **Resonova** — an emotion-preserving AI dubbing pipeline:

- **Input:** a video of a person speaking (self-recorded, 30-90 seconds, varied tone/emotion across samples)
- **Transcription:** accurate speech-to-text on the source language
- **Translation:** translate the transcript into a target language (prioritize at least one Indian language, e.g., Hindi)
- **Voice cloning:** synthesize the translated text in the *original speaker's own voice*
- **Emotion/prosody preservation (your unique layer):** detect the emotional intensity/tone of the original delivery (e.g., excited, calm, serious, urgent) and ensure the cloned voice in the target language matches that same tone — not a flat, monotone translation
- **Lip-sync:** re-render the video so lip movements match the new audio track
- **Evaluation:** quantify lip-sync accuracy, voice-similarity to the original speaker, translation quality, and — the genuinely novel metric — emotion-preservation accuracy (does the dubbed clip "feel" like the same emotional delivery as the original, measured both algorithmically and via a small human-rated test)
- **Delivery:** a usable local/web app where someone uploads a clip and gets back a dubbed version

---

## 3. Why This Matters for Placements

AI dubbing and voice-cloning is one of the fastest-growing applied-AI categories of 2026 — **HeyGen, Synthesia, ElevenLabs (Dubbing Studio), Dubverse, Reverie, Gnani.ai, and Indian-market players (Vernacular.ai-style startups, regional OTT localization teams)** are all building exactly this category of product, and it sits squarely at the intersection of speech AI, NLP, and computer vision (lip-sync is a vision task). Unlike a generic chatbot or RAG project, this demonstrates **multi-modal applied engineering** — audio in, vision out, language understanding throughout — which is rare at fresher level and directly relevant to AI Engineer / Applied AI Engineer / ML Engineer roles at any company touching media, ed-tech, or content localization.

---

## 4. Technical Direction

**Speech-to-text (ASR):**
- OpenAI Whisper (open-weight, free) — robust, well-documented, multilingual

**Translation:**
- IndicTrans2 (AI4Bharat, free, open-weight, purpose-built for Indian languages) — preferred over generic NLLB-200 for Hindi-quality reasons; justify this choice in an ADR
- Fallback/comparison: NLLB-200 (free, open-weight, more languages)

**Voice cloning / speech synthesis:**
- Coqui XTTS-v2 (free, open-weight, zero-shot voice cloning from a short reference clip)

**Emotion/prosody detection & preservation (your differentiator — the part with no off-the-shelf solution):**
- Extract prosodic features from the original audio: pitch (F0) contour, energy/intensity, speaking rate, pause patterns — using `librosa` (free, standard audio-analysis library)
- Optionally classify a coarse emotion label (e.g., via a free pretrained speech-emotion-recognition model, such as a wav2vec2-based SER model from HuggingFace) to validate your prosody features against a known label
- Condition the XTTS generation step using these extracted prosodic targets (XTTS supports reference-audio-based style conditioning — use a reference style clip matched to the detected emotion, or apply post-synthesis pitch/rate adjustment via `librosa`/`pyworld` to nudge the output toward the source's prosodic profile)
- This is genuinely research-adjacent — be upfront in your docs that this is an applied engineering approximation, not a published, validated technique; that honesty is itself a strength in front of technical interviers

**Lip-sync:**
- Wav2Lip (free, open-source) — re-renders mouth movement to match the new audio track
- Known limitation to document: Wav2Lip can show artifacts on fast head movement or non-frontal faces — note this as a real, named limitation in your README/ADRs (sophisticated, not a weakness, to name your own system's limits)

**Evaluation:**
- Lip-sync accuracy: LSE-D / LSE-C metrics (standard lip-sync research metrics) or a simpler frame-level mouth-landmark distance if the standard metrics are too heavy to implement in time
- Voice similarity: speaker-embedding cosine similarity (e.g., via Resemblyzer or SpeechBrain speaker-verification embeddings — both free) between original and cloned voice
- Translation quality: BLEU/chrF against a manually-checked reference translation for your test clips
- **Emotion-preservation accuracy (your novel metric):** compare the SER-model's predicted emotion label on the original clip vs. the dubbed clip — report agreement rate — plus a small human-rated test (ask 5-10 people: "does this dubbed clip feel like the same emotional tone as the original?")

**Serving:**
- A simple Gradio or Streamlit app (free, fast to build) for local use: upload video → select target language → get dubbed output
- FastAPI backend if you want it cleaner/more "production" feeling for the resume — your existing skill from HireIQ

---

## 5. Scope Boundaries

**In scope:**
- One source language (English) → one target language (Hindi), with the pipeline designed to be extensible to more
- Self-recorded source clips (5-10 clips, varied emotional tone, 30-90 sec each)
- Full ASR → translation → voice cloning → emotion-conditioning → lip-sync pipeline
- The full evaluation suite described above, including the novel emotion-preservation metric
- A usable local app

**Out of scope:**
- Real-time/streaming dubbing (this is a batch/offline pipeline, not a live call dubber)
- Multi-speaker videos (diarization is a separate hard problem — single speaker per clip only)
- A polished, instant-response hosted web demo (free-tier GPU hosting realistically means a "click and wait 1-3 minutes" experience, not instant — be upfront about this, don't overpromise)
- Production-scale video lengths (keep clips short — 30-90 sec — for both compute-budget and demo-clarity reasons)

**Bonus (only attempt once core pipeline is rock-solid):**
- A second target language, to show the pipeline genuinely generalizes
- A simple web UI showing a side-by-side waveform/pitch-contour comparison of original vs. dubbed audio, visually proving the prosody-matching claim
- A short ablation: dubbed-with-emotion-preservation vs. dubbed-without, so your eval report can show the preservation layer actually helps, with numbers

---

## 6. Final Deliverable Shape

- **GitHub repo:** ASR module, translation module, voice-cloning module, prosody-extraction/conditioning module, lip-sync module, eval scripts, Gradio/Streamlit (or FastAPI+frontend) app, Docker setup
- **Demo:** a recorded before/after video (original English clip → dubbed Hindi clip, same voice, matching lips, matching emotional tone) as the centerpiece of your Loom, plus a runnable local app
- **Eval report:** lip-sync accuracy, voice-similarity score, translation BLEU/chrF, emotion-preservation agreement rate, and the human-rated mini-survey results — with the ablation comparison if you complete the bonus
- **5 ADRs minimum**, including:
  - `ADR-001-translation-model-choice.md` (IndicTrans2 vs NLLB-200)
  - `ADR-002-voice-cloning-model-choice.md` (XTTS vs alternatives)
  - `ADR-003-emotion-preservation-approach.md` (your methodology, explicitly framed as applied engineering, not novel research)
  - `ADR-004-lipsync-model-and-known-limitations.md`
  - `ADR-005-evaluation-methodology.md`
- **README** that explains the pipeline, the unique emotion-preservation layer, and lets a stranger run it locally in under 10 minutes
- **5-min Loom**: open directly with the before/after clip (don't bury your best moment), then walk through the pipeline and the eval numbers
- **3-4 resume bullets**, e.g.:
  - *"Built Resonova, an emotion-preserving AI dubbing pipeline combining Whisper, IndicTrans2, XTTS voice cloning, and Wav2Lip lip-sync, achieving X% speaker-similarity and Y% emotion-preservation agreement on a self-recorded evaluation set."*
  - *"Designed a prosody-conditioning layer using pitch/energy/rate feature extraction to prevent flat, robotic-sounding AI dubbing — validated with both automated SER agreement and human-rated evaluation."*

---

## ARCHITECTURE OVERVIEW

```
[Source Video]
      │
      ▼
[Audio Extraction] ──────────────► [Video Frames] (held for later)
      │
      ▼
[Whisper ASR] ──► transcript (source language)
      │
      ▼
[IndicTrans2 / NLLB Translation] ──► transcript (target language)
      │
      ├──────────────────────────────┐
      ▼                              ▼
[Prosody Extraction]          [Reference voice clip]
(librosa: pitch, energy,      (original speaker's audio,
rate, pauses) + optional       for XTTS voice cloning)
SER emotion label
      │                              │
      └──────────────┬───────────────┘
                      ▼
         [XTTS Voice Cloning + Prosody Conditioning]
                      │
                      ▼
              [Dubbed Audio Track]
                      │
      ┌───────────────┴───────────────┐
      ▼                                ▼
[Wav2Lip] ◄───────── [Original Video Frames]
      │
      ▼
[Final Dubbed Video Output]
      │
      ▼
[Evaluation Suite]: lip-sync accuracy, voice similarity,
translation quality, emotion-preservation agreement
```

---

## WEEK-BY-WEEK WORKING PLAN (4 weeks core build, Week 5 buffer)

### Week 1 — Get every model running individually (the "plumbing" week — budget for friction)
- Day 1-2: environment setup, Colab/Kaggle GPU access confirmed and working, record 5-10 source clips yourself (varied tone: calm explanation, excited announcement, serious warning, neutral narration)
- Day 3: get Whisper transcribing your clips correctly
- Day 4: get IndicTrans2 translating your transcripts correctly (sanity-check translations yourself, you don't need a Hindi expert for this, just read-through judgment)
- Day 5: get XTTS cloning your voice on a sample sentence (no translation/prosody yet — just prove voice cloning itself works)
- Day 6: get Wav2Lip running on a sample clip with its *original* audio first (prove the lip-sync tool itself works before adding complexity)
- **Checkpoint:** all 4 tools individually work on your own sample data, even disconnected from each other

### Week 2 — Chain it into one pipeline, get a first ugly end-to-end result
- Day 1-2: connect ASR → translation → voice cloning (no prosody conditioning yet, no lip-sync yet) — get a translated, cloned-voice audio track
- Day 3-4: connect that audio track into Wav2Lip — get your first full dubbed video, even if rough
- Day 5-6: fix the obvious breakage (audio/video desync, file format issues, GPU memory errors) — this WILL take longer than you expect, budget for it
- **Checkpoint:** one full source clip → one full dubbed output, working end-to-end, even if quality is mediocre

### Week 3 — Build the differentiator: emotion/prosody preservation + real eval
- Day 1-2: build the prosody-extraction module (pitch/energy/rate via librosa) on your original clips
- Day 3-4: implement the conditioning approach (reference-style XTTS generation or post-synthesis pitch/rate adjustment) and compare dubbed-with-conditioning vs. without on the same clip
- Day 5: build the evaluation scripts — speaker-similarity (Resemblyzer), lip-sync metric, BLEU/chrF, SER-based emotion-agreement
- Day 6: run the human-rated mini-survey (5-10 people, your classmates/friends are fine) on a few before/after pairs
- **Checkpoint:** quantitative proof that your emotion-preservation layer measurably helps vs. a flat baseline

### Week 4 — Polish, package, document, deploy
- Day 1-2: Gradio/Streamlit app (or FastAPI+simple frontend) wrapping the full pipeline
- Day 3: Docker setup for local reproducibility
- Day 4: write README, finalize all 5 ADRs
- Day 5: record the Loom — lead with the before/after clip
- Day 6: finalize eval_report.md with all numbers, write resume bullets from real results
- **Checkpoint:** "if a recruiter cloned this and ran it locally, it would work in 15 minutes, and the before/after demo speaks for itself"

### Week 5 (your unofficial buffer)
- Rehearse your live pitch and demo out loud
- If time allows: add the second target language or the ablation comparison as bonus polish
- Final GitHub/LinkedIn/portfolio updates

---

## DEPLOYMENT PLAN (honest, free-tier-realistic)

**What "deployed" means for this project, realistically:**
- **Primary:** a Dockerized, locally-runnable Gradio/Streamlit app — `docker compose up` and it works on anyone's machine with a GPU (or CPU, slower) in under 10 minutes. This satisfies the internship's "production-grade" / "deployed" requirement honestly, because video+voice+lip-sync inference genuinely needs GPU compute that free always-on web hosting doesn't reliably provide.
- **Secondary (if you want a clickable public link):** deploy the Gradio app to **HuggingFace Spaces** (free tier) — note clearly in your README that the free CPU tier will be slow (1-3 minutes per clip) compared to local GPU use, and that this is a known, named tradeoff, not a flaw you're hiding.
- **Do NOT** promise a snappy, instant, production-SLA-style live demo — that would require paid GPU hosting. Be upfront about this exact tradeoff in your design doc submission; mentors respect documented, reasoned tradeoffs far more than silent overpromising.
- **For the live showcase/panel demo specifically:** run it live on your own laptop with GPU (Colab connected, or local GPU if you have one) rather than relying on a hosted version, to avoid latency embarrassment in front of the panel.

---

## MENTOR'S CLOSING NOTE

When you pitch this to your segment mentor, open with the before/after framing, not the architecture: *"I'm building a system that takes a video of me speaking English and outputs the same video with me speaking Hindi, in my own voice, with matching lips and matching emotional tone."* Let that land before you explain Whisper/XTTS/Wav2Lip. The "wow" comes first; the engineering credibility comes right after when you show you understand exactly which parts are genuinely yours (the prosody-preservation layer and evaluation methodology) versus which parts stand on existing open models (ASR, translation, base voice cloning, base lip-sync) — own that distinction proudly, it's how real applied-AI engineering works.
