# ADR-002: Voice Cloning Model Selection
**Status:** Accepted  
**Date:** Week 1, Phase 1  
**Authors:** Sakshi Verma  
**Context:** Project Mimi — emotion-preserving dubbing pipeline (English → Hindi)

---

## Context

Mimi needs a voice cloning / TTS model that:
1. Performs **zero-shot voice cloning** (no fine-tuning on the target speaker's data)
2. Supports **multilingual synthesis** — at minimum English and Hindi
3. Is fully open-weight (no paid API — hard requirement)
4. Runs on free-tier T4 GPU (~15 GB VRAM)
5. Produces sufficient voice similarity that a casual listener can recognize the speaker's identity across languages

---

## Options Evaluated

### Option A: Coqui XTTS-v2 (`tts_models/multilingual/multi-dataset/xtts_v2`)

| Criterion | Assessment |
|---|---|
| Zero-shot cloning | ✅ Yes — from a single reference clip ≥6 seconds |
| Hindi support | ✅ Yes (17 languages including `hi`) |
| Open-weight | ✅ Coqui Public Model License (free for non-commercial) |
| VRAM | ~4–5 GB — fits T4 |
| Voice similarity | High — among the best open-weight zero-shot TTS models as of 2024 |
| Installation | `pip install TTS==0.22.0` — straightforward |
| Prosody control | Limited — output style influenced by reference clip but not directly controllable |
| Active maintenance | ⚠️ Coqui the company shut down in 2024; model weights remain available on HuggingFace |

### Option B: Bark (Suno AI)

| Criterion | Assessment |
|---|---|
| Zero-shot cloning | ⚠️ Limited — produces consistent voice from text prompt, not true speaker cloning |
| Hindi support | ⚠️ Limited — primarily English and European languages |
| Open-weight | ✅ MIT licensed |
| VRAM | ~4–6 GB for large variant |
| Voice similarity | Lower than XTTS-v2 for speaker identity preservation |
| Suitability for Mimi | ❌ Hindi support is insufficient for the primary use case |

### Option C: YourTTS

| Criterion | Assessment |
|---|---|
| Zero-shot cloning | ✅ Yes |
| Hindi support | ❌ Not supported (English, Portuguese, French only) |
| Open-weight | ✅ MIT licensed |
| Suitability for Mimi | ❌ Eliminated due to no Hindi support |

### Option D: OpenVoice (MyShell AI)

| Criterion | Assessment |
|---|---|
| Zero-shot cloning | ✅ Yes |
| Hindi support | ⚠️ Limited / experimental as of 2024 |
| Open-weight | ✅ MIT licensed |
| VRAM | ~2–3 GB |
| Voice similarity | Good for English; Hindi quality untested at scale |
| Active maintenance | ✅ Actively maintained as of 2024 |
| Suitability for Mimi | Viable alternative but less documented for Hindi than XTTS-v2 |

---

## Decision

**Chose Coqui XTTS-v2 (Option A).**

### Rationale

1. **Hindi is non-negotiable**: Mimi's primary output language is Hindi. XTTS-v2 is the most tested open-weight zero-shot TTS model with documented Hindi support. YourTTS and Bark are eliminated outright.

2. **Voice similarity quality**: XTTS-v2 produces the highest speaker-similarity scores among the evaluated open-weight models for the English reference → Hindi synthesis use case.

3. **Zero-shot cloning from a short clip**: XTTS-v2 requires only 6+ seconds of reference audio with no fine-tuning — this matches Mimi's design goal of working with any speaker's recording without training.

4. **Documented in the prosody pipeline**: XTTS-v2's `speaker_wav` parameter accepts a reference clip that influences not just speaker identity but also prosody style — this is the hook Phase 3 will use for reference-style emotion conditioning (ADR-004, to be written in Phase 3).

### Coqui Shutdown Risk Mitigation

Coqui the company shut down in January 2024. This is a real risk worth documenting:
- The model weights remain on HuggingFace and will not disappear in the near term
- The `TTS` pip package continues to work from PyPI
- Mimi pins the working version (`TTS==0.22.0`) to prevent silent breakage from future changes
- If XTTS-v2 becomes unavailable, OpenVoice is the documented fallback

### Known Limitations (documented, not hidden)

1. **Hindi synthesis quality**: XTTS-v2 was trained on significantly more English than Hindi data. Hindi output is audibly recognizable as the cloned speaker but has more robotic/unnatural prosody than English output. This is a known, named limitation in the README.

2. **Prosody is reference-influenced, not controllable**: The cloned voice's pitch and rate reflect the reference clip and text, not a directly controllable parameter. Phase 3 addresses this via post-synthesis adjustment.

3. **Non-commercial license**: Coqui Public Model License permits non-commercial use. This project is a non-commercial portfolio/internship project — acceptable. If commercialized later, switch to a permissively licensed alternative.

---

## Consequences

**Positive:**
- Best available zero-shot Hindi voice cloning on free compute
- Simple installation
- Reference-clip conditioning hook available for Phase 3 emotion work

**Negative / Accepted Tradeoffs:**
- Non-commercial license (acceptable for portfolio context)
- Coqui company is defunct — model availability dependent on HuggingFace hosting
- Hindi quality lower than English — documented as a named limitation

---

## References
- XTTS-v2 model card: https://huggingface.co/coqui/XTTS-v2
- Coqui TTS GitHub: https://github.com/coqui-ai/TTS
- XTTS-v2 paper: https://arxiv.org/abs/2406.04904
- OpenVoice (documented fallback): https://github.com/myshell-ai/OpenVoice
