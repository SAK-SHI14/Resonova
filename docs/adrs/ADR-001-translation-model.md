# ADR-001: Translation Model Selection
**Status:** Accepted  
**Date:** Week 1, Phase 1  
**Authors:** Sakshi Verma  
**Context:** Project Mimi — emotion-preserving dubbing pipeline (English → Hindi)

---

## Context

Mimi needs a translation model that:
1. Produces high-quality English → Hindi translation (primary pair)
2. Is fully open-weight (no paid API — hard requirement)
3. Can run on a free-tier T4 GPU (~15 GB VRAM)
4. Is actively maintained and designed for Indic languages
5. Is extensible to additional Indic language pairs later (per product roadmap)

Two primary candidates were evaluated:

---

## Options Evaluated

### Option A: AI4Bharat IndicTrans2 (`ai4bharat/indictrans2-en-indic-1B`)

| Criterion | Assessment |
|---|---|
| Language focus | Built specifically for Indian languages — 22 Indic languages supported |
| En→Hi quality | State-of-the-art for English→Hindi as of 2024 (reported BLEU scores ~35+ on Flores-200) |
| Model size | 1B params (~4 GB VRAM) — fits T4 free tier comfortably |
| Open-weight | ✅ Apache 2.0 licensed |
| Indic script awareness | ✅ Trained on Devanagari, aware of script-specific morphology |
| Active maintenance | ✅ AI4Bharat is an active academic lab (IIT Madras) with ongoing updates |
| Installation | ⚠️ Not pip-installable; requires cloning the repo and `pip install -e .` |
| HuggingFace Hub | ✅ Available via `from_pretrained()` after repo install |

### Option B: Meta NLLB-200 (`facebook/nllb-200-distilled-600M`)

| Criterion | Assessment |
|---|---|
| Language focus | 200 languages including Hindi — general-purpose multilingual |
| En→Hi quality | Good but lower BLEU on Indic-specific benchmarks vs IndicTrans2 |
| Model size | 600M distilled (~2.5 GB VRAM) — fits T4; 3.3B variant does not |
| Open-weight | ✅ CC-BY-NC 4.0 (non-commercial) |
| Indic script awareness | Moderate — general multilingual training, not Indic-specialized |
| Active maintenance | Meta research release; less actively updated for Indic use cases |
| Installation | ✅ Pip-installable via `transformers` — simpler setup |
| HuggingFace Hub | ✅ Standard `from_pretrained()` |

---

## Decision

**Chose IndicTrans2 (Option A).**

### Rationale

1. **Domain fit**: Mimi's primary use case is English → Hindi for media dubbing. IndicTrans2 was trained and benchmarked specifically on Indic language pairs — this is exactly its design goal, not a side use case.

2. **Quality on primary pair**: IndicTrans2 consistently outperforms NLLB-200 on English→Hindi benchmarks (Flores-200, IN22 benchmarks). For a portfolio project being evaluated on technical depth, using the SOTA model for the specific task is the right choice.

3. **Extensibility**: IndicTrans2 covers all 22 Indian official languages. If Mimi later adds Tamil, Bengali, or other targets, the same model handles it without a model swap.

4. **License**: Apache 2.0 (vs NLLB's CC-BY-NC). Apache 2.0 is more permissive for portfolio/demo use and potential future commercial application.

### Why Not NLLB-200

NLLB-200 remains documented as a **fallback option** for two scenarios:
- If IndicTrans2 installation fails on a fresh environment (its non-standard install is a real friction point)
- If the 600M NLLB model is needed for very constrained VRAM environments (e.g., running on <4 GB GPU alongside other models)

The fallback is documented in `translation/translate.py` but not wired in Phase 1.

---

## Consequences

**Positive:**
- Best available quality for English → Hindi on free compute
- Apache 2.0 license — no commercial restriction
- Covers Indic language roadmap without future model changes

**Negative / Accepted Tradeoffs:**
- Non-standard installation (clone repo + `pip install -e .`) — documented step-by-step in README and Colab notebook
- `trust_remote_code=True` required — this is normal for this model, but worth noting for security review in enterprise contexts (not relevant for this portfolio project)
- Language codes use IndicTrans2 format (`eng_Latn`, `hin_Deva`) not standard BCP-47 — documented clearly in translation module docstring

---

## References
- IndicTrans2 paper: https://arxiv.org/abs/2305.16307
- IndicTrans2 repository: https://github.com/AI4Bharat/IndicTrans2
- NLLB-200 paper: https://arxiv.org/abs/2207.04672
- Flores-200 benchmark: https://github.com/facebookresearch/flores/tree/main/flores200
