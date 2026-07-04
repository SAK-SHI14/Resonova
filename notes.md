# Vaani — Dependency & Troubleshooting Notes
_Living document. Updated as issues are hit and resolved. Becomes the backbone of ADRs and README troubleshooting section._

---

## Environment

| Field | Value |
|---|---|
| Target compute | Google Colab T4 / Kaggle T4 / P100 |
| Python | 3.9 (pinned — Wav2Lip requires <3.10 for some deps) |
| CUDA | 11.3+ |

---

## Phase 1 Log

### [ENTRY TEMPLATE]
```
DATE: YYYY-MM-DD
COMPONENT: <asr|translation|voice_cloning|lipsync>
ISSUE: <what broke>
ROOT CAUSE: <why it broke>
FIX: <exact command or version change>
PINNED VERSION: <package==x.y.z>
```

---

## Known Pre-Documented Wav2Lip Dependency Landmines

These are known issues documented BEFORE hitting them, based on community reports.
Confirm and correct during Phase 1 execution.

| Package | Required | Common Wrong Version | Why It Breaks |
|---|---|---|---|
| torch | 1.9.x–1.12.x | 2.x | face_alignment API incompatibility |
| torchvision | 0.10.x–0.13.x | 0.15+ | matches torch version constraint |
| face_alignment | 1.3.5 | 1.4+ | `face_alignment.FaceAlignment` API changed |
| batch_face_detection | see commit below | pip latest | often incompatible with face_alignment 1.3.5 |
| librosa | 0.8.1 | 0.10+ | `librosa.filters.mel` signature changed |
| numpy | <1.24.0 | 1.24+ | `np.bool`, `np.int` deprecated/removed |
| opencv-python | 4.5.x | 4.8+ | minor API surface changes |

**Wav2Lip checkpoint:** Use `wav2lip_gan.pth` (higher visual quality) from official release.
**face_detection repo:** `https://github.com/hhj1897/face_detection` — specific commit may be needed.

---

## GPU Memory Profile (per model, T4 = ~15 GB VRAM)

| Model | Approx VRAM | Fit on T4? | Notes |
|---|---|---|---|
| Whisper medium | ~1.5 GB | ✅ Yes | Default. large-v3 = ~3 GB, also fits |
| Whisper large-v3 | ~3.0 GB | ✅ Yes | Use for final quality pass |
| IndicTrans2-1B | ~4.0 GB | ✅ Yes | Use this variant |
| IndicTrans2-3.3B | ~10 GB | ⚠️ Marginal | Do NOT use on free tier |
| XTTS-v2 | ~4–5 GB | ✅ Yes | Load time ~30–60s |
| Wav2Lip | ~2.0 GB | ✅ Yes | Dependency issues, not VRAM, are the risk |
| **Total (sequential)** | **~4–5 GB peak** | ✅ Yes | Load one at a time, offload after use |

**Strategy:** Do NOT load all models simultaneously. Load → run → del model → torch.cuda.empty_cache() between stages.

---

## Unresolved Items
_(add here as issues arise, mark resolved with ✅)_

- [ ] Confirm exact working torch + face_alignment combo on current Colab runtime
- [ ] Confirm IndicTrans2 tokenizer language code format (`eng_Latn` vs `en` etc.)
- [ ] Confirm XTTS-v2 minimum reference clip length (documented as 6s, verify)

---
