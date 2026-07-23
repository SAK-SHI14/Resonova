# 🖼️ Resonova — Public Showcase Slide Content & Narrative

Document Version: **1.0-final**  
Asset: **Part 3 Showcase Slide Specification (1 Slide PNG/PDF)**

---

## 🎨 Slide Layout Architecture (Single Slide Blueprint)

```
+---------------------------------------------------------------------------------------------------+
| 🎙️ Resonova — Zero-Shot Emotion-Preserving AI Video Dubbing                                      |
| Student: Sakshi Verma | B.Tech CSE (AI & Data) | Applied AI Track | Futurense Internship 2026      |
+----------------──────────────────┬─────────────────────────────────┬────────────────--------------+
| 💡 THE PROBLEM                   | 🏗️ THE ARCHITECTURE             | 📊 VERIFIED RESULTS          |
| Traditional AI dubbing loses     | 1. Audio: FFmpeg 16kHz WAV       | - Speaker Sim: 86.50%        |
| speaker emotion & voice identity.| 2. ASR: Whisper-medium           | - Emotion Preserved: 80.00%  |
| Resonova fixes this via custom   | 3. NMT: IndicTrans2-1B          | - Ablation Delta: +40.00pp   |
| RMS energy matching & PYIN F0    | 4. TTS: XTTS-v2 + Conditioning  | - Translation BLEU: 0.5120   |
| pitch tracking.                  | 5. Sync: FFmpeg atempo          | - Test Suite: 83/83 Passed   |
|                                  | 6. LipSync: Wav2Lip-GAN         | - Peak VRAM: 4.5 GB (T4)     |
+--------------------------------──┴─────────────────────────────────┴────────────────--------------+
| 🌐 Live Demo: https://0737df54ab8c099319.gradio.live | 📦 Repo: github.com/SAK-SHI14/Resonova         |
+---------------------------------------------------------------------------------------------------+
```

---

## 🎤 30-Second Showcase Elevator Pitch Script

> "Good evening everyone. I'm Sakshi Verma, and I built **Resonova**—a zero-shot, emotion-preserving AI video dubbing pipeline that translates English videos into Hindi while keeping the speaker's true voice, emotional intensity, and realistic lip movements intact.
> 
> Most existing dubbing tools output flat, monotone audio because text translation strips away vocal energy. Resonova extracts the original speaker's RMS energy profile and pitch contour, feeding them into a conditioned XTTS-v2 voice cloning engine.
> 
> The results speak for themselves: an **80% emotion preservation rate**, representing a **+40 percentage point improvement** over baseline models, and **86.50% speaker identity similarity**, all running under **4.5 GB VRAM at zero hardware cost**. Thank you!"
