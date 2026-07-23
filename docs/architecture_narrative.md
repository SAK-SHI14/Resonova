# 🏗️ Resonova — Container & System Architecture Narrative

This document provides a C4 Level 2 Container Architecture view of **Resonova**, describing how components interact, manage memory, isolate runtime dependencies, and process audio-visual streams.

---

## 1. C4 Level 2 Container Diagram

```mermaid
graph TD
    User["👤 User / Client Browser"] -->|HTTP / WebSocket| UI["🖥️ Web Application Interface (Gradio 4)"]
    
    subgraph "Resonova Application Boundary"
        UI -->|Upload Video| Pipe["⚙️ Master Pipeline Orchestrator (pipeline.py)"]
        
        subgraph "Audio Processing & AI Ingestion Subsystem"
            Pipe -->|Extract Audio| FFmpeg1["🎵 FFmpeg Audio Demuxer"]
            FFmpeg1 -->|16kHz PCM WAV| ASR["🎙️ OpenAI Whisper ASR (transcribe.py)"]
            ASR -->|English Text + Timestamps| NMT["🌐 IndicTrans2 NMT (translate.py)"]
            NMT -->|Hindi Text (Devanagari)| Prosody["📊 Prosody Profiler (extract.py)"]
        end
        
        subgraph "Voice Synthesis & Prosody Conditioning Engine"
            FFmpeg1 -->|3s Voice Reference| TTS["🗣️ Coqui XTTS-v2 Engine (clone_voice.py)"]
            Prosody -->|F0 Contour & RMS Energy Profile| Condition["⚡ Style Conditioning Layer (conditioning.py)"]
            Condition -->|Conditioned Vectors| TTS
            TTS -->|Cloned Hindi Audio| AudioSync["⏱️ FFmpeg Duration Stretch (atempo)"]
        end
        
        subgraph "Subprocess-Isolated Visual Synchronization"
            AudioSync -->|Duration-Matched Hindi Audio| Wav2LipProc["🎬 Wav2Lip Subprocess Executor (lipsync.py)"]
            Pipe -->|Source MP4 Frames| Wav2LipProc
            Wav2LipProc -->|PyTorch 1.x Isolated Subprocess| W2L["👄 Wav2Lip-GAN Inference Model"]
            W2L -->|Rendered Lip-Synced Frames| FFmpegMux["🎞️ FFmpeg Video-Audio Muxer"]
        end
        
        FFmpegMux -->|Final MP4 Dubbed Video| Pipe
        Pipe -->|Evaluation Metrics| Eval["📈 Evaluation & Metric Suite (metrics.py)"]
        Eval -->|Similarity & SER Scores| UI
    end
```

---

## 2. Container Architecture Narrative

### Container Breakdown & Interfaces

1. **Web Application Interface (`resonova/app/app.py`)**:
   - Built on Gradio 4 with embedded CSS styling (`resonova.css`).
   - Serves as the reactive presentation layer receiving client video uploads, language target selections, and configuration toggles (e.g., Demo Mode).
   - Displays real-time pipeline status, side-by-side transcripts, synthesized audio, final dubbed MP4 video, and visual evaluation report cards.

2. **Master Pipeline Orchestrator (`resonova/pipeline.py`)**:
   - Controls state transitions across all 7 pipeline stages.
   - Manages temporary file lifecycle within isolated execution directories (`/tmp/resonova_stage_*`).
   - Enforces sequential model loading and explicit garbage collection (`torch.cuda.empty_cache()` and `gc.collect()`) between pipeline steps to constrain peak VRAM under 4.5 GB on T4 GPUs.

3. **ASR & Machine Translation Container (`resonova/asr/` & `resonova/translation/`)**:
   - **ASR**: Loads `Whisper-medium` to perform robust English speech recognition, outputting text along with word-level timing boundaries.
   - **NMT**: Passes English transcripts through `AI4Bharat IndicTrans2-1B` using Devanagari script tokenization (`hin_Deva`). If HuggingFace transformer weights fail to load, automatically fails over to `Helsinki-NLP opus-mt-en-hi`.

4. **Prosody Conditioning Engine (`resonova/prosody/`)**:
   - Extracts pitch contours (F0 fundamental frequency) via Pyin/YIN algorithms and RMS energy envelopes using Librosa.
   - Normalizes energy envelopes and computes RMS scaling factors to condition Coqui XTTS-v2 zero-shot voice synthesis, ensuring whisper/shout emotional dynamics are preserved.

5. **Subprocess-Isolated Lip Synchronization (`resonova/lipsync/` & `Wav2Lip/`)**:
   - Executed as an isolated Python subprocess to resolve binary and PyTorch version incompatibilities (PyTorch 2.x host environment vs PyTorch 1.x Wav2Lip GAN requirements).
   - Receives target audio and input video frames, detects face bounding boxes using S3FD, and synthesizes lower-face lip movements matching Hindi speech phonemes.

6. **Audio-Video Muxer & Duration Synchronizer**:
   - Utilizes FFmpeg `atempo` audio filter chains to stretch or compress synthesized Hindi speech audio to match the exact duration of the source English video clip.
   - Muxes aligned audio with Wav2Lip rendered video frames into a web-optimized MP4 container (`libx264` video codec, `aac` audio codec).
