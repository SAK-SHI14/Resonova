# Vaani — Privacy Design Document

## TL;DR

Your videos are never stored, never shared, and never leave the session.

---

## What Happens to Your Uploaded Video

1. The video is received by the Gradio app (local Docker or HuggingFace Spaces session)
2. Audio is extracted in a temporary directory (`/tmp` or the session's working dir)
3. The pipeline runs: audio extraction → transcription → translation → voice cloning → lip-sync
4. The dubbed video is returned directly to your browser
5. All temporary files (extracted audio, intermediate transcripts, cloned audio) are cleared
   when the session ends or the container stops

**Nothing is logged. Nothing is saved to persistent storage. Nothing is shared.**

---

## Development Data

Source clips used during development and evaluation of Vaani were:

- Recorded by the developer (Sakshi Verma) personally for evaluation purposes
- Stored in a private Google Drive folder inaccessible to anyone else
- **Never committed to the GitHub repository** — `.gitignore` explicitly excludes
  all source video and audio files (`samples/*_source.*`, `samples/*_original.*`)
- **Never deployed to or embedded in HuggingFace Spaces**
- The example clips shown in the app (`samples/example_*.mp4`) are **dubbed OUTPUT
  videos only** — the original source footage is not included

---

## HuggingFace Spaces

When using the hosted demo at `https://huggingface.co/spaces/SAK-SHI14/vaani-dubbing`:

- Videos are processed within an isolated ZeroGPU session
- **HuggingFace's own privacy policy** applies to the hosting environment:
  https://huggingface.co/privacy
- We do not add any additional logging, analytics, or persistent storage beyond
  what HuggingFace Spaces provides by default
- Sessions are isolated and temporary — a new GPU allocation is created per request
  and released immediately after the function returns

---

## Local Docker Deployment

When running `docker compose up` on your own machine:

- All processing happens entirely locally — no network calls for inference
- Model weights are downloaded once from HuggingFace Hub and cached in a
  named Docker volume (`hf_cache`) on your own disk
- Uploaded videos never leave your machine
- Outputs are written to `./outputs/` on your local filesystem

---

## Responsible Use Statement

Vaani is built for research and portfolio demonstration purposes.

AI voice cloning and lip-sync technology carries real risks of misuse.
Users of this software are responsible for:

- **Consent**: Having explicit consent from all people whose voice or likeness appears
  in any video uploaded to Vaani
- **Legality**: Complying with applicable laws in their jurisdiction regarding
  synthetic media, impersonation, and voice cloning
- **Non-deception**: Not using Vaani to create deceptive, misleading, or harmful content

A production deployment of Vaani beyond portfolio demonstration would require:

- Explicit consent flows before processing any personal media
- Audit logging for misuse detection and accountability
- Content moderation to detect and prevent harmful output
- Compliance review against synthetic media laws in target jurisdictions

---

## Why This Document Exists

Most student AI projects do not have a privacy policy.
This one does — because the technology (voice cloning + lip-sync) has
real, documented potential for misuse, and acknowledging that honestly is
part of building AI responsibly.

Naming the risk is not weakness. It is the minimum floor of engineering maturity
for anyone building systems that touch people's voices and likenesses.
