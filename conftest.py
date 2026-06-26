"""
conftest.py — Shared pytest fixtures for the Mimi test suite.

Fixtures defined here are available to all tests without explicit imports.
"""

import struct
import wave
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------

def _write_wav(path: Path, duration_s: float, sample_rate: int = 16000) -> Path:
    """Write a mono 16-bit sine-wave WAV file and return the path."""
    import math
    n_samples = int(duration_s * sample_rate)
    samples = [
        int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for i in range(n_samples)
    ]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    return path


@pytest.fixture(scope="session")
def sample_wav_2s(tmp_path_factory) -> Path:
    """2-second WAV file, session-scoped (created once, reused across all tests)."""
    p = tmp_path_factory.mktemp("audio") / "sample_2s.wav"
    return _write_wav(p, duration_s=2.0)


@pytest.fixture(scope="session")
def sample_wav_8s(tmp_path_factory) -> Path:
    """8-second WAV file — satisfies XTTS-v2 minimum reference duration."""
    p = tmp_path_factory.mktemp("audio") / "sample_8s.wav"
    return _write_wav(p, duration_s=8.0)


@pytest.fixture(scope="session")
def sample_wav_30s(tmp_path_factory) -> Path:
    """30-second WAV file — simulates a full source clip."""
    p = tmp_path_factory.mktemp("audio") / "sample_30s.wav"
    return _write_wav(p, duration_s=30.0)


# ---------------------------------------------------------------------------
# Video fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_mp4(tmp_path_factory) -> Path:
    """Minimal MP4 stub file (bytes only — not a real playable video)."""
    p = tmp_path_factory.mktemp("video") / "sample.mp4"
    # Minimal ftyp box header for MP4 detection
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    return p


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def set_test_log_level(monkeypatch):
    """Force INFO log level during tests for cleaner output."""
    monkeypatch.setenv("LOG_LEVEL", "INFO")
