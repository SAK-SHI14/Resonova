"""
Mimi package setup — enables `pip install -e .` for editable installs
on Colab/Kaggle so that `from mimi.asr.transcribe import transcribe` works.
"""

from setuptools import setup, find_packages

setup(
    name="mimi-dubbing",
    version="0.1.0",
    description=(
        "Mimi: Emotion-preserving AI dubbing and voice-cloned translation pipeline. "
        "English → Hindi using Whisper, IndicTrans2, XTTS-v2, and Wav2Lip."
    ),
    author="Sakshi Verma",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "notebooks"]
    ),
    python_requires=">=3.9",
    install_requires=[
        # Core requirements kept minimal here — full pinned list is in requirements.txt
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-timeout>=2.2.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
