"""
Vaani — Translation Module
===========================
Wrapper around AI4Bharat IndicTrans2 for English → Hindi (and other Indic language) translation.

Model: ai4bharat/indictrans2-en-indic-1B  (open-weight, runs fully locally — no API calls)

ADR reference: docs/adrs/ADR-001-translation-model.md

GPU Memory (T4 reference):
  - indictrans2-en-indic-1B  : ~4 GB VRAM  ← USE THIS on free-tier
  - indictrans2-en-indic-3.3B: ~10 GB VRAM ← DO NOT USE on free-tier (marginal/OOM risk)

Language Code Format (IndicTrans2 uses BCP-47 + script):
  English  : "eng_Latn"
  Hindi    : "hin_Deva"
  Tamil    : "tam_Taml"
  Bengali  : "ben_Beng"
  Full list: https://github.com/AI4Bharat/IndicTrans2#supported-languages

Usage:
    from vaani.translation.translate import translate
    hindi_text = translate("Hello, how are you?", "eng_Latn", "hin_Deva")
"""

import os
import time
from typing import Optional

from vaani.exceptions import TranslationError
from vaani.logger import get_logger

logger = get_logger(__name__)

# Cache loaded model/tokenizer to avoid re-loading within the same session.
_model_cache: dict = {}

# Supported language codes (IndicTrans2 format). Extend as needed.
SUPPORTED_LANGS = {
    "eng_Latn", "hin_Deva", "tam_Taml", "tel_Telu",
    "ben_Beng", "mar_Deva", "guj_Gujr", "kan_Knda",
    "mal_Mlym", "pan_Guru", "urd_Arab",
}

# Devanagari Unicode block range — used for output validation
_DEVANAGARI_START = 0x0900
_DEVANAGARI_END   = 0x097F


def _load_model(model_name: str):
    """
    Load (or retrieve from cache) IndicTrans2 model and tokenizer.

    IndicTrans2 uses HuggingFace Transformers under the hood but requires
    its own custom tokenizer from the IndicTrans2 repository.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Tuple of (model, tokenizer, inference_engine).

    Raises:
        TranslationError: If the model cannot be loaded.
    """
    if model_name in _model_cache:
        logger.debug("IndicTrans2 model '%s' loaded from cache.", model_name)
        return _model_cache[model_name]

    logger.info(
        "Loading IndicTrans2 model '%s' — first run may take 1–3 min for download...",
        model_name,
    )
    t0 = time.perf_counter()

    try:
        # IndicTrans2 requires its own inference engine from the cloned repo.
        # The repo must be installed with: pip install -e IndicTrans2/
        from IndicTransToolkit import IndicProcessor  # noqa: PLC0415
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

        # Move to GPU if available
        import torch  # noqa: PLC0415
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        processor = IndicProcessor(inference=True)

    except ImportError as exc:
        raise TranslationError(
            "IndicTrans2 dependencies not installed. "
            "Run: git clone https://github.com/AI4Bharat/IndicTrans2 && "
            "pip install -e IndicTrans2/ "
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:
        raise TranslationError(
            f"Failed to load IndicTrans2 model '{model_name}': {exc}\n"
            "Check GPU memory (model requires ~4 GB VRAM for 1B variant). "
            "Ensure trust_remote_code=True is passed."
        ) from exc

    elapsed = time.perf_counter() - t0
    logger.info("IndicTrans2 model loaded in %.1f s | device='%s'", elapsed, device)

    cached = (model, tokenizer, processor)
    _model_cache[model_name] = cached
    return cached


def translate(
    text: str,
    source_lang: str = "eng_Latn",
    target_lang: str = "hin_Deva",
    model_name: Optional[str] = None,
    max_length: int = 512,
) -> str:
    """
    Translate text from source language to target language using IndicTrans2.

    Args:
        text:         The input text to translate. Must be non-empty.
        source_lang:  Source language in IndicTrans2 format (e.g., "eng_Latn").
        target_lang:  Target language in IndicTrans2 format (e.g., "hin_Deva").
        model_name:   HuggingFace model name. Defaults to the value of the
                      ``INDICTRANS2_MODEL_NAME`` environment variable, falling
                      back to ``ai4bharat/indictrans2-en-indic-1B``.
        max_length:   Maximum token length for generation. Default 512 is safe
                      for clips up to ~90 seconds of speech.

    Returns:
        Translated text as a string.

    Raises:
        ValueError:       If ``text`` is empty or language codes are unsupported.
        TranslationError: If the model fails or returns empty/identical output.

    Example:
        >>> result = translate("Hello, how are you?", "eng_Latn", "hin_Deva")
        >>> print(result)
        'नमस्ते, आप कैसे हैं?'
    """
    # --- Input validation ---
    if not text or not text.strip():
        raise TranslationError(
            "translate() received empty or whitespace-only input text. "
            "Ensure the ASR transcription step produced output before calling translate()."
        )

    if source_lang not in SUPPORTED_LANGS:
        raise ValueError(
            f"Unsupported source_lang: '{source_lang}'. "
            f"Supported codes: {sorted(SUPPORTED_LANGS)}. "
            "See IndicTrans2 documentation for the full list."
        )
    if target_lang not in SUPPORTED_LANGS:
        raise ValueError(
            f"Unsupported target_lang: '{target_lang}'. "
            f"Supported codes: {sorted(SUPPORTED_LANGS)}."
        )

    # --- Resolve model name ---
    if model_name is None:
        model_name = os.environ.get(
            "INDICTRANS2_MODEL_NAME", "ai4bharat/indictrans2-en-indic-1B"
        )

    logger.info(
        "[Translation] Starting | src='%s' | tgt='%s' | model='%s' | chars=%d",
        source_lang, target_lang, model_name, len(text),
    )
    t_start = time.perf_counter()

    # --- Load model ---
    model, tokenizer, processor = _load_model(model_name)

    # --- Preprocess ---
    try:
        import torch  # noqa: PLC0415

        batch = processor.preprocess_batch(
            [text.strip()],
            src_lang=source_lang,
            tgt_lang=target_lang,
        )
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

    except Exception as exc:
        raise TranslationError(
            f"IndicTrans2 preprocessing failed: {exc}"
        ) from exc

    # --- Generate ---
    try:
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=max_length,
                num_beams=5,
                num_return_sequences=1,
            )

        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        translations = processor.postprocess_batch(generated_tokens, lang=target_lang)

    except Exception as exc:
        raise TranslationError(
            f"IndicTrans2 generation failed: {exc}\n"
            "Possible causes: GPU OOM (try shorter input), model not loaded correctly, "
            "or tokenizer mismatch. Check notes.md for known fixes."
        ) from exc

    # --- Validate output ---
    if not translations or not translations[0].strip():
        raise TranslationError(
            "IndicTrans2 returned an empty translation. "
            f"Input was: '{text[:80]}...' | src='{source_lang}' | tgt='{target_lang}'. "
            "This may indicate a tokenizer issue or unsupported language pair."
        )

    translation = translations[0].strip()

    # Sanity check: if translating to Hindi, output should contain Devanagari script
    if target_lang == "hin_Deva":
        has_devanagari = any(
            _DEVANAGARI_START <= ord(c) <= _DEVANAGARI_END for c in translation
        )
        if not has_devanagari:
            logger.warning(
                "[Translation] Output for hin_Deva target does not contain Devanagari "
                "script — output may be incorrect. Review: '%s'",
                translation[:80],
            )

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[Translation] Complete | duration=%.2f s | output_chars=%d | preview='%s...'",
        elapsed,
        len(translation),
        translation[:60],
    )

    return translation


def get_supported_languages() -> set:
    """Return the set of supported IndicTrans2 language codes."""
    return SUPPORTED_LANGS.copy()


def unload_model() -> None:
    """Clear loaded IndicTrans2 models from cache and free CUDA memory."""
    global _model_cache
    if _model_cache:
        logger.info("Unloading IndicTrans2 models and clearing CUDA cache...")
        _model_cache.clear()

        import gc  # noqa: PLC0415
        import torch  # noqa: PLC0415

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("IndicTrans2 models cache cleared.")

