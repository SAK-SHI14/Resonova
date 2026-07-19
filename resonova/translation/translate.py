"""
Resonova — Translation Module
===========================
Wrapper around AI4Bharat IndicTrans2 and Helsinki-NLP for English → Hindi translation.

Supported translation backends:
  - indictrans2 (Requires IndicTransToolkit & transformers)
  - helsinki (Fallback: Helsinki-NLP/opus-mt-en-hi via transformers pipeline)
"""

import os
import time
from typing import Optional

from resonova.exceptions import TranslationError
from resonova.logger import get_logger

logger = get_logger(__name__)

# Supported language codes (IndicTrans2 format). Extend as needed.
SUPPORTED_LANGS = {
    "eng_Latn", "hin_Deva", "tam_Taml", "tel_Telu",
    "ben_Beng", "mar_Deva", "guj_Gujr", "kan_Knda",
    "mal_Mlym", "pan_Guru", "urd_Arab",
}

# Devanagari Unicode block range — used for output validation
_DEVANAGARI_START = 0x0900
_DEVANAGARI_END   = 0x097F

# Backend determination
TRANSLATION_BACKEND = "none"
INDICTRANS_MODEL = None  # lazy load holder: (model, tokenizer, processor)
HELSINKI_PIPELINE = None  # lazy load holder

try:
    from IndicTransToolkit import IndicProcessor  # noqa: PLC0415
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415
    TRANSLATION_BACKEND = "indictrans2"
    logger.info("Translation backend: IndicTrans2")
except ImportError:
    logger.warning("IndicTrans2 not available, falling back to Helsinki-NLP")

if TRANSLATION_BACKEND == "none":
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa: PLC0415
        TRANSLATION_BACKEND = "helsinki"
        logger.info("Translation backend: Helsinki-NLP opus-mt-en-hi")
    except ImportError:
        logger.error("No translation backend available!")


def _load_model(model_name: str):
    """
    Load (or retrieve from cache) translation model.
    Supporting mock patching in unit tests.
    """
    global INDICTRANS_MODEL, HELSINKI_PIPELINE
    
    if TRANSLATION_BACKEND == "indictrans2":
        if INDICTRANS_MODEL is None:
            # Trigger lazy load
            try:
                _translate_indictrans2("test", "eng_Latn", "hin_Deva", model_name)
            except Exception:
                pass
        if INDICTRANS_MODEL is not None:
            model, tokenizer, processor = INDICTRANS_MODEL
            return (model, tokenizer, processor, False)
        return (None, None, None, False)
    elif TRANSLATION_BACKEND == "helsinki":
        if HELSINKI_PIPELINE is None:
            try:
                _translate_helsinki("test")
            except Exception:
                pass
        if HELSINKI_PIPELINE is not None:
            model, tokenizer = HELSINKI_PIPELINE
            return (model, tokenizer, None, True)
        return (None, None, None, True)
    else:
        return (None, None, None, False)


def _translate_indictrans2(text: str, source_lang: str, target_lang: str, model_name: Optional[str] = None) -> str:
    global INDICTRANS_MODEL
    
    if model_name is None:
        model_name = os.environ.get(
            "INDICTRANS2_MODEL_NAME", "ai4bharat/indictrans2-en-indic-1B"
        )
        
    if INDICTRANS_MODEL is None:
        logger.info("Loading IndicTrans2 model '%s'...", model_name)
        t0 = time.perf_counter()
        try:
            from IndicTransToolkit import IndicProcessor  # noqa: PLC0415
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415
            import torch  # noqa: PLC0415

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()

            processor = IndicProcessor(inference=True)
            INDICTRANS_MODEL = (model, tokenizer, processor)
            elapsed = time.perf_counter() - t0
            logger.info("IndicTrans2 model loaded in %.1f s | device='%s'", elapsed, device)
        except Exception as exc:
            raise TranslationError(f"Failed to load IndicTrans2 model: {exc}") from exc
            
    model, tokenizer, processor = INDICTRANS_MODEL
    
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

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=512,
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
        raise TranslationError(f"IndicTrans2 translation generation failed: {exc}") from exc

    if not translations or not translations[0].strip():
        raise TranslationError("IndicTrans2 returned an empty translation.")
    return translations[0].strip()


def _translate_helsinki(text: str) -> str:
    global HELSINKI_PIPELINE
    if HELSINKI_PIPELINE is None:
        logger.info("Loading Helsinki-NLP translator (opus-mt-en-hi)...")
        t0 = time.perf_counter()
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa: PLC0415
            import torch  # noqa: PLC0415
            
            tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-hi')
            model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-hi')
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            
            HELSINKI_PIPELINE = (model, tokenizer)
            elapsed = time.perf_counter() - t0
            logger.info("Helsinki-NLP translator loaded in %.1f s", elapsed)
        except Exception as exc:
            raise TranslationError(f"Failed to load Helsinki-NLP translator: {exc}") from exc
            
    model, tokenizer = HELSINKI_PIPELINE
    try:
        import torch  # noqa: PLC0415
        device = next(model.parameters()).device
        inputs = tokenizer(text.strip(), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as exc:
        raise TranslationError(f"Helsinki-NLP translation generation failed: {exc}") from exc

    if not translation or not translation.strip():
        raise TranslationError("Helsinki-NLP returned an empty translation.")
    return translation.strip()


def translate(
    text: str,
    source_lang: str = "eng_Latn",
    target_lang: str = "hin_Deva",
    model_name: Optional[str] = None,
    max_length: int = 512,
) -> str:
    """
    Translate text from source language to target language.
    """
    if not text or not text.strip():
        raise TranslationError(
            "translate() received empty or whitespace-only input text."
        )

    if source_lang not in SUPPORTED_LANGS:
        raise ValueError(
            f"Unsupported source_lang: '{source_lang}'."
        )
    if target_lang not in SUPPORTED_LANGS:
        raise ValueError(
            f"Unsupported target_lang: '{target_lang}'."
        )

    logger.info(
        "[Translation] Starting | src='%s' | tgt='%s' | backend='%s' | chars=%d",
        source_lang, target_lang, TRANSLATION_BACKEND, len(text),
    )
    t_start = time.perf_counter()

    # Get model data supporting unit test mocking of _load_model
    model_data = _load_model(model_name)
    is_fallback = model_data[3] if (isinstance(model_data, tuple) and len(model_data) > 3) else False

    if is_fallback:
        translation = _translate_helsinki(text)
    elif TRANSLATION_BACKEND == "indictrans2" or (isinstance(model_data, tuple) and model_data[0] is not None):
        # Mocks or real IndicTrans2
        if isinstance(model_data, tuple) and len(model_data) >= 3 and model_data[0] is not None:
            # Use data from mocked _load_model
            model, tokenizer, processor = model_data[0], model_data[1], model_data[2]
            try:
                import torch  # noqa: PLC0415
                batch = processor.preprocess_batch([text.strip()], src_lang=source_lang, tgt_lang=target_lang)
                inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    generated_tokens = model.generate(**inputs, use_cache=True, min_length=0, max_length=max_length)
                with tokenizer.as_target_tokenizer():
                    generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), skip_special_tokens=True)
                translations = processor.postprocess_batch(generated_tokens, lang=target_lang)
                translation = translations[0].strip()
            except Exception as exc:
                raise TranslationError(f"IndicTrans2 execution failed: {exc}") from exc
        else:
            translation = _translate_indictrans2(text, source_lang, target_lang, model_name)
    else:
        translation = _translate_helsinki(text)

    # Empty translation output validation check
    if not translation or not translation.strip():
        raise TranslationError("Translation backend returned an empty translation.")

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
    """Return the set of supported language codes."""
    return SUPPORTED_LANGS.copy()


def unload_model() -> None:
    """Clear loaded translation models from cache and free CUDA memory."""
    global INDICTRANS_MODEL, HELSINKI_PIPELINE
    INDICTRANS_MODEL = None
    HELSINKI_PIPELINE = None
    
    import gc  # noqa: PLC0415
    import torch  # noqa: PLC0415

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("Translation models cleared.")
