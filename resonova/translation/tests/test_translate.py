"""
Unit tests for resonova.translation.translate
===========================================
Run with:  pytest resonova/translation/tests/test_translate.py -v

Smoke tests:
  - translate() returns non-empty string
  - Hindi output contains Devanagari characters (Unicode range check)
  - Typed exceptions raised on bad input — no bare Exception
  - Empty input raises TranslationError
  - Unsupported language code raises ValueError
"""

import pytest
from unittest.mock import MagicMock, patch

from resonova.exceptions import TranslationError

# Devanagari Unicode range for script validation
_DEVANAGARI_START = 0x0900
_DEVANAGARI_END   = 0x097F

SAMPLE_HINDI = "नमस्ते, आप कैसे हैं? मेरा नाम साक्षी है।"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_indictrans2():
    """Mock the entire _load_model call to return a fake (model, tokenizer, processor)."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_processor = MagicMock()

    # Simulate preprocessing
    mock_processor.preprocess_batch.return_value = ["preprocessed text"]
    mock_tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    mock_tokenizer.as_target_tokenizer.return_value.__enter__ = MagicMock(return_value=None)
    mock_tokenizer.as_target_tokenizer.return_value.__exit__ = MagicMock(return_value=False)
    mock_tokenizer.batch_decode.return_value = ["decoded_token_string"]
    mock_processor.postprocess_batch.return_value = [SAMPLE_HINDI]

    # model.generate returns a tensor-like mock
    mock_model.generate.return_value = MagicMock()
    mock_model.parameters.return_value = iter([MagicMock(device="cpu")])

    return mock_model, mock_tokenizer, mock_processor


# ---------------------------------------------------------------------------
# Tests: translate()
# ---------------------------------------------------------------------------

class TestTranslate:

    def test_returns_non_empty_string(self, mock_indictrans2):
        """translate() must return a non-empty string."""
        with patch("resonova.translation.translate._load_model", return_value=mock_indictrans2), \
             patch("torch.no_grad", return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))):
            from resonova.translation.translate import translate
            result = translate("Hello, how are you?", "eng_Latn", "hin_Deva")

        assert isinstance(result, str)
        assert len(result) > 0, "Translation must be non-empty"

    def test_hindi_output_contains_devanagari(self, mock_indictrans2):
        """
        When target_lang is 'hin_Deva', output must contain Devanagari script.
        Validates that output is actually Hindi, not Latin characters.
        """
        with patch("resonova.translation.translate._load_model", return_value=mock_indictrans2), \
             patch("torch.no_grad", return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))):
            from resonova.translation.translate import translate
            result = translate("Hello, how are you?", "eng_Latn", "hin_Deva")

        has_devanagari = any(_DEVANAGARI_START <= ord(c) <= _DEVANAGARI_END for c in result)
        assert has_devanagari, (
            f"Expected Devanagari script in Hindi output, got: '{result}'. "
            "This suggests the model returned Latin text — check language code format."
        )

    def test_empty_input_raises_translation_error(self):
        """translate() must raise TranslationError for empty input — not bare Exception."""
        from resonova.translation.translate import translate
        with pytest.raises(TranslationError, match="empty"):
            translate("", "eng_Latn", "hin_Deva")

    def test_whitespace_only_input_raises_translation_error(self):
        """translate() must raise TranslationError for whitespace-only input."""
        from resonova.translation.translate import translate
        with pytest.raises(TranslationError, match="empty"):
            translate("   \n\t   ", "eng_Latn", "hin_Deva")

    def test_unsupported_source_lang_raises_value_error(self):
        """translate() must raise ValueError for unsupported source language code."""
        from resonova.translation.translate import translate
        with pytest.raises(ValueError, match="Unsupported source_lang"):
            translate("Hello", "INVALID_LANG", "hin_Deva")

    def test_unsupported_target_lang_raises_value_error(self):
        """translate() must raise ValueError for unsupported target language code."""
        from resonova.translation.translate import translate
        with pytest.raises(ValueError, match="Unsupported target_lang"):
            translate("Hello", "eng_Latn", "INVALID_LANG")

    def test_empty_model_output_raises_translation_error(self, mock_indictrans2):
        """translate() must raise TranslationError if the model returns an empty string."""
        mock_model, mock_tokenizer, mock_processor = mock_indictrans2
        mock_processor.postprocess_batch.return_value = [""]  # model returns empty

        with patch("resonova.translation.translate._load_model", return_value=mock_indictrans2), \
             patch("torch.no_grad", return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))):
            from resonova.translation.translate import translate
            with pytest.raises(TranslationError, match="empty translation"):
                translate("Hello world", "eng_Latn", "hin_Deva")

    def test_result_is_stripped(self, mock_indictrans2):
        """translate() must return a stripped string."""
        mock_model, mock_tokenizer, mock_processor = mock_indictrans2
        mock_processor.postprocess_batch.return_value = [f"  {SAMPLE_HINDI}  "]

        with patch("resonova.translation.translate._load_model", return_value=mock_indictrans2), \
             patch("torch.no_grad", return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))):
            from resonova.translation.translate import translate
            result = translate("Hello", "eng_Latn", "hin_Deva")

        assert not result.startswith(" "), "Result must be left-stripped"
        assert not result.endswith(" "), "Result must be right-stripped"


# ---------------------------------------------------------------------------
# Tests: get_supported_languages()
# ---------------------------------------------------------------------------

class TestGetSupportedLanguages:

    def test_returns_set(self):
        from resonova.translation.translate import get_supported_languages
        langs = get_supported_languages()
        assert isinstance(langs, set)

    def test_contains_required_languages(self):
        from resonova.translation.translate import get_supported_languages
        langs = get_supported_languages()
        assert "eng_Latn" in langs, "English must be in supported languages"
        assert "hin_Deva" in langs, "Hindi must be in supported languages"

    def test_returns_copy(self):
        """Modifying the returned set must not affect internal state."""
        from resonova.translation.translate import get_supported_languages, SUPPORTED_LANGS
        langs = get_supported_languages()
        langs.add("FAKE_LANG")
        assert "FAKE_LANG" not in SUPPORTED_LANGS, "Internal set was mutated by caller"
