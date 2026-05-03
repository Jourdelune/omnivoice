"""
Tests for the OmniVoice RunPod handler.

Run with:
    python -m pytest tests/test_handler.py -v

The tests mock model.generate() so no GPU / model download is required.
"""

import base64
import io
import os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

# ── helpers ────────────────────────────────────────────────────────────────────

REF_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "intervenant.mp3")
SAMPLE_RATE = 24000
FAKE_AUDIO = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second of silence


def _audio_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _b64_to_array(audio_b64: str) -> np.ndarray:
    buf = io.BytesIO(base64.b64decode(audio_b64))
    data, _ = sf.read(buf)
    return data


def _make_job(input_dict: dict) -> dict:
    return {"id": "test-job", "input": input_dict}


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ref_audio_b64():
    return _audio_to_b64(REF_AUDIO_PATH)


@pytest.fixture(scope="module")
def handler_fn():
    """
    Import the handler module with OmniVoice patched out so no model is loaded.
    Returns the bare handler() function.
    """
    fake_audio_result = [FAKE_AUDIO]

    mock_model = MagicMock()
    mock_model.generate.return_value = fake_audio_result

    mock_omnivoice_cls = MagicMock()
    mock_omnivoice_cls.from_pretrained.return_value = mock_model

    # Patch omnivoice and runpod before importing handler
    omnivoice_mod = types.ModuleType("omnivoice")
    omnivoice_mod.OmniVoice = mock_omnivoice_cls

    runpod_mod = types.ModuleType("runpod")
    runpod_serverless = types.ModuleType("runpod.serverless")
    runpod_serverless.start = MagicMock()
    runpod_mod.serverless = runpod_serverless

    with patch.dict(
        sys.modules,
        {"omnivoice": omnivoice_mod, "runpod": runpod_mod},
    ):
        # Force fresh import
        if "handler" in sys.modules:
            del sys.modules["handler"]

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import handler as handler_module

        yield handler_module.handler, mock_model


# ── validation tests ──────────────────────────────────────────────────────────

class TestValidation:
    def test_missing_text(self, handler_fn):
        fn, _ = handler_fn
        result = fn(_make_job({}))
        assert "error" in result
        assert "text" in result["error"]

    def test_invalid_base64_ref_audio(self, handler_fn):
        fn, _ = handler_fn
        result = fn(_make_job({
            "text": "Hello",
            "ref_audio": "!!not_valid_base64!!",
        }))
        assert "error" in result
        assert "base64" in result["error"].lower()


# ── voice cloning ─────────────────────────────────────────────────────────────

class TestVoiceCloning:
    def test_with_ref_text(self, handler_fn, ref_audio_b64):
        fn, mock_model = handler_fn
        result = fn(_make_job({
            "text": "Bonjour, ceci est un test de clonage vocal.",
            "ref_audio": ref_audio_b64,
            "ref_audio_format": "mp3",
            "ref_text": "Transcription de l'audio de référence.",
        }))

        assert "error" not in result
        assert result["sample_rate"] == 24000
        assert result["format"] == "wav"
        audio = _b64_to_array(result["audio"])
        assert len(audio) > 0

        call_kwargs = mock_model.generate.call_args.kwargs
        assert call_kwargs["text"] == "Bonjour, ceci est un test de clonage vocal."
        assert "ref_audio" in call_kwargs
        assert call_kwargs["ref_text"] == "Transcription de l'audio de référence."

    def test_without_ref_text_whisper_fallback(self, handler_fn, ref_audio_b64):
        """ref_text omitted → OmniVoice auto-transcribes via Whisper."""
        fn, mock_model = handler_fn
        result = fn(_make_job({
            "text": "Hello world.",
            "ref_audio": ref_audio_b64,
            "ref_audio_format": "mp3",
        }))

        assert "error" not in result
        call_kwargs = mock_model.generate.call_args.kwargs
        assert "ref_audio" in call_kwargs
        assert "ref_text" not in call_kwargs

    def test_output_format_wav(self, handler_fn, ref_audio_b64):
        fn, _ = handler_fn
        result = fn(_make_job({
            "text": "Test.",
            "ref_audio": ref_audio_b64,
            "ref_audio_format": "mp3",
            "output_format": "wav",
        }))
        assert result["format"] == "wav"
        assert "audio" in result


# ── voice design ──────────────────────────────────────────────────────────────

class TestVoiceDesign:
    def test_basic_instruct(self, handler_fn):
        fn, mock_model = handler_fn
        result = fn(_make_job({
            "text": "This is a voice design test.",
            "instruct": "female, low pitch, british accent",
        }))

        assert "error" not in result
        call_kwargs = mock_model.generate.call_args.kwargs
        assert call_kwargs["instruct"] == "female, low pitch, british accent"
        assert "ref_audio" not in call_kwargs

    def test_instruct_variants(self, handler_fn):
        fn, mock_model = handler_fn
        for instruct in [
            "male, high pitch, american accent",
            "female, whisper",
            "child, chinese dialect",
        ]:
            result = fn(_make_job({"text": "Test.", "instruct": instruct}))
            assert "error" not in result
            assert mock_model.generate.call_args.kwargs["instruct"] == instruct


# ── auto voice ────────────────────────────────────────────────────────────────

class TestAutoVoice:
    def test_no_ref_no_instruct(self, handler_fn):
        fn, mock_model = handler_fn
        result = fn(_make_job({"text": "Auto voice test."}))

        assert "error" not in result
        call_kwargs = mock_model.generate.call_args.kwargs
        assert "ref_audio" not in call_kwargs
        assert "instruct" not in call_kwargs


# ── generation parameters ─────────────────────────────────────────────────────

class TestGenerationParams:
    def test_num_step(self, handler_fn):
        fn, mock_model = handler_fn
        fn(_make_job({"text": "Test.", "num_step": 16}))
        assert mock_model.generate.call_args.kwargs["num_step"] == 16

    def test_speed(self, handler_fn):
        fn, mock_model = handler_fn
        fn(_make_job({"text": "Test.", "speed": 1.5}))
        assert mock_model.generate.call_args.kwargs["speed"] == 1.5

    def test_duration(self, handler_fn):
        fn, mock_model = handler_fn
        fn(_make_job({"text": "Test.", "duration": 8.0}))
        assert mock_model.generate.call_args.kwargs["duration"] == 8.0

    def test_language_id(self, handler_fn):
        fn, mock_model = handler_fn
        fn(_make_job({"text": "Test.", "language_id": "fr"}))
        assert mock_model.generate.call_args.kwargs["language_id"] == "fr"

    def test_all_params_combined(self, handler_fn, ref_audio_b64):
        fn, mock_model = handler_fn
        fn(_make_job({
            "text": "Test.",
            "ref_audio": ref_audio_b64,
            "ref_audio_format": "mp3",
            "num_step": 16,
            "speed": 0.9,
            "language_id": "en",
        }))
        kw = mock_model.generate.call_args.kwargs
        assert kw["num_step"] == 16
        assert kw["speed"] == 0.9
        assert kw["language_id"] == "en"


# ── non-verbal & pronunciation ─────────────────────────────────────────────────

class TestNonVerbalAndPronunciation:
    def test_laughter_tag(self, handler_fn):
        fn, mock_model = handler_fn
        text = "[laughter] You really got me!"
        result = fn(_make_job({"text": text}))
        assert "error" not in result
        assert mock_model.generate.call_args.kwargs["text"] == text

    def test_multiple_nonverbal_tags(self, handler_fn):
        fn, mock_model = handler_fn
        text = "[sigh] Long day. [laughter] But we made it."
        result = fn(_make_job({"text": text}))
        assert "error" not in result

    def test_pronunciation_pinyin(self, handler_fn):
        fn, mock_model = handler_fn
        text = "这批货物打ZHE2出售后他严重SHE2本了。"
        result = fn(_make_job({"text": text}))
        assert "error" not in result
        assert mock_model.generate.call_args.kwargs["text"] == text

    def test_pronunciation_cmu(self, handler_fn):
        fn, mock_model = handler_fn
        text = "He plays the [B EY1 S] guitar while catching a [B AE1 S] fish."
        result = fn(_make_job({"text": text}))
        assert "error" not in result
        assert mock_model.generate.call_args.kwargs["text"] == text


# ── output encoding ────────────────────────────────────────────────────────────

class TestOutputEncoding:
    def test_output_is_valid_base64_wav(self, handler_fn):
        fn, _ = handler_fn
        result = fn(_make_job({"text": "Encoding test."}))
        assert "error" not in result
        decoded = _b64_to_array(result["audio"])
        assert isinstance(decoded, np.ndarray)

    def test_output_metadata(self, handler_fn):
        fn, _ = handler_fn
        result = fn(_make_job({"text": "Metadata test."}))
        assert result["sample_rate"] == 24000
        assert result["format"] == "wav"
