"""
Integration tests for the OmniVoice RunPod handler.

Starts the real handler with --rp_serve_api and sends HTTP requests to
/runsync, exactly as a RunPod worker behaves in production.

Requirements: omnivoice installed, model weights in HF_HOME.
Run:
    python -m pytest tests/test_handler.py -v -s
"""

import base64
import io
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
import requests
import soundfile as sf

# ── config ─────────────────────────────────────────────────────────────────────

HOST = "localhost"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"
HANDLER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "handler.py"))
REF_AUDIO = os.path.join(os.path.dirname(__file__), "intervenant.mp3")

# Model loading can take a while on first run
SERVER_READY_TIMEOUT = 180  # seconds


# ── helpers ────────────────────────────────────────────────────────────────────

def _wait_for_port(timeout: int = SERVER_READY_TIMEOUT) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((HOST, PORT), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise RuntimeError(f"Handler server did not start within {timeout}s")


def _runsync(payload: dict, timeout: int = 300) -> dict:
    resp = requests.post(f"{BASE_URL}/runsync", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _audio_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _b64_to_ndarray(audio_b64: str) -> tuple[np.ndarray, int]:
    buf = io.BytesIO(base64.b64decode(audio_b64))
    data, sr = sf.read(buf)
    return data, sr


# ── server fixture ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def server():
    """Spawn the handler server once for the entire test session."""
    proc = subprocess.Popen(
        [sys.executable, HANDLER, "--rp_serve_api", "--rp_log_level", "INFO"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        _wait_for_port()
    except RuntimeError:
        proc.terminate()
        pytest.fail("Handler server failed to start — check logs above")

    yield

    proc.terminate()
    proc.wait()


@pytest.fixture(scope="module")
def ref_audio_b64() -> str:
    return _audio_to_b64(REF_AUDIO)


# ── helpers shared across tests ────────────────────────────────────────────────

def _extract_error(result: dict) -> str | None:
    """
    Extract an error message from a RunPod /runsync response.

    RunPod sets status=FAILED (no 'output' key) in two cases:
      - handler returns {"error": "..."}  → result["error"] is the plain string
      - handler raises an exception       → result["error"] is a JSON string with
                                            an "error_message" field
    """
    if result.get("status") != "FAILED":
        return None
    raw = result.get("error", "")
    try:
        return json.loads(raw).get("error_message", raw)
    except (json.JSONDecodeError, AttributeError):
        return raw


def assert_valid_audio_output(result: dict) -> None:
    """Assert the /runsync response contains valid audio output."""
    assert result.get("status") == "COMPLETED", f"Job failed: {result}"
    output = result["output"]
    assert "error" not in output, f"Handler returned error: {output['error']}"
    assert "audio" in output
    assert output["sample_rate"] == 24000
    assert output["format"] in ("wav", "flac", "ogg")

    audio, sr = _b64_to_ndarray(output["audio"])
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0


# ── validation ─────────────────────────────────────────────────────────────────

class TestValidation:
    def test_missing_text_returns_error(self):
        result = _runsync({"input": {}})
        assert result.get("status") == "FAILED"
        assert "text" in (_extract_error(result) or "")

    def test_invalid_base64_ref_audio(self):
        result = _runsync({"input": {
            "text": "Hello.",
            "ref_audio": "!!!invalid_base64!!!",
        }})
        assert result.get("status") == "FAILED"
        assert "base64" in (_extract_error(result) or "").lower()


# ── voice cloning ──────────────────────────────────────────────────────────────

class TestVoiceCloning:
    def test_cloning_with_ref_text(self, ref_audio_b64):
        result = _runsync({"input": {
            "text": "Bonjour, ceci est un test de clonage vocal.",
            "ref_audio": ref_audio_b64,
            "ref_audio_format": "mp3",
            "ref_text": "Transcription de l'audio de référence.",
        }})
        assert_valid_audio_output(result)

    def test_cloning_without_ref_text_returns_error(self, ref_audio_b64):
        """ref_text is required when ref_audio is provided."""
        result = _runsync({"input": {
            "text": "Hello world, this is a cloning test.",
            "ref_audio": ref_audio_b64,
            "ref_audio_format": "mp3",
        }})
        assert result.get("status") == "FAILED"
        assert "ref_text" in (_extract_error(result) or "")

    def test_cloning_with_language_id(self, ref_audio_b64):
        result = _runsync({"input": {
            "text": "Bonjour tout le monde.",
            "ref_audio": ref_audio_b64,
            "ref_audio_format": "mp3",
            "language_id": "fr",
        }})
        assert_valid_audio_output(result)

    def test_cloning_output_sample_rate_is_24k(self, ref_audio_b64):
        result = _runsync({"input": {
            "text": "Sample rate check.",
            "ref_audio": ref_audio_b64,
            "ref_audio_format": "mp3",
        }})
        assert result["output"]["sample_rate"] == 24000


# ── voice design ───────────────────────────────────────────────────────────────

class TestVoiceDesign:
    @pytest.mark.parametrize("instruct", [
        "female, low pitch, british accent",
        "male, high pitch, american accent",
        "female, whisper",
    ])
    def test_design_instruct_variants(self, instruct):
        result = _runsync({"input": {
            "text": "This is a voice design test.",
            "instruct": instruct,
        }})
        assert_valid_audio_output(result)

    def test_design_does_not_require_ref_audio(self):
        result = _runsync({"input": {
            "text": "No reference audio needed.",
            "instruct": "male, low pitch",
        }})
        assert_valid_audio_output(result)


# ── auto voice ─────────────────────────────────────────────────────────────────

class TestAutoVoice:
    def test_auto_voice_no_prompt(self):
        result = _runsync({"input": {
            "text": "Auto voice, no reference, no instruct.",
        }})
        assert_valid_audio_output(result)


# ── generation parameters ──────────────────────────────────────────────────────

class TestGenerationParams:
    def test_faster_inference_with_16_steps(self, ref_audio_b64):
        result = _runsync({"input": {
            "text": "Faster inference with fewer diffusion steps.",
            "ref_audio": ref_audio_b64,
            "ref_audio_format": "mp3",
            "num_step": 16,
        }})
        assert_valid_audio_output(result)

    def test_speed_factor(self, ref_audio_b64):
        result = _runsync({"input": {
            "text": "Speaking faster than usual.",
            "ref_audio": ref_audio_b64,
            "ref_audio_format": "mp3",
            "speed": 1.5,
        }})
        assert_valid_audio_output(result)

    def test_fixed_duration(self, ref_audio_b64):
        result = _runsync({"input": {
            "text": "Fixed duration output.",
            "ref_audio": ref_audio_b64,
            "ref_audio_format": "mp3",
            "duration": 5.0,
        }})
        assert_valid_audio_output(result)

    def test_all_params_combined(self, ref_audio_b64):
        result = _runsync({"input": {
            "text": "All generation parameters active.",
            "ref_audio": ref_audio_b64,
            "ref_audio_format": "mp3",
            "num_step": 16,
            "speed": 0.9,
            "language_id": "en",
        }})
        assert_valid_audio_output(result)


# ── non-verbal & pronunciation ─────────────────────────────────────────────────

class TestNonVerbalAndPronunciation:
    @pytest.mark.parametrize("tag", [
        "[laughter] You really got me!",
        "[sigh] Long day.",
        "[laughter] But we made it. [sigh]",
    ])
    def test_nonverbal_tags(self, tag):
        result = _runsync({"input": {"text": tag}})
        assert_valid_audio_output(result)

    def test_pronunciation_pinyin(self):
        result = _runsync({"input": {
            "text": "这批货物打ZHE2出售后他严重SHE2本了，再也经不起ZHE1腾了。",
        }})
        assert_valid_audio_output(result)

    def test_pronunciation_cmu(self):
        result = _runsync({"input": {
            "text": "He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.",
        }})
        assert_valid_audio_output(result)


# ── output encoding ────────────────────────────────────────────────────────────

class TestOutputEncoding:
    def test_output_is_decodable_wav(self):
        result = _runsync({"input": {"text": "Encoding validation."}})
        assert_valid_audio_output(result)
        audio, sr = _b64_to_ndarray(result["output"]["audio"])
        assert sr == 24000
        assert audio.dtype in (np.float32, np.float64, np.int16, np.int32)

    def test_output_format_field_matches(self):
        result = _runsync({"input": {
            "text": "Format check.",
            "output_format": "wav",
        }})
        assert result["output"]["format"] == "wav"
