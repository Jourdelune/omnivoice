import base64
import io
import os
import tempfile

import runpod
import soundfile as sf
import torch
from omnivoice import OmniVoice

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=DEVICE,
    dtype=DTYPE,
)

# Optional generation params forwarded as-is to model.generate()
_OPTIONAL_PARAMS = ("num_step", "speed", "duration", "language_id")


def _decode_ref_audio(ref_audio_b64: str, fmt: str) -> str:
    """Decode base64 ref audio to a temp file and return its path."""
    try:
        audio_bytes = base64.b64decode(ref_audio_b64)
    except Exception:
        raise ValueError("Invalid base64 encoding for 'ref_audio'")

    tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


def handler(job):
    job_input = job["input"]

    # ── required ──────────────────────────────────────────────────────────────
    text = job_input.get("text")
    if not text:
        return {"error": "Missing required field: 'text'"}

    # ── mode detection ────────────────────────────────────────────────────────
    # Voice cloning  → ref_audio + ref_text required
    # Voice design   → instruct present, no ref_audio
    # Auto voice     → neither ref_audio nor instruct
    ref_audio_b64 = job_input.get("ref_audio")
    ref_text = job_input.get("ref_text")
    instruct = job_input.get("instruct")

    # ── build generate kwargs ─────────────────────────────────────────────────
    generate_kwargs: dict = {"text": text}

    tmp_path = None
    try:
        if ref_audio_b64:
            # Voice cloning — ref_text is required to avoid loading Whisper
            if not ref_text:
                return {"error": "Missing required field: 'ref_text' (required when 'ref_audio' is provided)"}

            ref_audio_format = job_input.get("ref_audio_format", "wav")
            try:
                tmp_path = _decode_ref_audio(ref_audio_b64, ref_audio_format)
            except ValueError as exc:
                return {"error": str(exc)}

            generate_kwargs["ref_audio"] = tmp_path
            generate_kwargs["ref_text"] = ref_text

        elif instruct:
            # Voice design
            generate_kwargs["instruct"] = instruct

        # Auto voice: no extra kwargs needed

        # ── optional generation parameters ───────────────────────────────────
        for param in _OPTIONAL_PARAMS:
            value = job_input.get(param)
            if value is not None:
                generate_kwargs[param] = value

        audio_arrays = model.generate(**generate_kwargs)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # ── encode output ─────────────────────────────────────────────────────────
    output_format = job_input.get("output_format", "wav")
    sample_rate = 24000

    buf = io.BytesIO()
    sf.write(buf, audio_arrays[0], sample_rate, format=output_format)
    buf.seek(0)
    audio_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "audio": audio_b64,
        "sample_rate": sample_rate,
        "format": output_format,
    }


runpod.serverless.start({"handler": handler})
