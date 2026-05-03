# OmniVoice RunPod Worker

[![RunPod](https://api.runpod.io/badge/Jourdelune/omnivoice)](https://www.runpod.io/console/hub/Jourdelune/omnivoice)

RunPod Serverless worker for [OmniVoice](https://github.com/k2-fsa/OmniVoice) — a zero-shot multilingual TTS model supporting 600+ languages, voice cloning, and voice design.

## API

All requests are POST to `/runsync` with a JSON body `{ "input": { ... } }`.  
The response always contains `status` (`COMPLETED` or `FAILED`) and an `output` or `error` field.

### Response format

```json
{
  "status": "COMPLETED",
  "output": {
    "audio": "<base64-encoded audio>",
    "sample_rate": 24000,
    "format": "wav"
  }
}
```

---

## Modes

### 1. Voice Cloning

Clone a voice from a short reference audio (3–10 seconds recommended).

**Required fields**

| Field | Type | Description |
|---|---|---|
| `text` | string | Text to synthesize |
| `ref_audio` | string | Base64-encoded reference audio |
| `ref_audio_format` | string | Format of `ref_audio` — `"mp3"`, `"wav"`, etc. (default: `"wav"`) |

**Optional fields**

| Field | Type | Description |
|---|---|---|
| `ref_text` | string | Transcription of the reference audio. If omitted, Whisper auto-transcribes it. |

**Example**

```json
{
  "input": {
    "text": "Bonjour, ceci est un test de clonage vocal.",
    "ref_audio": "<base64>",
    "ref_audio_format": "mp3",
    "ref_text": "Transcription de l'audio de référence."
  }
}
```

---

### 2. Voice Design

Generate speech with controlled speaker attributes — no reference audio needed.

**Required fields**

| Field | Type | Description |
|---|---|---|
| `text` | string | Text to synthesize |
| `instruct` | string | Comma-separated speaker attributes (see list below) |

**Valid English attributes**

| Category | Values |
|---|---|
| Gender | `male`, `female` |
| Age | `child`, `teenager`, `young adult`, `middle-aged`, `elderly` |
| Pitch | `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch` |
| Style | `whisper` |
| Accent | `american accent`, `british accent`, `australian accent`, `canadian accent`, `indian accent`, `chinese accent`, `japanese accent`, `korean accent`, `portuguese accent`, `russian accent` |

> Use only English **or** only Chinese attributes. Do not mix.

**Example**

```json
{
  "input": {
    "text": "This is a voice design test.",
    "instruct": "female, low pitch, british accent"
  }
}
```

---

### 3. Auto Voice

Let the model pick a voice automatically.

```json
{
  "input": {
    "text": "Hello, this is an auto voice test."
  }
}
```

---

## Optional generation parameters

These parameters apply to all three modes.

| Field | Type | Default | Description |
|---|---|---|---|
| `num_step` | int | `32` | Diffusion steps. Use `16` for faster inference. |
| `speed` | float | `1.0` | Speaking rate. `> 1.0` faster, `< 1.0` slower. |
| `duration` | float | — | Fixed output duration in seconds. Overrides `speed` if both are set. |
| `language_id` | string | — | Target language code (e.g. `"fr"`, `"en"`, `"zh"`). |
| `output_format` | string | `"wav"` | Output audio format: `"wav"`, `"flac"`, `"ogg"`. |

**Example**

```json
{
  "input": {
    "text": "Faster inference with fewer steps.",
    "ref_audio": "<base64>",
    "ref_audio_format": "mp3",
    "num_step": 16,
    "speed": 1.2,
    "language_id": "en"
  }
}
```

---

## Fine-grained control

### Non-verbal symbols

Insert tags directly in `text` to add expressive sounds.

| Tag | Effect |
|---|---|
| `[laughter]` | Laughter |
| `[sigh]` | Sigh |
| `[confirmation-en]` | "Uh-huh" |
| `[question-en]` | Rising intonation |
| `[surprise-ah]` | Surprise |
| `[dissatisfaction-hnn]` | Dissatisfaction |

```json
{
  "input": {
    "text": "[laughter] You really got me! I didn't see that coming. [sigh]"
  }
}
```

### Pronunciation correction

**Chinese (pinyin + tone number)**

```json
{
  "input": {
    "text": "这批货物打ZHE2出售后他严重SHE2本了，再也经不起ZHE1腾了。"
  }
}
```

**English (CMU phoneme dictionary)**

```json
{
  "input": {
    "text": "He plays the [B EY1 S] guitar while catching a [B AE1 S] fish."
  }
}
```

---

## Error handling

When `status` is `FAILED`, the error message is in the `error` field.

```json
{
  "status": "FAILED",
  "error": "Missing required field: 'text'"
}
```

Common errors:

| Error | Cause |
|---|---|
| `Missing required field: 'text'` | `text` not provided |
| `Invalid base64 encoding for 'ref_audio'` | `ref_audio` is not valid base64 |
| `Unsupported instruct items found in ...` | `instruct` contains invalid attributes |

---

## Decode the output audio

```python
import base64, json

with open("output.json") as f:
    result = json.load(f)

audio_bytes = base64.b64decode(result["output"]["audio"])
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

Or use the provided helper:

```bash
python save_output.py output.json
```

---

## Local testing

```bash
# Start the handler server
python src/handler.py --rp_serve_api

# Run the test suite
python -m pytest tests/test_handler.py -v
```

## Docker

```bash
docker build -t youruser/omnivoice-worker:latest .
docker push youruser/omnivoice-worker:latest

# Run locally
docker run --rm --gpus all -p 8000:8000 youruser/omnivoice-worker:latest \
  python -u handler.py --rp_serve_api --rp_api_host 0.0.0.0
```

On RunPod, set **Model** to `k2-fsa/OmniVoice` in the endpoint settings to enable model caching.

[![Runpod](https://api.runpod.io/badge/Jourdelune/omnivoice)](https://console.runpod.io/hub/Jourdelune/omnivoice)