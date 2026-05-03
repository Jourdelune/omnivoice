# CUDA 12.4 + cuDNN runtime — required for OmniVoice GPU inference
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/huggingface-cache

WORKDIR /app

# libsndfile1 for soundfile, ffmpeg-minimal for audio decoding (mp3 support)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# uv installs Python 3.12 from python-build-standalone (no PPA, no network issues)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv python install 3.12

# Copy manifests first for layer caching — dependencies only reinstall when
# pyproject.toml or uv.lock change, not when handler.py changes
COPY pyproject.toml uv.lock ./

# Install production deps — purge uv cache after to save ~500MB
RUN uv sync --frozen --no-dev && uv cache clean

# Activate venv for all subsequent RUN / CMD
ENV PATH="/app/.venv/bin:$PATH"

COPY src/handler.py .

CMD ["python", "-u", "handler.py"]