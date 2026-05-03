# CUDA 12.4 + cuDNN runtime — required for OmniVoice GPU inference
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/huggingface-cache

WORKDIR /app

# System libs only — Python is managed by uv (no apt PPA needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# uv installs Python 3.12 from python-build-standalone (no PPA, no network issues)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv python install 3.12

# Copy manifests first for layer caching — dependencies only reinstall when
# pyproject.toml or uv.lock change, not when handler.py changes
COPY pyproject.toml uv.lock ./

# Install production deps into an isolated venv
RUN uv sync --frozen --no-dev

# Activate venv for all subsequent RUN / CMD
ENV PATH="/app/.venv/bin:$PATH"

COPY src/handler.py .

CMD ["python", "-u", "handler.py"]
docker run --rm --gpus all -p 8000:8000 jourdelune876/omnivoice-worker:latest python -u handler.py --rp_serve_api --rp_api_host 0.0.0.0