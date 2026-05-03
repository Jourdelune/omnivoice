FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    PYTHONUNBUFFERED=1

WORKDIR /

# System libs — libsndfile for audio I/O, ffmpeg for codec support
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends \
        sudo ca-certificates git wget curl bash \
        libsndfile1 ffmpeg build-essential -y && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# uv manages Python 3.12 and all dependencies (no apt PPA needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv python install 3.12

# Install Python dependencies via uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev && uv cache clean

ENV PATH="/.venv/bin:$PATH"

# Bake OmniVoice model weights into the image — eliminates cold-start download
COPY builder/fetch_models.py /fetch_models.py
RUN python /fetch_models.py && rm /fetch_models.py

# Copy handler and test input
COPY src/ .
COPY test_input.json .

CMD ["python", "-u", "/handler.py"]
