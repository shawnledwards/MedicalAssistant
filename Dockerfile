# ── Build args ────────────────────────────────────────────────────
# For local GPU (RTX 3070):  docker build --build-arg GPU=1 .
# For HF Spaces / CPU-only:  docker build .  (default)
ARG GPU=0

FROM python:3.11-slim AS base

# System deps for PDF parsing and (CPU) OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    libgomp1 \
    git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# ── CPU-only path (HF Spaces default) ─────────────────────────────
FROM base AS cpu-build
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir -r requirements.txt

# ── GPU path (local RTX 3070 with CUDA 12) ────────────────────────
FROM base AS gpu-build
ENV CMAKE_ARGS="-DGGML_CUDA=ON"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir -r requirements.txt

# ── Final stage — pick CPU or GPU ─────────────────────────────────
FROM cpu-build AS final-0
FROM gpu-build AS final-1
FROM final-${GPU} AS final

WORKDIR /app
COPY . .

# HF Spaces requires port 7860; local docker-compose can override via PORT env
EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
