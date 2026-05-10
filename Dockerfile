ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN if [ -f /etc/apt/sources.list ]; then sed -i '/jammy-backports/d' /etc/apt/sources.list; fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

RUN python3 -m venv /opt/uv \
    && /opt/uv/bin/pip install --no-cache-dir uv \
    && ln -sf /opt/uv/bin/uv /usr/local/bin/uv

WORKDIR /app

ARG MODELS_DIR=models
ARG CHECKPOINTS_DIR=checkpoints

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --python 3.12

# Copy application code
COPY api.py predict.py preprocess.py ./
COPY utils/ ./utils/

# Copy models and checkpoints (for offline usage)
COPY ${MODELS_DIR}/ ./models/
COPY ${CHECKPOINTS_DIR}/ ./checkpoints/

# Copy entrypoint
COPY docker/entrypoint.sh /usr/local/bin/mgtab2-entrypoint
RUN chmod +x /usr/local/bin/mgtab2-entrypoint

EXPOSE 30102

ENTRYPOINT ["/usr/local/bin/mgtab2-entrypoint"]
