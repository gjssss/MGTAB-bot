FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN sed -i '/jammy-backports/d' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

RUN python3 -m pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY api.py predict.py test.py preprocess.py ./
COPY train_adaboost.py build_dataset.py download.py ./
COPY utils/ ./utils/

# Copy models and checkpoints (for offline usage)
COPY models/ ./models/
COPY checkpoints/ ./checkpoints/

# Copy entrypoint
COPY docker/entrypoint.sh /usr/local/bin/mgtab2-entrypoint
RUN chmod +x /usr/local/bin/mgtab2-entrypoint

# Create Dataset directory for volume mount
RUN mkdir -p /app/Dataset

EXPOSE 30102

ENTRYPOINT ["/usr/local/bin/mgtab2-entrypoint"]
