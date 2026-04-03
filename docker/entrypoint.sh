#!/bin/sh
set -eu

check_file() {
  if [ ! -f "$1" ]; then
    echo "Missing critical file: $1" >&2
    exit 1
  fi
}

check_dir() {
  if [ ! -d "$1" ]; then
    echo "Missing critical directory: $1" >&2
    exit 1
  fi
}

# Verify critical files exist
check_file /app/api.py
check_file /app/predict.py
check_file /app/preprocess.py
check_dir /app/models
check_dir /app/checkpoints

# Configuration via environment variables
export BOT_PORT="${BOT_PORT:-30102}"
export BOT_DEVICE="${BOT_DEVICE:-cuda}"

# Validate device
if [ "$BOT_DEVICE" = "cuda" ] && ! python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "Warning: CUDA requested but not available, falling back to CPU"
    export BOT_DEVICE="cpu"
fi

echo "Starting Bot Detection API on port $BOT_PORT with device $BOT_DEVICE"

cd /app
exec /app/.venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port "$BOT_PORT"
