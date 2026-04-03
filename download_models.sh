#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$ROOT_DIR/models"
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  ./download_models.sh [all|en|zh] [--force]

Examples:
  ./download_models.sh
  ./download_models.sh all
  ./download_models.sh en
  ./download_models.sh zh --force
EOF
}

log() {
  printf '%s\n' "$*"
}

download_file() {
  local url="$1"
  local output="$2"
  local required="${3:-1}"

  if [[ -f "$output" && "$FORCE" -ne 1 ]]; then
    log "skip $(basename "$output")"
    return
  fi

  local tmp_file="${output}.part"
  if curl -fL --retry 3 --connect-timeout 15 -o "$tmp_file" "$url"; then
    mv "$tmp_file" "$output"
    log "saved $(basename "$output")"
    return
  fi

  rm -f "$tmp_file"

  if [[ "$required" -eq 1 ]]; then
    log "failed $url"
    exit 1
  fi

  log "skip optional $(basename "$output")"
}

download_model_en() {
  local dir="$MODELS_DIR/roberta-base"
  mkdir -p "$dir"
  log "downloading en -> $dir"

  download_file "https://huggingface.co/roberta-base/resolve/main/config.json" "$dir/config.json"
  download_file "https://huggingface.co/roberta-base/resolve/main/model.safetensors" "$dir/model.safetensors"
  download_file "https://huggingface.co/roberta-base/resolve/main/tokenizer.json" "$dir/tokenizer.json"
  download_file "https://huggingface.co/roberta-base/resolve/main/tokenizer_config.json" "$dir/tokenizer_config.json"
  download_file "https://huggingface.co/roberta-base/resolve/main/special_tokens_map.json" "$dir/special_tokens_map.json" 0
  download_file "https://huggingface.co/roberta-base/resolve/main/vocab.json" "$dir/vocab.json" 0
  download_file "https://huggingface.co/roberta-base/resolve/main/merges.txt" "$dir/merges.txt" 0
}

download_model_zh() {
  local dir="$MODELS_DIR/chinese-roberta-wwm-ext"
  mkdir -p "$dir"
  log "downloading zh -> $dir"

  download_file "https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/config.json" "$dir/config.json"
  download_file "https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/model.safetensors" "$dir/model.safetensors"
  download_file "https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/tokenizer.json" "$dir/tokenizer.json"
  download_file "https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/tokenizer_config.json" "$dir/tokenizer_config.json"
  download_file "https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/special_tokens_map.json" "$dir/special_tokens_map.json" 0
  download_file "https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/vocab.txt" "$dir/vocab.txt" 0
}

TARGETS=()
for arg in "$@"; do
  case "$arg" in
    all|en|zh)
      TARGETS+=("$arg")
      ;;
    --force)
      FORCE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  TARGETS=("all")
fi

mkdir -p "$MODELS_DIR"

for target in "${TARGETS[@]}"; do
  case "$target" in
    all)
      download_model_en
      download_model_zh
      ;;
    en)
      download_model_en
      ;;
    zh)
      download_model_zh
      ;;
  esac
done
