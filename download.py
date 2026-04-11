"""Download a Qwen3-Embedding model from HuggingFace into ./models/<name>.

Reads HTTPS_PROXY / HTTP_PROXY / ALL_PROXY from the environment by default.
"""
import argparse
import os
import sys
from pathlib import Path

from utils.embedding import QWEN_SIZES, MODELS_DIR


PROXY_ENV_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                  "https_proxy", "http_proxy", "all_proxy")


def detect_proxies() -> dict[str, str]:
    found: dict[str, str] = {}
    for key in PROXY_ENV_KEYS:
        val = os.environ.get(key)
        if val:
            found[key] = val
    return found


def clear_proxies() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Qwen3-Embedding model")
    parser.add_argument("--qwen_size", required=True, choices=list(QWEN_SIZES))
    parser.add_argument("--output", default=None,
                        help="Override output directory (default: models/<name>)")
    parser.add_argument("--no-proxy", action="store_true",
                        help="Clear proxy env vars before download")
    parser.add_argument("--revision", default=None, help="Optional HF revision")
    args = parser.parse_args()

    if args.no_proxy:
        clear_proxies()
        print("[download] cleared proxy env vars")
    else:
        proxies = detect_proxies()
        if proxies:
            print("[download] using proxy from env:")
            for k, v in proxies.items():
                print(f"  {k}={v}")
        else:
            print("[download] no proxy env vars detected")

    spec = QWEN_SIZES[args.qwen_size]
    repo_id = spec["repo"]
    target = Path(args.output) if args.output else (MODELS_DIR / spec["dir"])
    target.mkdir(parents=True, exist_ok=True)

    print(f"[download] repo_id = {repo_id}")
    print(f"[download] target  = {target}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[download] huggingface_hub not installed. pip install huggingface_hub", file=sys.stderr)
        return 2

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            revision=args.revision,
            allow_patterns=[
                "*.json", "*.txt", "*.model", "*.safetensors",
                "*.safetensors.index.json", "tokenizer*", "special_tokens*",
            ],
        )
    except Exception as e:
        print(f"[download] failed: {e}", file=sys.stderr)
        return 1

    required = ["config.json"]
    missing = [f for f in required if not (target / f).exists()]
    if missing:
        print(f"[download] warning: missing files after download: {missing}",
              file=sys.stderr)
        return 1

    print(f"[download] done: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
