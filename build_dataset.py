"""Build features.pt + labels_bot.pt from a cleaned JSONL using Qwen3-Embedding."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from preprocess import extract_properties, NORMALIZATION_CONFIG, BOOL_FIELDS, PROPERTY_DIM
from utils.device import resolve_device
from utils.embedding import QWEN_SIZES, load_embedding, encode_texts_batched
from utils.jsonl import iter_jsonl_records


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description="Build bot detection dataset")
    p.add_argument("--jsonl", required=True, help="Input cleaned JSONL path")
    p.add_argument("--name", required=True, help="Output dataset name (Dataset/<name>)")
    p.add_argument("--qwen_size", required=True, choices=list(QWEN_SIZES))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_tweets", type=int, default=200)
    p.add_argument("--limit", type=int, default=None,
                   help="Only use the first N records (smoke test)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dry_run", action="store_true",
                   help="Process only the first 10 records and skip writing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    device = resolve_device(args.device)
    print(f"[build] device = {device}")

    if device.type == "cpu" and args.limit is None and not args.dry_run:
        print("[build] WARNING: running on CPU over the full dataset will be very slow. "
              "Pass --limit N for smoke tests.")

    tokenizer, model, dim = load_embedding(args.qwen_size, device)
    print(f"[build] qwen_size={args.qwen_size} dim={dim}")

    records: list[dict] = []
    for _, rec in iter_jsonl_records(jsonl_path):
        records.append(rec)
        if args.dry_run and len(records) >= 10:
            break
        if args.limit is not None and len(records) >= args.limit:
            break

    n = len(records)
    if n == 0:
        print("[build] no records, aborting")
        return 1
    print(f"[build] loaded {n} records")

    for r in records:
        if "label" not in r:
            raise ValueError("record missing 'label' field")

    props = np.stack([extract_properties(r) for r in records])  # (n, 20)
    users_tweets = [
        [t for t in r.get("tweets", []) if isinstance(t, str) and t.strip()]
        for r in records
    ]
    tweet_mat = encode_texts_batched(
        users_tweets,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_tweets_per_user=args.max_tweets,
        show_progress=True,
    )

    assert tweet_mat.shape == (n, dim), (tweet_mat.shape, (n, dim))
    features = np.concatenate([props, tweet_mat], axis=1).astype(np.float32)
    labels = np.array([int(bool(r["label"])) for r in records], dtype=np.int64)

    n_bot = int((labels == 1).sum())
    n_human = int((labels == 0).sum())
    print(f"[build] features shape={features.shape} dtype={features.dtype}")
    print(f"[build] label distribution: human={n_human} bot={n_bot}")

    if args.dry_run:
        print("[build] dry_run: skipping write")
        print(f"[build] first row preview (props): {features[0, :PROPERTY_DIM]}")
        print(f"[build] first row preview (emb head): {features[0, PROPERTY_DIM:PROPERTY_DIM+8]}")
        return 0

    out_dir = Path("Dataset") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(torch.from_numpy(features), out_dir / "features.pt")
    torch.save(torch.from_numpy(labels), out_dir / "labels_bot.pt")

    info = {
        "source_jsonl": str(jsonl_path),
        "source_sha256": sha256_file(jsonl_path),
        "n_samples": n,
        "property_dim": PROPERTY_DIM,
        "embedding_dim": int(dim),
        "feature_dim": int(PROPERTY_DIM + dim),
        "qwen_size": args.qwen_size,
        "qwen_repo": QWEN_SIZES[args.qwen_size]["repo"],
        "max_tweets_per_user": args.max_tweets,
        "batch_size": args.batch_size,
        "device": str(device),
        "limit": args.limit,
        "label_distribution": {"human": n_human, "bot": n_bot},
        "build_timestamp": datetime.utcnow().isoformat() + "Z",
        "normalization": NORMALIZATION_CONFIG,
        "bool_fields": BOOL_FIELDS,
    }
    with (out_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[build] saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
