import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

QWEN_SIZES: dict[str, dict] = {
    "0.6B": {"repo": "Qwen/Qwen3-Embedding-0.6B", "dir": "Qwen3-Embedding-0.6B", "dim": 1024},
    "4B":   {"repo": "Qwen/Qwen3-Embedding-4B",   "dir": "Qwen3-Embedding-4B",   "dim": 2560},
    "8B":   {"repo": "Qwen/Qwen3-Embedding-8B",   "dir": "Qwen3-Embedding-8B",   "dim": 4096},
}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MAX_LENGTH = 512

_cache: dict[str, tuple[AutoTokenizer, AutoModel, int]] = {}


def _local_path(qwen_size: str) -> Path:
    if qwen_size not in QWEN_SIZES:
        raise ValueError(
            f"Unknown qwen_size {qwen_size!r}, choose from {list(QWEN_SIZES)}"
        )
    return MODELS_DIR / QWEN_SIZES[qwen_size]["dir"]


def load_embedding(
    qwen_size: str,
    device: torch.device,
) -> tuple[AutoTokenizer, AutoModel, int]:
    key = f"{qwen_size}@{device}"
    if key in _cache:
        return _cache[key]

    local = _local_path(qwen_size)
    if not local.exists():
        raise FileNotFoundError(
            f"Qwen model not found at {local}. "
            f"Run: python download.py --qwen_size {qwen_size}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(local), padding_side="left")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(str(local), torch_dtype=dtype)
    model.eval()
    model.to(device)

    dim = int(getattr(model.config, "hidden_size", QWEN_SIZES[qwen_size]["dim"]))
    expected = QWEN_SIZES[qwen_size]["dim"]
    if dim != expected:
        raise RuntimeError(
            f"Qwen {qwen_size} hidden_size {dim} != expected {expected}"
        )

    _cache[key] = (tokenizer, model, dim)
    return tokenizer, model, dim


def _last_token_pool(last_hidden: Tensor, attention_mask: Tensor) -> Tensor:
    # padding_side='left' → last token is at [:, -1] when any pad exists on left.
    left_padded = bool((attention_mask[:, -1] == 1).all())
    if left_padded:
        return last_hidden[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
    return last_hidden[batch_idx, seq_lens]


def _clean(texts: Iterable[str]) -> list[str]:
    return [t for t in texts if isinstance(t, str) and t.strip()]


@torch.no_grad()
def encode_texts_single(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> np.ndarray:
    cleaned = _clean(texts)
    dim = int(model.config.hidden_size)
    if not cleaned:
        return np.zeros(dim, dtype=np.float32)

    vectors = []
    for text in cleaned:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        pooled = _last_token_pool(out.last_hidden_state, inputs["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        vectors.append(pooled[0].float().cpu())

    stacked = torch.stack(vectors, dim=0)
    mean = stacked.mean(dim=0, keepdim=True)
    mean = F.normalize(mean, p=2, dim=1)
    return mean[0].numpy().astype(np.float32)


@torch.no_grad()
def encode_texts_batched(
    users_texts: list[list[str]],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 16,
    max_tweets_per_user: int | None = 200,
    show_progress: bool = True,
) -> np.ndarray:
    dim = int(model.config.hidden_size)
    n_users = len(users_texts)
    result = np.zeros((n_users, dim), dtype=np.float32)

    flat_texts: list[str] = []
    owners: list[int] = []
    for uid, texts in enumerate(users_texts):
        cleaned = _clean(texts)
        if max_tweets_per_user is not None:
            cleaned = cleaned[:max_tweets_per_user]
        for t in cleaned:
            flat_texts.append(t)
            owners.append(uid)

    if not flat_texts:
        return result

    per_user_sums = torch.zeros((n_users, dim), dtype=torch.float32)
    per_user_counts = torch.zeros((n_users,), dtype=torch.int64)

    iterator = range(0, len(flat_texts), batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=(len(flat_texts) + batch_size - 1) // batch_size,
                            desc="encoding")
        except ImportError:
            pass

    for start in iterator:
        end = min(start + batch_size, len(flat_texts))
        batch_texts = flat_texts[start:end]
        batch_owners = owners[start:end]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        pooled = _last_token_pool(out.last_hidden_state, inputs["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1).float().cpu()

        for i, owner in enumerate(batch_owners):
            per_user_sums[owner] += pooled[i]
            per_user_counts[owner] += 1

    for uid in range(n_users):
        c = per_user_counts[uid].item()
        if c == 0:
            continue
        mean = per_user_sums[uid] / c
        mean = F.normalize(mean.unsqueeze(0), p=2, dim=1)[0]
        result[uid] = mean.numpy().astype(np.float32)

    return result
