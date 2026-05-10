from pathlib import Path


QWEN_SIZES: dict[str, dict] = {
    "0.6B": {"repo": "Qwen/Qwen3-Embedding-0.6B", "dir": "Qwen3-Embedding-0.6B", "dim": 1024},
    "4B":   {"repo": "Qwen/Qwen3-Embedding-4B",   "dir": "Qwen3-Embedding-4B",   "dim": 2560},
    "8B":   {"repo": "Qwen/Qwen3-Embedding-8B",   "dir": "Qwen3-Embedding-8B",   "dim": 4096},
}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
