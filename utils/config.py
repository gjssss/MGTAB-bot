import json
from pathlib import Path

CONFIG_PATH = Path("checkpoints/preprocess_config.json")


def load_preprocess_config(path: str | Path = CONFIG_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Preprocess config not found at {p}. "
            "Run train_adaboost.py first to generate it."
        )
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = {"qwen_size", "embedding_dim", "feature_dim", "property_dim"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"preprocess_config missing keys: {missing}")
    return cfg


def save_preprocess_config(cfg: dict, path: str | Path = CONFIG_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
