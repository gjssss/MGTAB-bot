"""Thin CLI wrapper: predict a single user JSON using the trained AdaBoost + Qwen embedding."""
import os
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from utils.config import load_preprocess_config
from utils.device import resolve_device
from utils.embedding import load_embedding
from utils.inference import load_classifier, predict_user


def predict_bot(
    user_json: dict,
    model_path: str = "checkpoints/adaboost_bot.joblib",
) -> dict:
    cfg = load_preprocess_config()
    device = resolve_device("auto")
    clf = load_classifier(model_path)
    tokenizer, emb_model, _ = load_embedding(cfg["qwen_size"], device)
    return predict_user(user_json, clf, tokenizer, emb_model, device)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <input.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    result = predict_bot(data)
    print(json.dumps(result, ensure_ascii=False, indent=2))
