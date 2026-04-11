from pathlib import Path

import joblib
import numpy as np
import torch

from preprocess import build_feature_vector
from .embedding import encode_texts_single

_clf_cache: dict[str, object] = {}


def load_classifier(path: str | Path = "checkpoints/adaboost_bot.joblib"):
    key = str(path)
    if key not in _clf_cache:
        _clf_cache[key] = joblib.load(path)
    return _clf_cache[key]


def predict_user(
    user: dict,
    clf,
    tokenizer,
    emb_model,
    device: torch.device,
) -> dict:
    tweet_emb = encode_texts_single(user.get("tweets", []), tokenizer, emb_model, device)
    feature_vec = build_feature_vector(user, tweet_emb).reshape(1, -1)

    expected = getattr(clf, "n_features_in_", None)
    if expected is not None and feature_vec.shape[1] != expected:
        raise RuntimeError(
            f"Feature dim mismatch: classifier expects {expected}, got {feature_vec.shape[1]}. "
            "Qwen model size likely mismatches preprocess_config.json."
        )

    pred = int(clf.predict(feature_vec)[0])
    proba = clf.predict_proba(feature_vec)[0]
    label = "bot" if pred == 1 else "human"
    return {
        "label": label,
        "prediction": pred,
        "confidence": round(float(proba[pred]), 4),
        "probabilities": {
            "human": round(float(proba[0]), 4),
            "bot": round(float(proba[1]), 4),
        },
    }
