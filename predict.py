import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"

from pathlib import Path

import numpy as np
import joblib

from preprocess import build_feature_vector, EMBEDDING_DIM

_model_cache = None
_tokenizer_cache = {}
_embed_model_cache = {}

EMBEDDING_MODELS = {
    "en": "roberta-base",
    "zh": "chinese-roberta-wwm-ext",
}

# Local models directory
MODELS_DIR = Path(__file__).resolve().parent / "models"


def _load_model(model_path: str = "checkpoints/adaboost_bot.joblib"):
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(model_path)
    return _model_cache


def _load_embedding_model(lang: str):
    if lang not in _embed_model_cache:
        import logging
        import warnings
        import torch
        from transformers import AutoTokenizer, AutoModel

        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*not sharded.*")

        model_name = EMBEDDING_MODELS[lang]
        model_path = MODELS_DIR / model_name
        
        # Use local model if exists, otherwise fallback to HuggingFace
        if model_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModel.from_pretrained(str(model_path), ignore_mismatched_sizes=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        
        model.eval()
        _tokenizer_cache[lang] = tokenizer
        _embed_model_cache[lang] = model
    return _tokenizer_cache[lang], _embed_model_cache[lang]


def encode_tweets(tweets: list[str], lang: str = "en") -> np.ndarray:
    """Encode a list of tweets into a single 768-dim vector.
    
    Each tweet is encoded separately via [CLS] token,
    then averaged to produce the final vector.
    """
    if not tweets:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    import torch

    tokenizer, model = _load_embedding_model(lang)
    embeddings = []

    with torch.no_grad():
        for text in tweets:
            if not text or not text.strip():
                continue
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=512, padding=True,
            )
            output = model(**inputs)
            cls_vec = output.last_hidden_state[:, 0, :].squeeze(0)
            embeddings.append(cls_vec.numpy())

    if not embeddings:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    return np.mean(embeddings, axis=0).astype(np.float32)


def predict_bot(
    user_json: dict,
    lang: str = "en",
    model_path: str = "checkpoints/adaboost_bot.joblib",
) -> dict:
    """Predict whether a user is a bot.

    Args:
        user_json: User profile + tweets data.
        lang: "en" or "zh", selects the embedding model.
        model_path: Path to the trained AdaBoost model.

    Returns:
        {"label": "bot" | "human", "confidence": float}
    """
    if lang not in EMBEDDING_MODELS:
        raise ValueError(f"Unsupported lang '{lang}', choose from {list(EMBEDDING_MODELS.keys())}")

    tweets = user_json.get("tweets", [])
    tweet_embedding = encode_tweets(tweets, lang=lang)

    feature_vec = build_feature_vector(user_json, tweet_embedding)
    feature_vec = feature_vec.reshape(1, -1)

    clf = _load_model(model_path)
    pred = clf.predict(feature_vec)[0]
    proba = clf.predict_proba(feature_vec)[0]

    label = "bot" if pred == 1 else "human"
    confidence = float(proba[pred])

    return {"label": label, "confidence": round(confidence, 4)}


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <input.json> [lang]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    result = predict_bot(data, lang=lang)
    print(json.dumps(result, ensure_ascii=False, indent=2))
