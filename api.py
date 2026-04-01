import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import logging
import warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*not sharded.*")

import numpy as np
import joblib
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

from preprocess import build_feature_vector, EMBEDDING_DIM

EMBEDDING_MODELS = {
    "en": "roberta-base",
    "zh": "hfl/chinese-roberta-wwm-ext",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf_model = None
tokenizers: dict = {}
embed_models: dict = {}

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_clf()
    yield


app = FastAPI(title="Bot Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_clf():
    global clf_model
    if clf_model is None:
        clf_model = joblib.load("checkpoints/adaboost_bot.joblib")
    return clf_model


def load_embedding(lang: str):
    if lang not in embed_models:
        model_name = EMBEDDING_MODELS[lang]
        tokenizers[lang] = AutoTokenizer.from_pretrained(model_name)
        m = AutoModel.from_pretrained(
            model_name, ignore_mismatched_sizes=True,
            use_safetensors=(lang == "en"),
        )
        m.eval()
        m.to(device)
        embed_models[lang] = m
    return tokenizers[lang], embed_models[lang]


def encode_tweets(tweets: list[str], lang: str) -> np.ndarray:
    if not tweets:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    tokenizer, model = load_embedding(lang)
    embeddings = []

    with torch.no_grad():
        for text in tweets:
            if not text or not text.strip():
                continue
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=512, padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model(**inputs)
            cls_vec = output.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            embeddings.append(cls_vec)

    if not embeddings:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    return np.mean(embeddings, axis=0).astype(np.float32)


def predict_single(user: dict, lang: str) -> dict:
    tweet_emb = encode_tweets(user.get("tweets", []), lang)
    feature_vec = build_feature_vector(user, tweet_emb).reshape(1, -1)

    clf = load_clf()
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


# --- Pydantic models ---

class UserProfile(BaseModel):
    verified: bool = False
    default_profile_image: bool = False
    default_profile: bool = False
    protected: bool = False
    geo_enabled: bool = False
    contributors_enabled: bool = False
    followers_count: int = 0
    friends_count: int = 0
    listed_count: int = 0
    favourites_count: int = 0
    statuses_count: int = 0
    created_at: str = ""
    screen_name: str = ""
    name: str = ""
    description: str = ""
    location: str = ""
    url: str = ""
    tweets: list[str] = []


class SingleRequest(BaseModel):
    lang: str = "en"
    user: UserProfile


class BatchRequest(BaseModel):
    lang: str = "en"
    items: list[UserProfile]


# --- Routes ---

@app.get("/health")
def health():
    return {
        "success": True,
        "message": "ok",
        "data": {
            "status": "healthy",
            "model": "AdaBoost (n_estimators=50)",
            "device": str(device),
            "embedding_models": EMBEDDING_MODELS,
        },
    }


@app.post("/bot")
def detect_single(req: SingleRequest):
    try:
        result = predict_single(req.user.model_dump(), req.lang)
        return {"success": True, "message": "ok", "data": result}
    except Exception as e:
        return {"success": False, "message": str(e), "data": None}


@app.post("/bot/batch")
def detect_batch(req: BatchRequest):
    results = []
    errors = []

    for i, user in enumerate(req.items):
        try:
            r = predict_single(user.model_dump(), req.lang)
            r["index"] = i
            results.append(r)
        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    return {
        "success": True,
        "message": "ok",
        "data": {
            "total": len(req.items),
            "success_count": len(results),
            "failed_count": len(errors),
            "results": results,
            "errors": errors if errors else None,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=30102)
