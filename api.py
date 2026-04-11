import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import logging
import warnings

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*not sharded.*")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from utils.config import load_preprocess_config
from utils.device import resolve_device
from utils.embedding import load_embedding
from utils.inference import load_classifier, predict_user


_runtime: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cfg = load_preprocess_config()
    device = resolve_device("auto")
    clf = load_classifier()
    tokenizer, emb_model, dim = load_embedding(cfg["qwen_size"], device)
    _runtime.update(
        cfg=cfg, device=device, clf=clf, tokenizer=tokenizer, emb_model=emb_model, dim=dim
    )
    yield


app = FastAPI(title="Bot Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def predict_single(user: dict) -> dict:
    return predict_user(
        user,
        _runtime["clf"],
        _runtime["tokenizer"],
        _runtime["emb_model"],
        _runtime["device"],
    )


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")
    user: UserProfile


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[UserProfile]


@app.get("/health")
def health():
    return {
        "success": True,
        "message": "ok",
        "data": {
            "status": "healthy",
            "model": "AdaBoost (n_estimators=50)",
            "device": str(_runtime.get("device")),
            "embedding": {
                "qwen_size": _runtime.get("cfg", {}).get("qwen_size"),
                "embedding_dim": _runtime.get("dim"),
            },
        },
    }


@app.post("/bot")
def detect_single(req: SingleRequest):
    try:
        result = predict_single(req.user.model_dump())
        return {"success": True, "message": "ok", "data": result}
    except Exception as e:
        return {"success": False, "message": str(e), "data": None}


@app.post("/bot/batch")
def detect_batch(req: BatchRequest):
    results = []
    errors = []
    for i, user in enumerate(req.items):
        try:
            r = predict_single(user.model_dump())
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
