import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import shuffle
from tqdm import tqdm

from preprocess import NORMALIZATION_CONFIG, BOOL_FIELDS, PROPERTY_DIM
from utils.config import save_preprocess_config

parser = argparse.ArgumentParser(description="Train AdaBoost for bot detection")
parser.add_argument("--n_estimators", type=int, default=50)
parser.add_argument("--learning_rate", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=[0, 1, 2, 3, 4], nargs="+")
parser.add_argument("--output_dir", type=str, default="checkpoints")
parser.add_argument("--dataset_name", type=str, default=None,
                    help="Dataset name under Dataset/<name>; overrides --dataset_dir")
parser.add_argument("--dataset_dir", type=str, default=None)
parser.add_argument("--limit", type=int, default=None,
                    help="Use only the first N samples (smoke test)")
args = parser.parse_args()


def resolve_dataset_dir() -> Path:
    if args.dataset_name:
        return Path("Dataset") / args.dataset_name
    if args.dataset_dir:
        return Path(args.dataset_dir)
    raise SystemExit("Must pass --dataset_name or --dataset_dir")


def load_dataset_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.exists():
        raise SystemExit(
            f"{info_path} not found. Run build_dataset.py first to generate it."
        )
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dir = resolve_dataset_dir()
    info = load_dataset_info(dataset_dir)

    features = torch.load(dataset_dir / "features.pt", weights_only=False)
    bot_labels = torch.load(dataset_dir / "labels_bot.pt", weights_only=False)

    x = np.array(features, dtype=np.float32)
    y = np.array(bot_labels, dtype=np.int64)

    if args.limit is not None:
        x = x[: args.limit]
        y = y[: args.limit]

    expected_dim = info["feature_dim"]
    if x.shape[1] != expected_dim:
        raise SystemExit(
            f"features.pt dim {x.shape[1]} != dataset_info feature_dim {expected_dim}"
        )

    n = len(y)
    print(f"Dataset dir: {dataset_dir}")
    print(f"Qwen size:   {info['qwen_size']}  embedding_dim={info['embedding_dim']}")
    print(f"Dataset:     {n} samples, {x.shape[1]} features")
    print(f"Labels:      human={int((y==0).sum())} bot={int((y==1).sum())}")
    print(f"AdaBoost:    n_estimators={args.n_estimators} lr={args.learning_rate}")
    print(f"Seeds:       {args.random_seed}\n")

    best_f1 = -1.0
    best_clf = None
    best_seed = None

    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    seed_pbar = tqdm(args.random_seed, desc="training seeds", unit="seed")
    for seed in seed_pbar:
        idx = shuffle(np.arange(n), random_state=seed)
        train_idx = idx[: int(0.7 * n)]
        test_idx = idx[int(0.9 * n):]

        if len(train_idx) == 0 or len(test_idx) == 0:
            raise SystemExit(
                f"Not enough samples for split (n={n}). Increase --limit."
            )

        clf = AdaBoostClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            random_state=seed,
        )
        clf.fit(x[train_idx], y[train_idx])
        pred = clf.predict(x[test_idx])

        acc = accuracy_score(y[test_idx], pred)
        prec = precision_score(y[test_idx], pred, average="macro", zero_division=0)
        rec = recall_score(y[test_idx], pred, average="macro", zero_division=0)
        f1 = f1_score(y[test_idx], pred, average="macro", zero_division=0)

        acc_list.append(acc * 100)
        prec_list.append(prec * 100)
        rec_list.append(rec * 100)
        f1_list.append(f1 * 100)

        seed_pbar.set_postfix(
            seed=seed,
            acc=f"{acc*100:.2f}",
            f1=f"{f1*100:.2f}",
        )
        seed_pbar.write(
            f"seed={seed}: acc={acc*100:.2f}  prec={prec*100:.2f}  "
            f"rec={rec*100:.2f}  f1={f1*100:.2f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_clf = clf
            best_seed = seed
    seed_pbar.close()

    print(f"\nacc:       {np.mean(acc_list):.2f} +/- {np.std(acc_list):.2f}")
    print(f"precision: {np.mean(prec_list):.2f} +/- {np.std(prec_list):.2f}")
    print(f"recall:    {np.mean(rec_list):.2f} +/- {np.std(rec_list):.2f}")
    print(f"f1:        {np.mean(f1_list):.2f} +/- {np.std(f1_list):.2f}")

    model_path = os.path.join(args.output_dir, "adaboost_bot.joblib")
    joblib.dump(best_clf, model_path)
    print(f"\nBest model (seed={best_seed}, f1={best_f1*100:.2f}) saved to {model_path}")

    cfg = {
        "qwen_size": info["qwen_size"],
        "embedding_dim": info["embedding_dim"],
        "property_dim": PROPERTY_DIM,
        "feature_dim": info["feature_dim"],
        "normalization": NORMALIZATION_CONFIG,
        "bool_fields": BOOL_FIELDS,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "best_seed": best_seed,
        "trained_samples": n,
        "dataset_dir": str(dataset_dir),
    }
    save_preprocess_config(cfg, Path(args.output_dir) / "preprocess_config.json")
    print(f"Config saved to {Path(args.output_dir) / 'preprocess_config.json'}")


if __name__ == "__main__":
    main()
