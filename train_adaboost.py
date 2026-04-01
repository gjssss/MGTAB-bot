import argparse
import json
import os

import joblib
import numpy as np
import torch
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import shuffle

from preprocess import NORMALIZATION_CONFIG, BOOL_FIELDS, FEATURE_DIM

parser = argparse.ArgumentParser(description="Train AdaBoost for bot detection")
parser.add_argument("--n_estimators", type=int, default=50)
parser.add_argument("--learning_rate", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=[0, 1, 2, 3, 4], nargs="+")
parser.add_argument("--output_dir", type=str, default="checkpoints")
parser.add_argument("--dataset_dir", type=str, default="Dataset/MGTAB")
args = parser.parse_args()


def main():
    os.makedirs(args.output_dir, exist_ok=True)

    features = torch.load(
        os.path.join(args.dataset_dir, "features.pt"), weights_only=False
    )
    bot_labels = torch.load(
        os.path.join(args.dataset_dir, "labels_bot.pt"), weights_only=False
    )

    x = np.array(features, dtype=np.float32)
    y = np.array(bot_labels, dtype=np.int64)
    n = len(y)

    print(f"Dataset: {n} samples, {x.shape[1]} features")
    print(f"Label distribution: human={int((y==0).sum())}, bot={int((y==1).sum())}")
    print(f"AdaBoost: n_estimators={args.n_estimators}, lr={args.learning_rate}")
    print(f"Seeds: {args.random_seed}\n")

    best_f1 = -1
    best_clf = None
    best_seed = None

    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for seed in args.random_seed:
        idx = shuffle(np.arange(n), random_state=seed)
        train_idx = idx[: int(0.7 * n)]
        test_idx = idx[int(0.9 * n) :]

        clf = AdaBoostClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            random_state=seed,
        )
        clf.fit(x[train_idx], y[train_idx])
        pred = clf.predict(x[test_idx])

        acc = accuracy_score(y[test_idx], pred)
        prec = precision_score(y[test_idx], pred, average="macro")
        rec = recall_score(y[test_idx], pred, average="macro")
        f1 = f1_score(y[test_idx], pred, average="macro")

        acc_list.append(acc * 100)
        prec_list.append(prec * 100)
        rec_list.append(rec * 100)
        f1_list.append(f1 * 100)

        print(f"seed={seed}: acc={acc*100:.2f}  prec={prec*100:.2f}  "
              f"rec={rec*100:.2f}  f1={f1*100:.2f}")

        if f1 > best_f1:
            best_f1 = f1
            best_clf = clf
            best_seed = seed

    print(f"\nacc:       {np.mean(acc_list):.2f} +/- {np.std(acc_list):.2f}")
    print(f"precision: {np.mean(prec_list):.2f} +/- {np.std(prec_list):.2f}")
    print(f"recall:    {np.mean(rec_list):.2f} +/- {np.std(rec_list):.2f}")
    print(f"f1:        {np.mean(f1_list):.2f} +/- {np.std(f1_list):.2f}")

    # Save best model
    model_path = os.path.join(args.output_dir, "adaboost_bot.joblib")
    joblib.dump(best_clf, model_path)
    print(f"\nBest model (seed={best_seed}, f1={best_f1*100:.2f}) saved to {model_path}")

    # Save preprocessing config
    config = {
        "feature_dim": FEATURE_DIM,
        "normalization": NORMALIZATION_CONFIG,
        "bool_fields": BOOL_FIELDS,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "best_seed": best_seed,
        "embedding_models": {
            "en": "roberta-base",
            "zh": "hfl/chinese-roberta-wwm-ext",
        },
    }
    config_path = os.path.join(args.output_dir, "preprocess_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
