import argparse
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


MODEL_SPECS = {
    "en": {
        "remote": "roberta-base",
        "local": "roberta-base",
    },
    "zh": {
        "remote": "hfl/chinese-roberta-wwm-ext",
        "local": "chinese-roberta-wwm-ext",
    },
}

MODELS_DIR = Path(__file__).resolve().parent / "models"


def parse_args():
    parser = argparse.ArgumentParser(
        description="下载推理所需的 embedding 模型到本地 models 目录。"
    )
    parser.add_argument(
        "targets",
        nargs="*",
        choices=["all", "en", "zh"],
        default=["all"],
        help="要下载的模型，默认 all",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使目标目录已存在，也重新下载并覆盖保存",
    )
    return parser.parse_args()


def resolve_targets(raw_targets: list[str]) -> list[str]:
    if not raw_targets or "all" in raw_targets:
        return list(MODEL_SPECS.keys())
    # 去重并保留顺序
    return list(dict.fromkeys(raw_targets))


def download_model(key: str, force: bool):
    spec = MODEL_SPECS[key]
    target_dir = MODELS_DIR / spec["local"]

    if target_dir.exists() and not force:
        print(f"skip {key}: {target_dir} already exists")
        return

    print(f"downloading {key}: {spec['remote']} -> {target_dir}")
    tokenizer = AutoTokenizer.from_pretrained(spec["remote"])
    model = AutoModel.from_pretrained(spec["remote"], ignore_mismatched_sizes=True)

    target_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
    print(f"saved {key}: {target_dir}")


def main():
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for key in resolve_targets(args.targets):
        download_model(key, force=args.force)


if __name__ == "__main__":
    main()
