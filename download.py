import os
import argparse
import platform
import subprocess
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


def normalize_proxy_environment() -> dict[str, str]:
    proxy_env = {}
    for lower, upper in (
        ("http_proxy", "HTTP_PROXY"),
        ("https_proxy", "HTTPS_PROXY"),
        ("all_proxy", "ALL_PROXY"),
        ("no_proxy", "NO_PROXY"),
    ):
        value = os.environ.get(lower) or os.environ.get(upper)
        if value:
            os.environ[lower] = value
            os.environ[upper] = value
            proxy_env[lower] = value
    return proxy_env


def parse_scutil_proxy_output(output: str) -> dict[str, str]:
    values = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if " : " not in line:
            continue
        key, value = line.split(" : ", 1)
        values[key.strip()] = value.strip()

    proxy_env = {}

    if values.get("HTTPEnable") == "1" and values.get("HTTPProxy") and values.get("HTTPPort"):
        proxy_env["http_proxy"] = f"http://{values['HTTPProxy']}:{values['HTTPPort']}"
    if values.get("HTTPSEnable") == "1" and values.get("HTTPSProxy") and values.get("HTTPSPort"):
        proxy_env["https_proxy"] = f"http://{values['HTTPSProxy']}:{values['HTTPSPort']}"
    if values.get("SOCKSEnable") == "1" and values.get("SOCKSProxy") and values.get("SOCKSPort"):
        proxy_env["all_proxy"] = f"socks5://{values['SOCKSProxy']}:{values['SOCKSPort']}"

    return proxy_env


def read_macos_system_proxy() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["scutil", "--proxy"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}

    return parse_scutil_proxy_output(result.stdout)


def configure_proxy_environment() -> dict[str, str]:
    proxy_env = normalize_proxy_environment()
    if proxy_env:
        print("using proxy from environment")
        return proxy_env

    if platform.system() != "Darwin":
        print("proxy not configured")
        return {}

    proxy_env = read_macos_system_proxy()
    if not proxy_env:
        print("proxy not configured")
        return {}

    for lower, value in proxy_env.items():
        os.environ[lower] = value
        os.environ[lower.upper()] = value

    print("using proxy from macOS system settings")
    return proxy_env


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
    configure_proxy_environment()

    for key in resolve_targets(args.targets):
        download_model(key, force=args.force)


if __name__ == "__main__":
    main()
