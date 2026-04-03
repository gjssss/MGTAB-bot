import argparse
import json
from pathlib import Path

from predict import predict_bot


def parse_args():
    parser = argparse.ArgumentParser(
        description="评测 jsonl 数据集上的 bot 检测正确率。"
    )
    parser.add_argument("input", help="输入 jsonl 文件路径")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="输出评测结果的 json 文件路径",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="默认语言，单条数据未提供 lang 时使用，默认 en",
    )
    return parser.parse_args()


def normalize_record(record: dict, default_lang: str) -> tuple[dict, bool, str]:
    if "label" not in record:
        raise ValueError("缺少 label 字段")
    if not isinstance(record["label"], bool):
        raise ValueError("label 必须为 boolean")

    user = record.get("user", record)
    if not isinstance(user, dict):
        raise ValueError("user 字段必须为对象")

    user = dict(user)
    user.pop("label", None)
    user.pop("lang", None)

    lang = record.get("lang", default_lang)
    if not isinstance(lang, str) or not lang:
        raise ValueError("lang 必须为非空字符串")

    return user, record["label"], lang


def build_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for item in results if item["correct"])

    tp = sum(1 for item in results if item["predicted_is_bot"] and item["expected_is_bot"])
    tn = sum(1 for item in results if not item["predicted_is_bot"] and not item["expected_is_bot"])
    fp = sum(1 for item in results if item["predicted_is_bot"] and not item["expected_is_bot"])
    fn = sum(1 for item in results if not item["predicted_is_bot"] and item["expected_is_bot"])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = correct / total if total else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "bot_precision": round(precision, 6),
        "bot_recall": round(recall, 6),
        "bot_f1": round(f1, 6),
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    }


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    results = []
    errors = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("每一行必须是 JSON object")

                user, expected_is_bot, lang = normalize_record(record, args.lang)
                prediction = predict_bot(user, lang=lang)
                predicted_is_bot = prediction["label"] == "bot"

                results.append(
                    {
                        "line": line_no,
                        "lang": lang,
                        "expected_is_bot": expected_is_bot,
                        "expected_label": "bot" if expected_is_bot else "human",
                        "predicted_is_bot": predicted_is_bot,
                        "predicted_label": prediction["label"],
                        "confidence": prediction["confidence"],
                        "correct": predicted_is_bot == expected_is_bot,
                    }
                )
            except Exception as e:
                errors.append(
                    {
                        "line": line_no,
                        "error": str(e),
                    }
                )

    output = {
        "success": True,
        "message": "ok",
        "data": {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "metrics": build_metrics(results),
            "success_count": len(results),
            "failed_count": len(errors),
            "results": results,
            "errors": errors if errors else None,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    metrics = output["data"]["metrics"]
    print(
        f"accuracy={metrics['accuracy']:.6f} "
        f"correct={metrics['correct']}/{metrics['total']} "
        f"failed={output['data']['failed_count']}"
    )
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
