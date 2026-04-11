"""Evaluate bot detection accuracy over a labeled JSONL."""
import argparse
import json
from pathlib import Path

from utils.config import load_preprocess_config
from utils.device import resolve_device
from utils.embedding import load_embedding
from utils.inference import load_classifier, predict_user
from utils.jsonl import iter_jsonl_records


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate bot detection on a JSONL dataset")
    p.add_argument("input", help="Input JSONL path")
    p.add_argument("-o", "--output", required=True, help="Output JSON report path")
    return p.parse_args()


def normalize_record(record: dict) -> tuple[dict, bool]:
    if "label" not in record:
        raise ValueError("missing 'label' field")
    if not isinstance(record["label"], bool):
        raise ValueError("'label' must be boolean")

    user = record.get("user", record)
    if not isinstance(user, dict):
        raise ValueError("'user' field must be an object")

    user = dict(user)
    user.pop("label", None)
    return user, record["label"]


def build_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    tp = sum(1 for r in results if r["predicted_is_bot"] and r["expected_is_bot"])
    tn = sum(1 for r in results if not r["predicted_is_bot"] and not r["expected_is_bot"])
    fp = sum(1 for r in results if r["predicted_is_bot"] and not r["expected_is_bot"])
    fn = sum(1 for r in results if not r["predicted_is_bot"] and r["expected_is_bot"])

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
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    cfg = load_preprocess_config()
    device = resolve_device("auto")
    clf = load_classifier()
    tokenizer, emb_model, dim = load_embedding(cfg["qwen_size"], device)
    print(f"[test] qwen_size={cfg['qwen_size']} dim={dim} device={device}")

    results: list[dict] = []
    errors: list[dict] = []

    for line_no, record in iter_jsonl_records(input_path):
        try:
            user, expected = normalize_record(record)
            prediction = predict_user(user, clf, tokenizer, emb_model, device)
            predicted_is_bot = prediction["label"] == "bot"
            results.append({
                "line": line_no,
                "expected_is_bot": expected,
                "expected_label": "bot" if expected else "human",
                "predicted_is_bot": predicted_is_bot,
                "predicted_label": prediction["label"],
                "confidence": prediction["confidence"],
                "correct": predicted_is_bot == expected,
            })
        except Exception as e:
            errors.append({"line": line_no, "error": str(e)})

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

    m = output["data"]["metrics"]
    print(
        f"accuracy={m['accuracy']:.6f} "
        f"correct={m['correct']}/{m['total']} "
        f"failed={output['data']['failed_count']}"
    )
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
