import json
from pathlib import Path
from typing import Iterator


def iter_jsonl_records(path: str | Path) -> Iterator[tuple[int, dict]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error at line {line_no}: {e}") from e
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            yield line_no, record
