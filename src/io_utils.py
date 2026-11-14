"""Small helpers for reading and writing JSONL files (one JSON object per line)."""

import json
from pathlib import Path


def read_jsonl(path):
    """Yield one dict per line. Blank lines are skipped."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e


def write_jsonl(path, records):
    """Write records to a JSONL file, creating folders as needed. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
