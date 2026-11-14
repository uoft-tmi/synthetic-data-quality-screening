"""Check that JSONL files follow the team's data format.

Required fields: id, instruction, output (non-empty strings, id must be unique).
Optional fields: input (string), meta (object).

Usage, from the repo root:
    python -m src.validate data/raw/yourname.jsonl
"""

import sys

from src.io_utils import read_jsonl

REQUIRED_FIELDS = ("id", "instruction", "output")


def validate_records(records):
    """Return (number of records, list of problems found)."""
    errors = []
    seen_ids = set()
    count = 0
    for i, rec in enumerate(records, start=1):
        count += 1
        if not isinstance(rec, dict):
            errors.append(f"line {i}: not a JSON object")
            continue
        for field in REQUIRED_FIELDS:
            value = rec.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"line {i}: '{field}' is missing or not a non-empty string")
        if "input" in rec and not isinstance(rec["input"], str):
            errors.append(f"line {i}: 'input' must be a string")
        if "meta" in rec and not isinstance(rec["meta"], dict):
            errors.append(f"line {i}: 'meta' must be an object")
        rec_id = rec.get("id")
        if isinstance(rec_id, str):
            if rec_id in seen_ids:
                errors.append(f"line {i}: duplicate id '{rec_id}'")
            seen_ids.add(rec_id)
    return count, errors


def validate_file(path):
    """Validate one JSONL file. Returns (number of records, list of problems)."""
    try:
        records = list(read_jsonl(path))
    except (ValueError, OSError) as e:
        return 0, [str(e)]
    return validate_records(records)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("Usage: python -m src.validate FILE.jsonl [FILE2.jsonl ...]")
        return 2
    any_problems = False
    for path in argv:
        count, errors = validate_file(path)
        if errors:
            any_problems = True
            print(f"{path}: {len(errors)} problem(s) in {count} example(s)")
            for message in errors[:20]:
                print(f"  - {message}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")
        else:
            print(f"{path}: OK ({count} examples)")
    return 1 if any_problems else 0


if __name__ == "__main__":
    sys.exit(main())
