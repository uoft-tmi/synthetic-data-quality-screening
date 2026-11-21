"""Turn raw text pasted from a chat window into a JSONL data file.

Ask the chatbot for JSON lines with the keys "instruction", "input" and "output"
(see docs/synthetic-data-generation.md). Paste its replies into a text file, then run:

    python -m src.prepare_chat_data --input pasted.txt --author yourname \
        --method self_instruct --generator "ChatGPT, model name, 2026-11-21" \
        --output data/raw/yourname.jsonl

It keeps every valid example, adds an id and meta, and tells you which lines it had to
skip. Chatty text around the examples, code fences and list brackets are ignored.

Use --append to add more examples to an existing file. Numbering continues where it
left off. Without --append, the script refuses to overwrite an existing file.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from src.io_utils import read_jsonl, write_jsonl


def extract_json_objects(text):
    """Find every JSON object in the text. Returns (objects, bad), where bad is a list of
    (line_number, reason) for places that looked like an object but could not be parsed."""
    decoder = json.JSONDecoder()
    objects, bad = [], []
    i = 0
    while True:
        i = text.find("{", i)
        if i == -1:
            break
        line_no = text.count("\n", 0, i) + 1
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            bad.append((line_no, "not valid JSON"))
            newline = text.find("\n", i)
            if newline == -1:
                break
            i = newline  # skip the rest of this line and carry on
            continue
        objects.append((line_no, obj))
        i = end
    return objects, bad


def parse_pasted_text(text):
    """Return (examples, problems). Each example has instruction, input and output."""
    objects, problems = extract_json_objects(text)
    examples = []
    for line_no, obj in objects:
        if not isinstance(obj, dict):
            problems.append((line_no, "not a JSON object"))
            continue
        instruction, output = obj.get("instruction"), obj.get("output")
        if not isinstance(instruction, str) or not instruction.strip():
            problems.append((line_no, "missing or empty 'instruction'"))
            continue
        if not isinstance(output, str) or not output.strip():
            problems.append((line_no, "missing or empty 'output'"))
            continue
        input_text = obj.get("input", "")
        examples.append(
            {
                "instruction": instruction.strip(),
                "input": input_text.strip() if isinstance(input_text, str) else "",
                "output": output.strip(),
            }
        )
    return examples, sorted(problems)


def next_number(existing, author):
    """The next id number for this author, continuing after any existing ids."""
    pattern = re.compile(rf"^{re.escape(author)}-(\d+)$")
    numbers = [int(m.group(1)) for r in existing if (m := pattern.match(str(r.get("id", ""))))]
    return max(numbers, default=0) + 1


def build_records(examples, author, method, generator, start=1):
    return [
        {
            "id": f"{author}-{start + i:04d}",
            **example,
            "meta": {"author": author, "method": method, "generator": generator, "route": "chat"},
        }
        for i, example in enumerate(examples)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Turn pasted chat output into a JSONL data file.")
    parser.add_argument("--input", required=True, help="text file with the chatbot's replies pasted in")
    parser.add_argument("--output", required=True, help="e.g. data/raw/yourname.jsonl")
    parser.add_argument("--author", required=True, help="your name, used in each id (no spaces)")
    parser.add_argument("--method", required=True, choices=["self_instruct", "evol_instruct", "magpie"])
    parser.add_argument("--generator", required=True, help='which chatbot and model, e.g. "ChatGPT, model name, date"')
    parser.add_argument("--append", action="store_true", help="add to an existing output file")
    args = parser.parse_args(argv)

    if re.search(r"\s", args.author):
        print("The author name must not contain spaces (it is used in each id).")
        return 1

    out_path = Path(args.output)
    existing = []
    if out_path.exists():
        if not args.append:
            print(f"{out_path} already exists. Use --append to add to it, or choose a new --output.")
            return 1
        existing = list(read_jsonl(out_path))

    text = Path(args.input).read_text(encoding="utf-8")
    examples, problems = parse_pasted_text(text)
    start = next_number(existing, args.author)
    records = build_records(examples, args.author, args.method, args.generator, start)
    total = write_jsonl(out_path, existing + records)

    print(f"Read {len(examples)} valid example(s) from {args.input}.")
    if problems:
        print(f"Skipped {len(problems)} problem(s). Fix these in the pasted text and re-run if you want them:")
        for line_no, reason in problems[:20]:
            print(f"  - line {line_no}: {reason}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
    print(f"Wrote {total} example(s) in total to {out_path}.")
    print(f"Next: python -m src.validate {out_path}   then   python -m src.check_data {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
