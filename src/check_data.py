"""Check that a JSONL file of synthetic data meets the team's standard.

validate.py only checks the file FORMAT. This script goes further: it applies the
team's agreed mechanical cleanup rules and prints a report with simple statistics,
so you can see whether your file is ready to commit.

It does NOT judge whether answers are good. Quality problems are left in on purpose,
because finding them is the job of the filtering methods.

Levels in the report:
  FAIL  must be fixed before you commit
  WARN  worth a look; fix it if it is a real problem
  NOTE  information only
  PASS  check is clean

Usage, from the repo root:
    python -m src.check_data data/raw/yourname.jsonl
    python -m src.check_data data/raw/yourname.jsonl --expected 1000
    python -m src.check_data data/raw/yourname.jsonl --strict    # warnings also fail

Exit code: 0 if the file is ready, 1 otherwise.
"""

import argparse
import re
import statistics
import sys
from collections import Counter

from src.io_utils import read_jsonl
from src.validate import validate_records

# --- Team rules. Change these only if the whole team agrees. -----------------------

ALLOWED_METHODS = {"self_instruct", "evol_instruct", "magpie"}
EXPECTED_META_FIELDS = ("author", "method", "generator")

# If an output opens with one of these, it is a refusal, not training data.
REFUSAL_PHRASES = (
    "as an ai language model",
    "as an ai,",
    "i'm sorry, but i",
    "i am sorry, but i",
    "i cannot assist",
    "i can't assist",
    "i cannot help with",
    "i can't help with",
    "i'm unable to",
    "i am unable to",
)
REFUSAL_WINDOW = 120  # only look at the first N characters of the output

# Leftover template placeholders mean the prompt was not filled in properly.
PLACEHOLDERS = (
    "{seed instruction}",
    "{new instruction}",
    "{instruction}",
    "{generated instruction}",
    "{model response}",
)

NEAR_DUPLICATE_THRESHOLD = 0.7   # word overlap (0 to 1) above which two instructions are "too similar"
MAX_NEAR_DUPLICATE_CHECK = 5000  # the pairwise check is slow on huge files
MIN_WORDS = 3                    # shorter instructions or outputs are suspicious
MAX_TOTAL_WORDS = 700            # roughly 1,000 tokens, the training max_seq_length

ID_PATTERN = re.compile(r"^(.+)-(\d+)$")
INSTRUCTION_LEFTOVER = re.compile(r"^\s*(task\s*\d+\s*[:.)-]|\d+\s*[.)]\s|instruction\s*:)", re.I)
OUTPUT_LEFTOVER = re.compile(r"^\s*(output|input|response)\s*:", re.I)

LEVEL_ORDER = {"FAIL": 0, "WARN": 1, "NOTE": 2, "PASS": 3}


# --- Small helpers ----------------------------------------------------------------

def normalize(text):
    return " ".join(str(text).lower().split())


def word_count(text):
    return len(str(text).split())


def first_ids(ids, limit=5):
    return [str(i) for i in ids[:limit]]


def describe_lengths(values):
    if not values:
        return "n/a"
    return (
        f"min {min(values)} / median {int(statistics.median(values))} / "
        f"mean {statistics.mean(values):.1f} / max {max(values)}"
    )


def expected_from_config(path="configs/generation.yaml"):
    """Read num_examples from the shared config, or return None if unavailable."""
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            return int(yaml.safe_load(f)["num_examples"])
    except Exception:
        return None


def result(name, level, message, ids=()):
    return {"name": name, "level": level, "message": message, "ids": first_ids(list(ids))}


# --- The checks -------------------------------------------------------------------

def check_count(records, expected):
    n = len(records)
    if expected is None:
        return result("count", "NOTE", f"{n} examples (no expected count to compare with)")
    if n == expected:
        return result("count", "PASS", f"{n} examples, as expected")
    return result("count", "WARN", f"{n} examples, but {expected} were expected")


def check_exact_duplicates(records):
    seen, dups = set(), []
    for r in records:
        key = (normalize(r["instruction"]), normalize(r.get("input", "")))
        if key in seen:
            dups.append(r["id"])
        seen.add(key)
    if dups:
        return result(
            "duplicates", "FAIL",
            f"{len(dups)} example(s) repeat an earlier instruction and input. Remove the repeats", dups,
        )
    return result("duplicates", "PASS", "no exact duplicate instructions")


def check_refusals(records):
    bad = []
    for r in records:
        opening = normalize(r["output"])[:REFUSAL_WINDOW]
        if any(phrase in opening for phrase in REFUSAL_PHRASES):
            bad.append(r["id"])
    if bad:
        return result("refusals", "FAIL", f"{len(bad)} output(s) look like refusals. Remove them", bad)
    return result("refusals", "PASS", "no refusals")


def check_placeholders(records):
    bad = []
    for r in records:
        text = " ".join(str(r.get(f, "")) for f in ("instruction", "input", "output"))
        if any(p in text for p in PLACEHOLDERS):
            bad.append(r["id"])
    if bad:
        return result(
            "placeholders", "FAIL",
            f"{len(bad)} example(s) still contain template placeholders like {{instruction}}", bad,
        )
    return result("placeholders", "PASS", "no leftover template placeholders")


def check_parsing_leftovers(records):
    bad = [
        r["id"]
        for r in records
        if INSTRUCTION_LEFTOVER.match(r["instruction"]) or OUTPUT_LEFTOVER.match(r["output"])
    ]
    if bad:
        return result(
            "parsing", "WARN",
            f"{len(bad)} example(s) start with numbering or a label such as 'Task 1:' or 'Output:'. "
            "Your parsing code may not be stripping it", bad,
        )
    return result("parsing", "PASS", "no leftover numbering or labels from parsing")


def check_instruction_equals_output(records):
    bad = [r["id"] for r in records if normalize(r["instruction"]) == normalize(r["output"])]
    if bad:
        return result("copy", "WARN", f"{len(bad)} example(s) have an output identical to the instruction", bad)
    return result("copy", "PASS", "no outputs that just repeat the instruction")


def check_lengths(records):
    out = []
    short = [r["id"] for r in records if word_count(r["instruction"]) < MIN_WORDS or word_count(r["output"]) < MIN_WORDS]
    if short:
        out.append(result(
            "short", "WARN", f"{len(short)} example(s) have an instruction or output under {MIN_WORDS} words", short,
        ))
    else:
        out.append(result("short", "PASS", "no extremely short instructions or outputs"))
    long_ids = [
        r["id"] for r in records
        if word_count(r["instruction"]) + word_count(r.get("input", "")) + word_count(r["output"]) > MAX_TOTAL_WORDS
    ]
    if long_ids:
        out.append(result(
            "long", "WARN",
            f"{len(long_ids)} example(s) are over about {MAX_TOTAL_WORDS} words and may be cut off during training",
            long_ids,
        ))
    else:
        out.append(result("long", "PASS", "no examples likely to exceed the training length limit"))
    return out


def check_truncated(records):
    bad = [r["id"] for r in records if r["output"].rstrip() and r["output"].rstrip()[-1].isalpha()]
    pct = 100 * len(bad) / len(records)
    if bad:
        return result(
            "truncated", "WARN",
            f"{len(bad)} output(s) ({pct:.0f}%) do not end in punctuation and may be cut off. "
            "Many are fine, such as short answers", bad,
        )
    return result("truncated", "PASS", "all outputs end cleanly")


def check_duplicate_outputs(records):
    counts = Counter(normalize(r["output"]) for r in records if len(r["output"]) >= 20)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    if repeated:
        return result("same_output", "WARN", f"{repeated} output(s) are word-for-word copies of another output")
    return result("same_output", "PASS", "no repeated outputs")


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def check_near_duplicates(records):
    if len(records) > MAX_NEAR_DUPLICATE_CHECK:
        return result("near_dupes", "NOTE", f"near-duplicate check skipped (more than {MAX_NEAR_DUPLICATE_CHECK} examples)")
    keys = [(normalize(r["instruction"]), normalize(r.get("input", ""))) for r in records]
    words = [set(k[0].split()) for k in keys]
    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if keys[i] == keys[j]:
                continue  # exact duplicates are reported separately
            small, big = sorted((len(words[i]), len(words[j])))
            if big == 0 or small / big < NEAR_DUPLICATE_THRESHOLD:
                continue  # sizes too different to be this similar
            if jaccard(words[i], words[j]) >= NEAR_DUPLICATE_THRESHOLD:
                pairs.append(f"{records[i]['id']}~{records[j]['id']}")
    if pairs:
        return result(
            "near_dupes", "WARN",
            f"{len(pairs)} pair(s) of very similar instructions. Ask for more variety (topics, task types) or change your seed tasks. "
            "Only remove them if the team agreed on a near-duplicate rule", pairs,
        )
    return result("near_dupes", "PASS", "no near-duplicate instructions")


def check_meta(records):
    out = []
    missing = []
    for r in records:
        meta = r.get("meta") if isinstance(r.get("meta"), dict) else {}
        if any(f not in meta for f in EXPECTED_META_FIELDS):
            missing.append(r["id"])
    if missing:
        out.append(result(
            "meta", "WARN",
            f"{len(missing)} example(s) are missing one of meta.{', meta.'.join(EXPECTED_META_FIELDS)}", missing,
        ))
    else:
        out.append(result("meta", "PASS", "every example records its author, method and generator"))

    methods = Counter((r.get("meta") or {}).get("method") for r in records if isinstance(r.get("meta"), dict))
    unknown = [m for m in methods if m is not None and m not in ALLOWED_METHODS]
    if unknown:
        out.append(result(
            "method", "WARN",
            f"unrecognised meta.method value(s): {', '.join(map(str, unknown))} "
            f"(expected one of {', '.join(sorted(ALLOWED_METHODS))})",
        ))
    for field in ("generator", "author"):
        values = {(r.get("meta") or {}).get(field) for r in records if isinstance(r.get("meta"), dict)}
        values.discard(None)
        if len(values) > 1:
            out.append(result(
                "mixed_" + field, "WARN",
                f"this file mixes several meta.{field} values: {', '.join(sorted(map(str, values)))}",
            ))
    return out


def check_ids(records):
    bad = [r["id"] for r in records if not ID_PATTERN.match(r["id"])]
    prefixes = {ID_PATTERN.match(r["id"]).group(1) for r in records if ID_PATTERN.match(r["id"])}
    out = []
    if bad:
        out.append(result("id_format", "WARN", f"{len(bad)} id(s) don't look like name-0001", bad))
    else:
        out.append(result("id_format", "PASS", "ids follow the name-0001 pattern"))
    if len(prefixes) > 1:
        out.append(result("id_prefix", "WARN", f"ids use more than one name prefix: {', '.join(sorted(prefixes))}"))
    return out


# --- Putting it together ----------------------------------------------------------

def build_summary(records):
    n = len(records)
    with_input = sum(1 for r in records if str(r.get("input", "")).strip())
    methods = Counter((r.get("meta") or {}).get("method") for r in records if isinstance(r.get("meta"), dict))
    generators = Counter((r.get("meta") or {}).get("generator") for r in records if isinstance(r.get("meta"), dict))
    lines = [
        f"examples: {n}",
        f"instruction words: {describe_lengths([word_count(r['instruction']) for r in records])}",
        f"output words:      {describe_lengths([word_count(r['output']) for r in records])}",
        f"examples with an input: {with_input} ({100 * with_input / n:.0f}%)",
    ]
    methods.pop(None, None)
    generators.pop(None, None)
    if methods:
        lines.append("methods: " + ", ".join(f"{k} ({v})" for k, v in methods.items()))
    if generators:
        lines.append("generators: " + ", ".join(f"{k} ({v})" for k, v in generators.items()))
    return lines


def run_checks(records, expected=None):
    """Run every check on a list of example dicts. Returns {'summary': [...], 'results': [...]}."""
    _, format_errors = validate_records(records)
    if format_errors:
        return {
            "summary": [],
            "results": [
                result("format", "FAIL", f"{len(format_errors)} format problem(s). Run src.validate for details"),
                *[result("format", "FAIL", e) for e in format_errors[:10]],
            ],
        }
    if not records:
        return {"summary": [], "results": [result("empty", "FAIL", "the file has no examples")]}

    results = [check_count(records, expected), check_exact_duplicates(records), check_refusals(records),
               check_placeholders(records), check_parsing_leftovers(records),
               check_instruction_equals_output(records)]
    results += check_lengths(records)
    results += [check_truncated(records), check_duplicate_outputs(records), check_near_duplicates(records)]
    results += check_meta(records)
    results += check_ids(records)
    results.sort(key=lambda r: LEVEL_ORDER[r["level"]])  # stable: keeps check order within a level
    return {"summary": build_summary(records), "results": results}


def print_report(path, report):
    print(f"Checking {path}\n")
    if report["summary"]:
        print("Summary")
        for line in report["summary"]:
            print(f"  {line}")
        print()
    print("Checks")
    for r in report["results"]:
        ids = f" (e.g. {', '.join(r['ids'])})" if r["ids"] else ""
        print(f"  {r['level']:<5} {r['message']}{ids}")
    fails = sum(1 for r in report["results"] if r["level"] == "FAIL")
    warns = sum(1 for r in report["results"] if r["level"] == "WARN")
    print()
    if fails:
        print(f"NOT READY: {fails} failure(s) to fix, {warns} warning(s) to review.")
    elif warns:
        print(f"READY, with {warns} warning(s) to review.")
    else:
        print("READY: all checks passed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check a synthetic data file against the team standard.")
    parser.add_argument("file", help="JSONL file, e.g. data/raw/yourname.jsonl")
    parser.add_argument("--expected", type=int, default=None,
                        help="expected number of examples (default: num_examples in configs/generation.yaml)")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args(argv)

    try:
        records = list(read_jsonl(args.file))
    except (ValueError, OSError) as e:
        print(f"Could not read {args.file}: {e}")
        return 1

    expected = args.expected if args.expected is not None else expected_from_config()
    report = run_checks(records, expected)
    print_report(args.file, report)

    levels = [r["level"] for r in report["results"]]
    if "FAIL" in levels or (args.strict and "WARN" in levels):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
