"""Run one filtering method on a JSONL file.

Usage, from the repo root:
    python -m src.filter_data --method llm_judge \
        --input data/raw/yourname.jsonl \
        --output data/filtered/llm_judge/yourname.jsonl \
        --threshold 4
"""

import argparse

from src.filters import FILTERS, get_filter


def main():
    parser = argparse.ArgumentParser(description="Filter a JSONL file of synthetic data.")
    parser.add_argument("--method", required=True, choices=sorted(FILTERS))
    parser.add_argument("--input", required=True, help="JSONL file to filter")
    parser.add_argument("--output", required=True, help="where to write the kept examples")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="minimum score to keep (only for methods that score each example)",
    )
    args = parser.parse_args()

    kwargs = {} if args.threshold is None else {"threshold": args.threshold}
    get_filter(args.method)(**kwargs).run(args.input, args.output)


if __name__ == "__main__":
    main()
