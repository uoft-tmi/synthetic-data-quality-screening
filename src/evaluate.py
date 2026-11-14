"""Score a trained model on the shared test set in data/eval/.

Everyone must evaluate on the same test set with the same metrics. The metrics will be
covered in the Week 8 lesson. Write results to results/ as one file per run.

This is a skeleton: implement `evaluate` once the model and metrics are chosen.
"""

import argparse


def evaluate(model_dir, eval_path):
    """TODO: load the model in model_dir, run it on eval_path, and return a dict of metrics."""
    raise NotImplementedError("Implement evaluation once the metrics are chosen")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--eval-data", default="data/eval/test.jsonl")
    args = parser.parse_args()
    print(evaluate(args.model_dir, args.eval_data))


if __name__ == "__main__":
    main()
