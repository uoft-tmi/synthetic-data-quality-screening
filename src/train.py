"""Fine-tune the shared model on a JSONL file.

Everyone uses this same script and configs/training.yaml. The only thing that changes
between runs is the data file, so differences in results come from the data and the
filter, not the training setup.

This is a skeleton: the base model has not been chosen yet. Once it is, implement
`train` (for example with Hugging Face transformers, peft and trl).
"""

import argparse

import yaml


def train(data_path, output_dir, config):
    """TODO: fine-tune config['base_model'] on data_path and save to output_dir."""
    raise NotImplementedError("Implement training once the base model is chosen")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on a JSONL file.")
    parser.add_argument("--data", required=True, help="e.g. data/filtered/llm_judge/yourname.jsonl")
    parser.add_argument("--output-dir", required=True, help="where to save the trained model")
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(args.data, args.output_dir, config)


if __name__ == "__main__":
    main()
