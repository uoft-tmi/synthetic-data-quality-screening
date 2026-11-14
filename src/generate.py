"""Generate synthetic data with the team's shared prompt template and settings.

Settings come from configs/generation.yaml and prompts come from prompts/.
Output is a JSONL file in the team's data format.

This is a skeleton: connect your generator model in `call_generator`.
"""

import argparse

import yaml

from src.io_utils import write_jsonl


def call_generator(prompt, config):
    """TODO: send `prompt` to the generator model named in the config and return its text."""
    raise NotImplementedError("Connect your generator model in call_generator")


def parse_examples(reply):
    """TODO: turn the model's reply into a list of (instruction, input, output) tuples.

    The parsing depends on the method and the prompt wording you use.
    """
    raise NotImplementedError("Write a parser for your prompt's output format")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data.")
    parser.add_argument("--author", required=True, help="your name, used in each example id")
    parser.add_argument("--output", required=True, help="e.g. data/raw/yourname.jsonl")
    parser.add_argument("--config", default="configs/generation.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(config["prompt_template"], encoding="utf-8") as f:
        template = f.read()

    records = []
    while len(records) < config["num_examples"]:
        # TODO: for Self-Instruct, fill the {seed instruction} slots with random seeds here.
        reply = call_generator(template, config)
        for instruction, input_text, output in parse_examples(reply):
            records.append(
                {
                    "id": f"{args.author}-{len(records) + 1:04d}",
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                    "meta": {"author": args.author, "method": config["method"]},
                }
            )
    write_jsonl(args.output, records[: config["num_examples"]])


if __name__ == "__main__":
    main()
