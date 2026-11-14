"""The shared interface every filtering method follows.

A filter takes a JSONL file of examples and writes a JSONL file of the examples it keeps.
Because every filter has this same shape, train.py does not care which filter made the
data, and combining filters later (Week 13) is straightforward.

To add a method, subclass one of these:
  - ScoreFilter: you give each example a number; examples at or above `threshold` are kept.
  - BaseFilter:  you decide directly which examples to keep (for methods that rank,
                 cluster or select a subset rather than score one example at a time).
"""

import abc

from src.io_utils import read_jsonl, write_jsonl
from src.validate import validate_records


class BaseFilter(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def filter(self, examples):
        """Take a list of example dicts and return the list to keep.

        You may add information under example["meta"], such as the score you gave.
        Never change the id, instruction, input or output fields.
        """

    def run(self, input_path, output_path):
        """Read input_path, filter it, and write the kept examples to output_path."""
        examples = list(read_jsonl(input_path))
        _, errors = validate_records(examples)
        if errors:
            raise ValueError(
                f"{input_path} does not match the data format "
                f"(run `python -m src.validate {input_path}` for details)"
            )
        kept = self.filter(examples)
        written = write_jsonl(output_path, kept)
        print(f"[{self.name}] kept {written} of {len(examples)} examples -> {output_path}")
        return written


class ScoreFilter(BaseFilter):
    """A filter that scores each example and keeps those with score >= threshold."""

    def __init__(self, threshold):
        self.threshold = threshold

    @abc.abstractmethod
    def score(self, example):
        """Return a number for this example. Higher means better."""

    def filter(self, examples):
        kept = []
        for example in examples:
            value = self.score(example)
            result = dict(example)
            meta = dict(result.get("meta") or {})
            meta[f"{self.name}_score"] = value
            result["meta"] = meta
            if value >= self.threshold:
                kept.append(result)
        return kept
