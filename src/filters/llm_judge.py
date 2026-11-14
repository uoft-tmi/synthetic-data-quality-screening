"""Method 1: LLM-as-a-Judge (for example AlpaGasus).

An LLM rates each example against a rubric, and low-rated examples are dropped.
Connect your judge model in `call_judge`, then run:

    python -m src.filter_data --method llm_judge \
        --input data/raw/yourname.jsonl --output data/filtered/llm_judge/yourname.jsonl
"""

import re

from src.filters.base import ScoreFilter

RUBRIC = """You are grading a training example for a language model.
Rate it from 1 (poor) to 5 (excellent) for correctness, clarity and usefulness.
Reply with only the number.

Instruction: {instruction}
Input: {input}
Response: {output}
Rating:"""


class LLMJudgeFilter(ScoreFilter):
    name = "llm_judge"

    def __init__(self, threshold=4.0):
        super().__init__(threshold)

    def call_judge(self, prompt):
        """TODO: send `prompt` to your judge LLM and return its text reply."""
        raise NotImplementedError("Connect your judge model in LLMJudgeFilter.call_judge")

    def score(self, example):
        prompt = RUBRIC.format(
            instruction=example["instruction"],
            input=example.get("input", ""),
            output=example["output"],
        )
        reply = self.call_judge(prompt)
        match = re.search(r"[1-5]", reply)
        if match is None:
            print(f"Warning: no 1-5 rating in judge reply for {example['id']}: {reply!r}")
            return 0.0
        return float(match.group())
