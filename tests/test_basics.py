"""Basic checks. Run from the repo root with:
    python -m unittest discover -s tests -t .
"""

import unittest

from src.filters.base import ScoreFilter
from src.filters.llm_judge import LLMJudgeFilter
from src.validate import validate_records


def make(i, **overrides):
    record = {"id": f"t-{i}", "instruction": "Do a thing.", "input": "", "output": "Done."}
    record.update(overrides)
    return record


class ValidateTests(unittest.TestCase):
    def test_good_records_pass(self):
        count, errors = validate_records([make(1), make(2)])
        self.assertEqual(count, 2)
        self.assertEqual(errors, [])

    def test_missing_output_and_duplicate_id_are_caught(self):
        records = [make(1), make(1, output="")]
        _, errors = validate_records(records)
        self.assertTrue(any("output" in e for e in errors))
        self.assertTrue(any("duplicate id" in e for e in errors))


class LengthFilter(ScoreFilter):
    """A toy filter used only for testing: longer outputs score higher."""

    name = "length"

    def score(self, example):
        return float(len(example["output"]))


class FilterTests(unittest.TestCase):
    def test_score_filter_keeps_high_scores_and_records_the_score(self):
        examples = [make(1, output="short"), make(2, output="a much longer answer")]
        kept = LengthFilter(threshold=10).filter(examples)
        self.assertEqual([e["id"] for e in kept], ["t-2"])
        self.assertEqual(kept[0]["meta"]["length_score"], 20.0)

    def test_llm_judge_reads_the_rating_from_the_reply(self):
        class FakeJudge(LLMJudgeFilter):
            def call_judge(self, prompt):
                return "Rating: 5"

        kept = FakeJudge(threshold=4).filter([make(1)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["meta"]["llm_judge_score"], 5.0)


if __name__ == "__main__":
    unittest.main()
