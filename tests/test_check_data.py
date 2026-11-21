"""Checks for src/check_data.py. Run from the repo root with:
    python -m unittest discover -s tests -t .
"""

import unittest

from src.check_data import run_checks


def make(i, **overrides):
    record = {
        "id": f"alex-{i:04d}",
        "instruction": f"Explain topic number {i} to a beginner in simple terms.",
        "input": "",
        "output": f"Here is a clear explanation of topic number {i}, written in plain language.",
        "meta": {"author": "alex", "method": "self_instruct", "generator": "test-model"},
    }
    record.update(overrides)
    return record


def levels(report):
    return {(r["name"], r["level"]) for r in report["results"]}


def run(records, expected=None):
    return levels(run_checks(records, expected))


class CheckDataTests(unittest.TestCase):
    def test_clean_file_has_no_failures(self):
        report = run_checks([make(i) for i in range(1, 6)], expected=5)
        self.assertFalse([r for r in report["results"] if r["level"] == "FAIL"])

    def test_bad_format_fails_early(self):
        report = run_checks([{"id": "alex-0001", "instruction": "x"}])
        self.assertTrue(any(r["level"] == "FAIL" and r["name"] == "format" for r in report["results"]))

    def test_exact_duplicate_instruction_fails(self):
        records = [make(1), make(2, instruction=make(1)["instruction"])]
        self.assertIn(("duplicates", "FAIL"), run(records))

    def test_refusal_fails(self):
        records = [make(1, output="As an AI language model, I cannot answer that.")]
        self.assertIn(("refusals", "FAIL"), run(records))

    def test_leftover_placeholder_fails(self):
        records = [make(1, instruction="Explain this: {instruction}")]
        self.assertIn(("placeholders", "FAIL"), run(records))

    def test_numbering_left_over_from_parsing_warns(self):
        records = [make(1, instruction="Task 4: Explain how a list differs from a tuple.")]
        self.assertIn(("parsing", "WARN"), run(records))

    def test_near_duplicates_warn(self):
        records = [
            make(1, instruction="Explain the difference between a list and a tuple in Python."),
            make(2, instruction="Explain the difference between a list and a tuple in Python please."),
        ]
        self.assertIn(("near_dupes", "WARN"), run(records))

    def test_missing_meta_warns(self):
        records = [make(1, meta={"author": "alex"})]
        self.assertIn(("meta", "WARN"), run(records))

    def test_wrong_count_warns(self):
        self.assertIn(("count", "WARN"), run([make(1), make(2)], expected=1000))


if __name__ == "__main__":
    unittest.main()
