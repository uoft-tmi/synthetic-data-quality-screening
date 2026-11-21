"""Checks for src/prepare_chat_data.py. Run from the repo root with:
    python -m unittest discover -s tests -t .
"""

import unittest

from src.prepare_chat_data import build_records, next_number, parse_pasted_text

MESSY_PASTE = '''Sure! Here are your examples:

```json
{"instruction": "Name two primary colors.", "input": "", "output": "Red and blue."},
{"instruction": "Summarize the text.", "input": "Cats sleep a lot.", "output": "Cats sleep much."}
```

{
  "instruction": "Add 2 and 3.",
  "output": "5"
}
{"instruction": "Missing an output"}
{this is not valid json}
Hope that helps!
'''


class ParseTests(unittest.TestCase):
    def test_keeps_valid_examples_and_ignores_chatter(self):
        examples, problems = parse_pasted_text(MESSY_PASTE)
        self.assertEqual([e["instruction"] for e in examples],
                         ["Name two primary colors.", "Summarize the text.", "Add 2 and 3."])
        self.assertEqual(examples[2]["input"], "")  # a missing input becomes an empty string

    def test_reports_the_lines_it_skipped(self):
        _, problems = parse_pasted_text(MESSY_PASTE)
        reasons = [reason for _, reason in problems]
        self.assertIn("missing or empty 'output'", reasons)
        self.assertIn("not valid JSON", reasons)


class BuildTests(unittest.TestCase):
    def test_ids_and_meta(self):
        records = build_records([{"instruction": "a", "input": "", "output": "b"}], "sam", "magpie", "chatbot", start=7)
        self.assertEqual(records[0]["id"], "sam-0007")
        self.assertEqual(records[0]["meta"]["method"], "magpie")
        self.assertEqual(records[0]["meta"]["route"], "chat")

    def test_numbering_continues_after_existing_ids(self):
        existing = [{"id": "sam-0001"}, {"id": "sam-0012"}, {"id": "other-0099"}]
        self.assertEqual(next_number(existing, "sam"), 13)
        self.assertEqual(next_number([], "sam"), 1)


if __name__ == "__main__":
    unittest.main()
