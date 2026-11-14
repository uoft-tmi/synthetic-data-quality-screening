"""Method 2: Uncertainty-aware filtering (for example SelectIT).

Idea: measure how confident a model is in its own quality rating, using
token-level, sentence-level and model-level uncertainty. Uncertain examples rank lower.
Needs open weights so you can read the model's token probabilities.

TODO: implement `score` so it returns a confidence-based quality number.
"""

from src.filters.base import ScoreFilter


class UncertaintyFilter(ScoreFilter):
    name = "uncertainty"

    def __init__(self, threshold=0.5):
        super().__init__(threshold)

    def score(self, example):
        raise NotImplementedError("TODO: implement uncertainty-aware scoring")
