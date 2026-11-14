"""Method 5: Difficulty-based filtering (for example IFD / Cherry LLM, Superfiltering).

Idea: estimate how hard each example is for a model (loss, perplexity, or the
Instruction-Following Difficulty score), then keep examples in a useful difficulty
range. Superfiltering uses a small model such as GPT-2 to score data for larger models.

TODO: implement `filter`.
"""

from src.filters.base import BaseFilter


class DifficultyFilter(BaseFilter):
    name = "difficulty"

    def filter(self, examples):
        raise NotImplementedError("TODO: implement difficulty-based filtering")
