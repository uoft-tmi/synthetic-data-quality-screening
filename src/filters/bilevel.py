"""Method 6: Bilevel filtering (for example SEAL, ScaleBiO).

Idea: jointly optimize a data selector (upper level) and the downstream model
(lower level). The selector is updated based on how the chosen data affects
validation performance. Needs full access to the model's gradients.

TODO: implement `filter`.
"""

from src.filters.base import BaseFilter


class BilevelFilter(BaseFilter):
    name = "bilevel"

    def filter(self, examples):
        raise NotImplementedError("TODO: implement bilevel filtering")
