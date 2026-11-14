"""Method 3: Shapley-based filtering (for example SHED).

Idea: estimate each example's contribution to model performance using Shapley-value
approximations, then remove the low-value examples. Exact values need many training
runs, so use an approximation and a small model.

TODO: implement `filter`.
"""

from src.filters.base import BaseFilter


class ShapleyFilter(BaseFilter):
    name = "shapley"

    def filter(self, examples):
        raise NotImplementedError("TODO: implement Shapley-based filtering")
