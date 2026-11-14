"""Method 4: Multi-dimensional quality assessment (for example DEITA).

Idea: score several dimensions at once, such as quality, complexity and diversity
(diversity usually comes from embeddings), then select a high-quality, complex,
non-redundant subset.

TODO: implement `filter`.
"""

from src.filters.base import BaseFilter


class MultiDimFilter(BaseFilter):
    name = "multidim"

    def filter(self, examples):
        raise NotImplementedError("TODO: implement multi-dimensional filtering")
