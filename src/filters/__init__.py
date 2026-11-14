"""Registry of filtering methods. Add new filters here so filter_data.py can find them."""

from src.filters.bilevel import BilevelFilter
from src.filters.difficulty import DifficultyFilter
from src.filters.llm_judge import LLMJudgeFilter
from src.filters.multidim import MultiDimFilter
from src.filters.shapley import ShapleyFilter
from src.filters.uncertainty import UncertaintyFilter

FILTERS = {
    cls.name: cls
    for cls in (
        LLMJudgeFilter,
        UncertaintyFilter,
        ShapleyFilter,
        MultiDimFilter,
        DifficultyFilter,
        BilevelFilter,
    )
}


def get_filter(name):
    if name not in FILTERS:
        raise KeyError(f"Unknown filter '{name}'. Options: {', '.join(sorted(FILTERS))}")
    return FILTERS[name]
