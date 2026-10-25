from copy import deepcopy

import pytest

from tlab_recsys.config import Config
from tlab_recsys.evaluation import bootstrap_contrast
from tlab_recsys.selection import greedy_select


@pytest.mark.parametrize("path,value", [
    (("cohort_size",), 1.5), (("recent_candidates",), True), (("cohort_seed",), -1),
    (("model_seeds",), [17, 17]), (("model_seeds",), [1.5]), (("model_seeds",), [True]),
    (("model_seeds",), [-1]), (("model_seeds",), [2**32]),
    (("search", "max_fits"), 0.5), (("search", "max_groups"), 1.5),
    (("search", "min_improvement"), float("nan")), (("search", "min_improvement"), float("inf")),
    (("bootstrap", "samples"), 0), (("bootstrap", "samples"), 2.5), (("bootstrap", "seed"), -1),
    (("catboost", "depth"), []), (("catboost", "depth"), [4.5]), (("catboost", "depth"), [17]),
    (("catboost", "depth"), [4, 4]), (("catboost", "learning_rate"), [0]),
    (("catboost", "learning_rate"), [float("nan")]), (("catboost", "learning_rate"), [1.1]),
    (("catboost", "l2_leaf_reg"), [-1]), (("catboost", "l2_leaf_reg"), [float("inf")]),
    (("catboost", "max_iterations"), 1.5), (("catboost", "early_stopping_rounds"), 0),
    (("catboost", "thread_count"), 1.5), (("missing_recency_seconds",), float("inf")),
    (("target_played_ratio_gt",), float("nan")),
])
def test_invalid_configuration_is_rejected_at_construction(path, value):
    values = deepcopy(Config.load().values)
    current = values
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value
    with pytest.raises(ValueError):
        Config(values)


@pytest.mark.parametrize("budget", [0.5, True, -1, float("inf")])
def test_direct_search_rejects_noninteger_budget_before_scoring(budget):
    def forbidden(_):
        pytest.fail("Invalid search started scoring")
    with pytest.raises(ValueError):
        greedy_select(["a"], forbidden, budget, 1, 0.0001)
    with pytest.raises(ValueError):
        greedy_select(["a"], forbidden, 1, budget, 0.0001)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_direct_search_rejects_invalid_threshold_before_scoring(threshold):
    with pytest.raises(ValueError):
        greedy_select([], lambda _: pytest.fail("Scorer called"), 1, 1, threshold)


@pytest.mark.parametrize("samples,seed", [(0, 1), (-1, 1), (1.5, 1), (True, 1), (2, -1), (2, 1.5), (2, True), (2, 2**32)])
def test_direct_bootstrap_validates_counts_and_seed_even_for_empty_results(samples, seed):
    with pytest.raises(ValueError):
        bootstrap_contrast([], samples=samples, seed=seed)
