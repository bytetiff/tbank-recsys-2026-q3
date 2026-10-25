import polars as pl
from polars.testing import assert_frame_equal
import pytest

from tlab_recsys.config import DAY
from tlab_recsys.features import FeatureSpec, compute_features, feature_groups, grammar, reference_features
from tlab_recsys.protocol import history


def test_p1_p2_optimized_matches_independent_reference(events, config):
    rows = pl.DataFrame({"user_id": [1, 2, 3, 1], "item_id": [10, 20, 99, 30]})
    specs = grammar([1, 3, None])
    expected = reference_features(events, rows, 3 * DAY, specs, config)
    actual = compute_features(events, rows, 3 * DAY, specs, config)
    assert_frame_equal(actual, expected, check_dtypes=False, rel_tol=1e-6)
    assert_frame_equal(actual, compute_features(history(events, 3 * DAY), rows, 3 * DAY, specs, config))
    assert actual["window_001d__all__pair_count"].to_list() == [0, 1, 0, 0]
    assert actual["window_all__all__pair_count"].to_list() == [3, 1, 0, 0]


def test_empty_history_and_unknown_category(events, config):
    rows = pl.DataFrame({"user_id": [1, 9], "item_id": [10, 100]})
    specs = grammar([None], categories=True)
    categories = {10: ("a", "b"), 20: ("b",), 30: ("c",)}
    actual = compute_features(events, rows, 3 * DAY, specs, config, categories)
    expected = reference_features(events, rows, 3 * DAY, specs, config, categories)
    assert_frame_equal(actual, expected, check_dtypes=False)
    empty = compute_features(events.head(0), rows, 0, specs, config, categories)
    assert empty["window_all__all__user_count"].to_list() == [0, 0]
    assert empty["window_all__all__pair_missing"].to_list() == [1, 1]
    assert empty["window_all__all__item_share"].to_list() == [0, 0]


def test_p5_finite_grammar_and_stable_deduplication():
    specs = grammar([7, None, 7])
    assert len(specs) == 24
    assert list(feature_groups(specs)) == ["window_007d", "window_all"]
    for kwargs in ({"window": 1, "aggregate": "user_count", "event_type": "listen"}, {"window": 0, "aggregate": "user_count"}, {"window": 1, "aggregate": "exec"}):
        with pytest.raises(ValueError):
            FeatureSpec(**kwargs)
