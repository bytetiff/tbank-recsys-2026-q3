import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from tlab_recsys.config import Config, DAY
from tlab_recsys.data import adapt_kion, adapt_yambda, cohort_order, complete_cohort, validate_events
from tlab_recsys.protocol import candidate_rows, cutoffs, history, make_slice, target_sets


def test_p1_p4_half_open_boundaries(events):
    q = 3 * DAY
    assert set(history(events, q)["row_id"]) == {0, 1, 2, 3, 7}
    assert set(history(events, q, DAY)["row_id"]) == {2, 3}
    assert target_sets(events, [1, 2], q, DAY) == {1: {30}, 2: {30}}
    assert target_sets(events, [1], q + DAY, DAY) == {1: {99}}
    assert not set(history(events, q)["row_id"]) & {4, 5}


@pytest.mark.parametrize("invalid", [0, -1, 1.5])
def test_invalid_time_bounds(events, invalid):
    with pytest.raises(ValueError):
        history(events, 1, invalid)
    with pytest.raises(ValueError):
        cutoffs(100, invalid)


def test_p3_complete_histories_keep_duplicate_contents(events):
    selected = complete_cohort(events, [1])
    assert selected["row_id"].to_list() == [0, 1, 2, 5, 6, 7]
    assert selected.height == events.filter(pl.col("user_id") == 1).height
    validate_events(selected)
    assert complete_cohort(events, []).is_empty()
    with pytest.raises(ValueError):
        validate_events(pl.concat([events, events.head(1)]))


def test_source_time_adapters_and_targets():
    kion = adapt_kion(pl.DataFrame({"user_id": [1, 2], "item_id": [2, 3], "last_watch_dt": ["2021-03-13", "2021-03-13"]}))
    assert kion["timestamp"][0] == kion["timestamp"][1] == 1615593600
    yambda = adapt_yambda(pl.DataFrame({"uid": [1] * 4, "item_id": [2] * 4, "timestamp": [5, 5, 10, 15], "event_type": ["listen", "listen", "listen", "like"], "played_ratio_pct": [50, 51, None, 100]}))
    assert yambda["timestamp"].to_list() == [5, 5, 10, 15]
    assert yambda["is_target"].to_list() == [False, True, False, False]


def test_p2_candidates_ignore_future_and_no_forced_positives(events, config):
    q = 3 * DAY
    past = history(events, q)
    original, catalog = candidate_rows(events, [1, 2], q, "yambda", config)
    truncated, _ = candidate_rows(past, [1, 2], q, "yambda", config)
    assert_frame_equal(original, truncated)
    assert catalog == {10, 20}
    assert 30 not in original["item_id"]
    full, _ = candidate_rows(events, [1], q, "kion", config, full=True)
    assert full.is_empty()


def test_candidates_reference_ties_exclusions_and_recency(events, config):
    values = dict(config.values, popular_candidates=1, recent_candidates=1)
    tiny = Config(values)
    q = 3 * DAY
    rows, _ = candidate_rows(events, [1, 2, 3], q, "yambda", tiny)
    assert rows.filter(pl.col("user_id") == 1)["item_id"].to_list() == [10, 20]
    assert rows.filter(pl.col("user_id") == 2)["item_id"].to_list() == [10, 20]
    assert rows.filter(pl.col("user_id") == 3)["item_id"].to_list() == [10]
    kion, _ = candidate_rows(events, [2], q, "kion", tiny)
    assert kion["item_id"].to_list() == [10]


def test_cohort_order_is_independent_of_model_seed_and_future(events, config):
    eligible = history(events, 3 * DAY)["user_id"].to_list()
    assert cohort_order(eligible, 1729) == cohort_order(list(reversed(eligible)), 1729)
    assert cohort_order(eligible, config["cohort_seed"]) == cohort_order(eligible, Config(dict(config.values, model_seeds=[999]))["cohort_seed"])
    assert [q for _, q in cutoffs(6 * DAY, DAY)] == [0, DAY, 2 * DAY, 3 * DAY, 4 * DAY, 5 * DAY]
