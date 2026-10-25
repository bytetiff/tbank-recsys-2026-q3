"""Leakage and correctness checks on synthetic events."""
import numpy as np
import polars as pl
import pytest

from strongfeat.common import DAY, HORIZON, candidates, evaluate, make_slice, history
from strongfeat.features import Spec, compute, grammar, reference, parse
from strongfeat.manual import MANUAL_SPECS, extras

CONFIG = {"candidates": {"history": 5, "recent": 3, "artist": 2, "popular": 3}}
Q = 200 * DAY


def synthetic(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    types = rng.choice(["listen", "listen", "listen", "like", "dislike", "unlike", "undislike"], n)
    ratio = rng.integers(0, 120, n).astype(np.float32)
    item = rng.integers(0, 40, n)
    frame = pl.DataFrame({
        "row_id": np.arange(n), "user_id": rng.integers(0, 6, n), "item_id": item,
        "timestamp": rng.integers(0, 230 * DAY, n), "event_type": types,
        "is_organic": rng.integers(0, 2, n).astype(np.int8), "played_ratio_pct": ratio,
        "track_length_seconds": (item * 10 + 100).astype(np.float32), "artist_id": item % 7, "album_id": item % 11,
    }).with_columns(((pl.col("event_type") == "listen") & (pl.col("played_ratio_pct") > 50)).alias("is_target"))
    return frame.sort("timestamp")


def rows_for(events):
    rows = candidates(events, sorted(events["user_id"].unique().to_list()), Q, CONFIG)
    return rows.with_columns((pl.col("item_id") % 7).alias("artist_id"), (pl.col("item_id") % 11).alias("album_id"))


def test_grammar_size_and_parse():
    specs = grammar()
    assert len(specs) == len(set(specs)) > 900
    assert all(parse(s.name) == s for s in specs)
    assert set(MANUAL_SPECS) <= set(specs)


def test_engine_matches_naive_reference():
    events = synthetic()
    rows = rows_for(events)
    sample = [s for s in grammar() if s.window in (7, 30, None)][::7] + MANUAL_SPECS
    sample = list(dict.fromkeys(sample))
    fast = compute(history(events, Q), rows, Q, sample)
    records = events.to_dicts()
    for i, row in enumerate(rows.head(25).to_dicts()):
        for spec in sample:
            expected = reference(records, row, Q, spec)
            got = fast[spec.name][i]
            if np.isnan(expected):
                assert got is None or np.isnan(got), spec.name
            else:
                assert got == pytest.approx(expected, rel=1e-5), spec.name


def test_future_events_do_not_change_inputs():
    events = synthetic()
    rows = rows_for(events)
    past = events.filter(pl.col("timestamp") < Q)
    rng = np.random.default_rng(1)
    future = events.filter(pl.col("timestamp") >= Q)
    perm = rng.permutation(future["item_id"].to_numpy())
    scrambled = future.with_columns(pl.Series("item_id", perm), pl.Series("artist_id", perm % 7), pl.Series("album_id", perm % 11),
                                    pl.Series("event_type", rng.permutation(future["event_type"].to_numpy())))
    other = pl.concat([past, scrambled]).sort("timestamp")
    rows2 = rows_for(other)
    assert rows.drop("artist_id", "album_id").equals(rows2.drop("artist_id", "album_id"))
    specs = grammar()
    assert compute(history(events, Q), rows, Q, specs).equals(compute(history(other, Q), rows2, Q, specs))
    assert extras(history(events, Q), rows, Q).equals(extras(history(other, Q), rows2, Q))


def test_engine_rejects_future_history():
    events = synthetic()
    with pytest.raises(ValueError):
        compute(events, rows_for(events), Q, MANUAL_SPECS[:2])
    with pytest.raises(ValueError):
        extras(events, rows_for(events), Q)


def test_labels_come_only_from_horizon():
    events = synthetic()
    users = sorted(events["user_id"].unique().to_list())
    slice_ = make_slice(events, users, "x", Q, CONFIG)
    horizon = events.filter((pl.col("timestamp") >= Q) & (pl.col("timestamp") < Q + HORIZON) & pl.col("is_target"))
    positives = set(horizon.select("user_id", "item_id").iter_rows())
    for user, item, label in slice_.rows.select("user_id", "item_id", "label").iter_rows():
        assert label == int((user, item) in positives)
    assert sum(slice_.targets.values()) == len(positives)


def test_ndcg_definition():
    events = synthetic()
    users = sorted(events["user_id"].unique().to_list())
    slice_ = make_slice(events, users, "x", Q, CONFIG)
    labels = slice_.rows["label"].to_numpy().astype(float)
    perfect = evaluate(slice_, labels * 0.9 + 0.05)
    for row in perfect["per_user"]:
        n = row["target_count"]
        hits = slice_.rows.filter(pl.col("user_id") == row["user_id"])["label"].sum()
        d = 1 / np.log2(np.arange(2, 12))
        assert row["ndcg_10"] == pytest.approx(d[: min(hits, 10)].sum() / d[: min(n, 10)].sum())
