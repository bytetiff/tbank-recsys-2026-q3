"""Feature grammar: key x event filter x time window x aggregation.

A feature is the SQL query

    SELECT <agg> FROM events
    WHERE timestamp < q AND timestamp >= q - window AND <filter>
    GROUP BY <key>

joined to the candidate rows on the key. The manual baseline and the automatic search use this same
engine, so they differ only in which specs (and a few hand-made extras) they use.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .common import DAY

KEYS = {
    "user": ("user_id",), "item": ("item_id",), "artist": ("artist_id",), "album": ("album_id",),
    "user_item": ("user_id", "item_id"), "user_artist": ("user_id", "artist_id"), "user_album": ("user_id", "album_id"),
}
FILTERS = ("all", "listen", "full", "skip", "like", "dislike", "organic")
WINDOWS = (1, 3, 7, 14, 30, 90, None)
AGGS = ("count", "recency", "mean_ratio", "nuniq", "ratio_full", "ratio_skip", "ratio_like", "share")
NUNIQ_KEYS = ("user", "artist", "album", "user_artist", "user_album")
SHARE_KEYS = ("user_item", "user_artist", "user_album")


def filter_expr(name: str) -> pl.Expr:
    listen = pl.col("event_type") == "listen"
    return {
        "all": pl.col("timestamp").is_not_null(), "listen": listen, "full": listen & (pl.col("played_ratio_pct") > 50),
        "skip": listen & (pl.col("played_ratio_pct") < 20), "like": pl.col("event_type") == "like",
        "dislike": pl.col("event_type") == "dislike", "organic": listen & (pl.col("is_organic") == 1),
    }[name]


@dataclass(frozen=True, order=True)
class Spec:
    key: str
    filt: str
    window: int | None
    agg: str

    def __post_init__(self):
        if self.key not in KEYS or self.filt not in FILTERS or self.agg not in AGGS:
            raise ValueError(f"Unknown spec {self}")
        if self.window is not None and (not isinstance(self.window, int) or self.window <= 0):
            raise ValueError("Window must be positive days or None")
        if self.agg == "mean_ratio" and self.filt != "listen":
            raise ValueError("mean_ratio is defined on listens")
        if self.agg.startswith("ratio_") and self.filt != "listen":
            raise ValueError("ratios use listens as denominator")
        if self.agg == "nuniq" and (self.filt != "full" or self.key not in NUNIQ_KEYS):
            raise ValueError("nuniq is defined for full listens of non-item keys")
        if self.agg == "share" and (self.filt != "full" or self.key not in SHARE_KEYS):
            raise ValueError("share is defined for full listens of user x object keys")

    @property
    def name(self) -> str:
        return f"{self.key}__{self.filt}__{'all' if self.window is None else f'{self.window}d'}__{self.agg}"

    @property
    def family(self) -> str:
        return f"{self.key}__{self.filt}__{self.agg}"


def grammar() -> list[Spec]:
    specs = []
    for key in KEYS:
        for window in WINDOWS:
            for filt in FILTERS:
                specs += [Spec(key, filt, window, "count"), Spec(key, filt, window, "recency")]
            specs.append(Spec(key, "listen", window, "mean_ratio"))
            specs += [Spec(key, "listen", window, agg) for agg in ("ratio_full", "ratio_skip", "ratio_like")]
            if key in NUNIQ_KEYS:
                specs.append(Spec(key, "full", window, "nuniq"))
            if key in SHARE_KEYS:
                specs.append(Spec(key, "full", window, "share"))
    return specs


def parse(name: str) -> Spec:
    key, filt, window, agg = name.split("__")
    return Spec(key, filt, None if window == "all" else int(window[:-1]), agg)


def _needed_counts(specs: list[Spec]) -> set[str]:
    needed = set()
    for spec in specs:
        if spec.agg == "count":
            needed.add(spec.filt)
        elif spec.agg.startswith("ratio_"):
            needed |= {"listen", {"ratio_full": "full", "ratio_skip": "skip", "ratio_like": "like"}[spec.agg]}
        elif spec.agg == "share":
            needed.add("full")
    return needed


def compute(past: pl.DataFrame, rows: pl.DataFrame, q: int, specs: list[Spec]) -> pl.DataFrame:
    """Return a frame aligned with rows (same order) with one Float32 column per spec.

    past must already be restricted to timestamp < q; this is re-checked here because it is the
    only leakage guard that matters for the features.
    """
    if past.height and past["timestamp"].max() >= q:
        raise ValueError("Feature history contains events at or after the cutoff")
    base = rows.select("user_id", "item_id", "artist_id", "album_id").with_row_index("_pos")
    out = {}
    by_group: dict[tuple[str, int | None], list[Spec]] = {}
    for spec in specs:
        by_group.setdefault((spec.key, spec.window), []).append(spec)
    user_full_cache: dict[int | None, pl.DataFrame] = {}
    for (key, window), group in by_group.items():
        keys = list(KEYS[key])
        wpast = past if window is None else past.filter(pl.col("timestamp") >= q - window * DAY)
        exprs = [filter_expr(f).sum().cast(pl.Float64).alias(f"c_{f}") for f in sorted(_needed_counts(group))]
        exprs += [(q - pl.col("timestamp").filter(filter_expr(s.filt)).max()).cast(pl.Float64).alias(f"r_{s.filt}") for s in group if s.agg == "recency"]
        if any(s.agg == "mean_ratio" for s in group):
            exprs.append(pl.col("played_ratio_pct").filter(filter_expr("listen")).mean().cast(pl.Float64).alias("mean_ratio"))
        if any(s.agg == "nuniq" for s in group):
            exprs.append(pl.col("item_id").filter(filter_expr("full")).n_unique().cast(pl.Float64).alias("nuniq"))
        agg = wpast.group_by(keys).agg(exprs)
        joined = base.join(agg, on=keys, how="left").sort("_pos")
        if any(s.agg == "share" for s in group):
            if window not in user_full_cache:
                user_full_cache[window] = wpast.group_by("user_id").agg(filter_expr("full").sum().cast(pl.Float64).alias("u_full"))
            joined = joined.join(user_full_cache[window], on="user_id", how="left").sort("_pos")
        for spec in group:
            if spec.agg == "count":
                expr = pl.col(f"c_{spec.filt}").fill_null(0)
            elif spec.agg == "recency":
                expr = pl.col(f"r_{spec.filt}")
            elif spec.agg == "mean_ratio":
                expr = pl.col("mean_ratio")
            elif spec.agg == "nuniq":
                expr = pl.col("nuniq").fill_null(0)
            elif spec.agg.startswith("ratio_"):
                numerator = {"ratio_full": "c_full", "ratio_skip": "c_skip", "ratio_like": "c_like"}[spec.agg]
                expr = pl.when(pl.col("c_listen") > 0).then(pl.col(numerator) / pl.col("c_listen")).otherwise(None)
            else:  # share
                expr = pl.when(pl.col("u_full") > 0).then(pl.col("c_full").fill_null(0) / pl.col("u_full")).otherwise(None)
            out[spec.name] = joined.select(expr.cast(pl.Float32)).to_series()
    return pl.DataFrame({spec.name: out[spec.name] for spec in specs}) if specs else pl.DataFrame()


def reference(past_records: list[dict], row: dict, q: int, spec: Spec) -> float:
    """Naive per-row implementation used only by tests."""
    lo = -np.inf if spec.window is None else q - spec.window * DAY
    keys = KEYS[spec.key]

    def passes(e, filt):
        listen = e["event_type"] == "listen"
        return {"all": True, "listen": listen, "full": listen and e["played_ratio_pct"] > 50, "skip": listen and e["played_ratio_pct"] < 20,
                "like": e["event_type"] == "like", "dislike": e["event_type"] == "dislike", "organic": listen and e["is_organic"] == 1}[filt]

    rel = [e for e in past_records if e["timestamp"] < q and e["timestamp"] >= lo and all(e[k] == row[k] for k in keys)]
    count = lambda f: sum(passes(e, f) for e in rel)
    if spec.agg == "count":
        return float(count(spec.filt))
    if spec.agg == "recency":
        ts = [e["timestamp"] for e in rel if passes(e, spec.filt)]
        return float(q - max(ts)) if ts else np.nan
    if spec.agg == "mean_ratio":
        values = [e["played_ratio_pct"] for e in rel if passes(e, "listen")]
        return float(np.mean(values)) if values else np.nan
    if spec.agg == "nuniq":
        return float(len({e["item_id"] for e in rel if passes(e, "full")}))
    if spec.agg.startswith("ratio_"):
        listens = count("listen")
        return count(spec.agg.split("_")[1]) / listens if listens else np.nan
    user_full = sum(passes(e, "full") for e in past_records if e["timestamp"] < q and e["timestamp"] >= lo and e["user_id"] == row["user_id"])
    return count("full") / user_full if user_full else np.nan
