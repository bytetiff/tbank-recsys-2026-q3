from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .config import Config, DAY
from .protocol import history

AGGREGATES = ("user_count", "item_count", "pair_count", "user_recency", "item_recency", "pair_recency", "user_missing", "item_missing", "pair_missing", "item_share", "user_empty", "available_period")


@dataclass(frozen=True, order=True)
class FeatureSpec:
    window: int | None
    aggregate: str
    event_type: str = "all"

    def __post_init__(self):
        if self.event_type != "all":
            raise ValueError("The first grammar supports event_type=all only")
        if self.aggregate not in AGGREGATES + ("category_count",):
            raise ValueError("Unknown aggregate")
        if self.window is not None and (not isinstance(self.window, int) or self.window <= 0):
            raise ValueError("Window must be positive integer days or None")

    @property
    def group(self) -> str:
        return "window_all" if self.window is None else f"window_{self.window:03d}d"

    @property
    def name(self) -> str:
        return f"{self.group}__{self.event_type}__{self.aggregate}"


def grammar(windows: list[int | None], categories: bool = False) -> list[FeatureSpec]:
    if any(w is not None and (not isinstance(w, int) or w <= 0) for w in windows):
        raise ValueError("Invalid grammar window")
    aggregates = AGGREGATES + (("category_count",) if categories else ())
    return list(dict.fromkeys(FeatureSpec(window, aggregate) for window in windows for aggregate in aggregates))


def feature_groups(specs: list[FeatureSpec]) -> dict[str, list[str]]:
    return {group: [spec.name for spec in specs if spec.group == group] for group in sorted({spec.group for spec in specs})}


def compute_features(events: pl.DataFrame, rows: pl.DataFrame, q: int, specs: list[FeatureSpec], config: Config, categories: dict[int, tuple[str, ...]] | None = None) -> pl.DataFrame:
    past = history(events, q)
    output = rows.select("user_id", "item_id").with_row_index("_position")
    for window in dict.fromkeys(spec.window for spec in specs):
        current_specs = [spec for spec in specs if spec.window == window]
        window_past = history(past, q, None if window is None else window * DAY)
        values = output.select("_position", "user_id", "item_id")
        for entity, keys in (("user", ["user_id"]), ("item", ["item_id"]), ("pair", ["user_id", "item_id"])):
            aggregates = window_past.group_by(keys).agg(pl.len().alias(f"{entity}_count"), pl.col("timestamp").max().alias(f"_{entity}_last"), pl.col("timestamp").min().alias(f"_{entity}_first"))
            values = values.join(aggregates, on=keys, how="left", maintain_order="left").with_columns(
                pl.col(f"{entity}_count").fill_null(0),
                pl.col(f"_{entity}_last").is_null().cast(pl.Float32).alias(f"{entity}_missing"),
                (q - pl.col(f"_{entity}_last")).fill_null(config["missing_recency_seconds"]).alias(f"{entity}_recency"),
            )
        values = values.with_columns(
            pl.when(pl.col("user_count") > 0).then(pl.col("pair_count") / pl.col("user_count")).otherwise(0.0).alias("item_share"),
            (pl.col("user_count") == 0).cast(pl.Float32).alias("user_empty"),
            (q - pl.col("_user_first")).fill_null(0).alias("available_period"),
        )
        if any(spec.aggregate == "category_count" for spec in current_specs):
            if categories is None:
                raise ValueError("Category features require explicit static metadata")
            user_events = {user: frame["item_id"].to_list() for (user,), frame in window_past.partition_by("user_id", as_dict=True).items()}
            # Each historical event is counted once when any category overlaps.
            category_counts = []
            cache = {}
            for user, item in rows.select("user_id", "item_id").iter_rows():
                category = categories.get(item, ("unknown",))
                key = (user, category)
                if key not in cache:
                    wanted = set(category)
                    cache[key] = sum(bool(wanted.intersection(categories.get(old_item, ("unknown",)))) for old_item in user_events.get(user, []))
                category_counts.append(cache[key])
            values = values.with_columns(pl.Series("category_count", category_counts))
        output = output.hstack(values.select(pl.col(spec.aggregate).cast(pl.Float32).alias(spec.name) for spec in current_specs))
    return output.drop("_position").select("user_id", "item_id", *[spec.name for spec in specs])


def reference_features(events: pl.DataFrame, rows: pl.DataFrame, q: int, specs: list[FeatureSpec], config: Config, categories: dict[int, tuple[str, ...]] | None = None) -> pl.DataFrame:
    records = events.to_dicts()
    results = []
    for user, item in rows.select("user_id", "item_id").iter_rows():
        result = {"user_id": user, "item_id": item}
        for spec in specs:
            past = [e for e in records if e["timestamp"] < q and (spec.window is None or e["timestamp"] >= q - spec.window * DAY)]
            entities = {"user": [e for e in past if e["user_id"] == user], "item": [e for e in past if e["item_id"] == item], "pair": [e for e in past if e["user_id"] == user and e["item_id"] == item]}
            aggregate = spec.aggregate
            entity, _, operator = aggregate.partition("_")
            if operator == "count":
                if entity == "category":
                    if categories is None:
                        raise ValueError("Missing categories")
                    wanted = set(categories.get(item, ("unknown",)))
                    value = sum(bool(wanted.intersection(categories.get(e["item_id"], ("unknown",)))) for e in entities["user"])
                else:
                    value = len(entities[entity])
            elif operator == "missing":
                value = int(not entities[entity])
            elif operator == "recency":
                value = q - max(e["timestamp"] for e in entities[entity]) if entities[entity] else config["missing_recency_seconds"]
            elif aggregate == "item_share":
                value = len(entities["pair"]) / len(entities["user"]) if entities["user"] else 0
            elif aggregate == "user_empty":
                value = int(not entities["user"])
            elif aggregate == "available_period":
                value = q - min(e["timestamp"] for e in entities["user"]) if entities["user"] else 0
            else:
                raise ValueError(f"Unknown aggregate: {aggregate}")
            result[spec.name] = np.float32(value)
        results.append(result)
    if not results:
        return pl.DataFrame(schema={"user_id": pl.Int64, "item_id": pl.Int64, **{spec.name: pl.Float32 for spec in specs}})
    return pl.DataFrame(results).select("user_id", "item_id", *[spec.name for spec in specs])
