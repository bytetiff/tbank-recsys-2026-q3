from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

DAY = 86400
EVALUATION_BATCH_USERS = 5
ROOT = Path(__file__).resolve().parents[2]


def require_integer(value: object, name: str, minimum: int = 0, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in the allowed range")


def require_finite(value: object, name: str, minimum: float = 0, maximum: float | None = None, strict_minimum: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < minimum or (strict_minimum and value == minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be finite and in the allowed range")


def validate_search(max_fits: int, max_groups: int, min_improvement: float) -> None:
    require_integer(max_fits, "max_fits")
    require_integer(max_groups, "max_groups")
    require_finite(min_improvement, "min_improvement", maximum=1)


def validate_bootstrap(samples: int, seed: int) -> None:
    require_integer(samples, "bootstrap samples", minimum=1)
    require_integer(seed, "bootstrap seed", maximum=2**32 - 1)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Config:
    values: dict

    def __post_init__(self):
        self.validate()

    @classmethod
    def load(cls, profile: str = "pilot", path: str | Path | None = None) -> "Config":
        values = json.loads(Path(path or ROOT / "configs" / f"{profile}.json").read_text(encoding="utf-8"))
        return cls(values)

    def __getitem__(self, key: str):
        return self.values[key]

    @property
    def digest(self) -> str:
        return canonical_hash(self.values)

    def validate(self) -> None:
        v = self.values
        for key in ("cohort_size", "horizon_days", "popular_candidates", "full_catalog_users"):
            require_integer(v[key], key, minimum=1)
        require_integer(v["recent_candidates"], "recent_candidates")
        require_integer(v["cohort_seed"], "cohort_seed", maximum=2**32 - 1)
        seeds = v["model_seeds"]
        if not isinstance(seeds, (list, tuple)) or not seeds:
            raise ValueError("model_seeds must be a nonempty sequence")
        for seed in seeds:
            require_integer(seed, "model seed", maximum=2**32 - 1)
        if len(set(seeds)) != len(seeds):
            raise ValueError("model_seeds must be unique")
        if v["metadata_mode"] not in ("none", "categories"):
            raise ValueError("metadata_mode must be none or categories")
        for key in ("manual_windows_days", "generated_windows_days"):
            if not isinstance(v[key], (list, tuple)) or not v[key]:
                raise ValueError("Feature windows must be a nonempty sequence")
            for window in v[key]:
                if window is not None:
                    require_integer(window, key, minimum=1)
        s = v["search"]
        validate_search(**s)
        validate_bootstrap(**v["bootstrap"])
        require_finite(v["missing_recency_seconds"], "missing_recency_seconds")
        require_finite(v["target_played_ratio_gt"], "target_played_ratio_gt")
        if not isinstance(v["ablation_manual_selected"], bool):
            raise ValueError("ablation_manual_selected must be boolean")
        c = v["catboost"]
        require_integer(c["thread_count"], "thread_count", minimum=1, maximum=8)
        require_integer(c["max_iterations"], "max_iterations", minimum=1, maximum=500)
        require_integer(c["early_stopping_rounds"], "early_stopping_rounds", minimum=1, maximum=c["max_iterations"])
        for key in ("depth", "learning_rate", "l2_leaf_reg"):
            grid = c[key]
            if not isinstance(grid, (list, tuple)) or not grid:
                raise ValueError(f"{key} grid must be nonempty")
            for value in grid:
                if key == "depth":
                    require_integer(value, "depth", minimum=1, maximum=16)
                else:
                    require_finite(value, key, maximum=1 if key == "learning_rate" else None, strict_minimum=key == "learning_rate")
            if len(set(grid)) != len(grid):
                raise ValueError(f"{key} grid must contain unique values")
