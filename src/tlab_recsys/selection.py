from __future__ import annotations

import math
from collections.abc import Callable

from .config import validate_search


def greedy_select(groups: list[str], scorer: Callable[[tuple[str, ...]], float | None], max_fits: int, max_groups: int, min_improvement: float) -> dict:
    validate_search(max_fits, max_groups, min_improvement)
    selected: tuple[str, ...] = ()
    initial = scorer(selected)
    if initial is None or not math.isfinite(initial):
        raise ValueError("Initial core selection metric must be defined and finite")
    best_score = float(initial)
    trace = [{"groups": [], "score": best_score, "kind": "initial_core"}]
    fits = 0
    universe = sorted(set(groups))
    while fits < max_fits and len(selected) < max_groups:
        candidates = []
        for group in universe:
            if group in selected:
                continue
            if fits >= max_fits:
                break
            proposal = tuple(sorted((*selected, group)))
            score = scorer(proposal)
            fits += 1
            valid = score is not None and math.isfinite(score)
            trace.append({"groups": list(proposal), "score": float(score) if valid else None, "kind": "additional_fit"})
            if valid:
                candidates.append((float(score), group, proposal))
        if not candidates:
            break
        winner = min(candidates, key=lambda row: (-row[0], row[1]))
        if winner[0] <= best_score or winner[0] - best_score < min_improvement:
            break
        best_score, _, selected = winner
    return {"groups": list(selected), "score": best_score, "additional_fits": fits, "initial_core_fits": 1, "trace": trace}
