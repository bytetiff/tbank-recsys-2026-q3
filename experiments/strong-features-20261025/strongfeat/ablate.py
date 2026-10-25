"""Ablation of manual feature groups on the selection slice (seed 17). Test is not read."""
from __future__ import annotations

import sys
from pathlib import Path

from .common import EXPERIMENT, ResourceMeasure, evaluate, write_json
from .features import parse
from .run import Run


def group_of(name: str) -> set[str]:
    if name.startswith("src_"):
        return {"source_flags"}
    if name.startswith("x_"):
        return {"extras"}
    spec = parse(name)
    groups = {"indirect"} if spec.key in ("artist", "album", "user_artist", "user_album") else set()
    if spec.filt in ("like", "dislike", "skip", "organic") or spec.agg in ("ratio_skip", "ratio_like"):
        groups.add("event_types")
    if spec.agg in ("mean_ratio", "ratio_full", "ratio_skip", "ratio_like", "share", "nuniq"):
        groups.add("ratios_and_shares")
    return groups or {"base"}


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else EXPERIMENT.parents[1]
    run = Run(root, EXPERIMENT / "artifacts" / "main")
    frames = {n: run.matrix("manual", n) for n in ("train0", "train1", "train2", "dev", "selection")}
    sel = run.load_slice("selection")
    columns = frames["dev"].columns
    results = {}
    for drop in ("none", "extras", "indirect", "event_types", "ratios_and_shares", "source_flags"):
        keep = [c for c in columns if drop not in group_of(c)]
        with ResourceMeasure() as m:
            model = run.fit_variant("manual", 17, frames={n: f.select(keep) for n, f in frames.items()})
            scores = model.predict_proba(run.pool(frames["selection"].select(keep)))[:, 1]
        metrics = evaluate(sel, scores)["metrics"]
        results[drop] = {"columns": len(keep), "selection_ndcg_10": metrics["ndcg_10"], "roc_auc": metrics["roc_auc"], **m.result}
        print(drop, results[drop], flush=True)
    write_json(EXPERIMENT / "reports" / "manual-ablation-selection.json", results)


if __name__ == "__main__":
    main()
