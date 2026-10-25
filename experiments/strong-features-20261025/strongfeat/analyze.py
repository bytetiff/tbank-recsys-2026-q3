"""Build reports/results.md and reports/results.json from the saved run artifacts (no training)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import EXPERIMENT, write_json
from .features import parse

NAMES = {"popular": "Popular (7 дней)", "history_count": "History count (90 дней)", "old_manual": "Old manual (прежний пайплайн)",
         "manual": "Strong manual", "auto": "Auto-grammar", "auto_manual": "Auto + manual"}


def main(out: Path, final_name: str = "final.json"):
    final = json.loads((out / final_name).read_text())
    suffix = final_name.replace("final", "").replace(".json", "")
    selection = json.loads((out / "selection.json").read_text())
    screen = json.loads((out / "screen.json").read_text())
    slices = json.loads((out / "slices-summary.json").read_text())
    stages = json.loads((out / "stage-resources.json").read_text())
    rows = []
    for name, ev in final["baselines"].items():
        m = ev["metrics"]
        rows.append((name, {k: (m[k], 0.0) for k in ("ndcg_10", "recall_10", "recall_100", "roc_auc")} | {"logloss": (None, None)}, None, None, None))
    for variant, runs in final["variants"].items():
        agg = {}
        for key in ("ndcg_10", "recall_10", "recall_100", "roc_auc", "logloss"):
            values = [r["metrics"][key] for r in runs]
            agg[key] = (float(np.mean(values)), float(np.std(values)))
        fit = float(np.mean([r["fit"]["wall_seconds"] for r in runs]))
        iters = int(np.mean([r["best_iteration"] for r in runs]))
        rows.append((variant, agg, runs[0]["columns"], fit, iters))
    contrasts = final["contrasts_ndcg_10"]

    def fmt(v, digits=4):
        return "—" if v is None else f"{v:.{digits}f}"

    lines = ["| Вариант | Признаков | NDCG@10 | Recall@10 | Recall@100 | ROC-AUC | LogLoss | Δ NDCG@10 к strong manual [95% CI] | Обучение, с | Итераций |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for variant, agg, cols, fit, iters in rows:
        c = contrasts.get(f"{variant}-manual")
        delta = "0 (опора)" if variant == "manual" else f"{c['mean']:+.4f} [{c['ci95'][0]:+.4f}; {c['ci95'][1]:+.4f}]"
        nd = agg["ndcg_10"]
        lines.append(f"| {NAMES[variant]} | {cols or '—'} | {fmt(nd[0])} ± {fmt(nd[1])} | {fmt(agg['recall_10'][0])} | {fmt(agg['recall_100'][0])} | {fmt(agg['roc_auc'][0])} | {fmt(agg['logloss'][0])} | {delta} | {fmt(fit, 0) if fit else '—'} | {iters or '—'} |")
    table = "\n".join(lines)

    # importance by grammar axis: auto models of the final run, mean over seeds, normalised to 100
    axes = {"key": defaultdict(float), "filt": defaultdict(float), "window": defaultdict(float), "agg": defaultdict(float)}
    for run in final["variants"]["auto"]:
        for name, value in run["importance"].items():
            if name.startswith("src_"):
                axes["key"]["(флаги источников)"] += value / len(final["variants"]["auto"])
                continue
            spec = parse(name)
            for axis in axes:
                label = getattr(spec, axis)
                axes[axis]["all" if label is None and axis == "window" else str(label)] += value / len(final["variants"]["auto"])
    manual_groups = defaultdict(float)
    for run in final["variants"]["manual"]:
        for name, value in run["importance"].items():
            if name.startswith("src_"):
                group = "флаги источников"
            elif name.startswith("x_"):
                group = "доп. ручные (x_*)"
            else:
                group = parse(name).key
            manual_groups[group] += value / len(final["variants"]["manual"])
    chosen = selection["auto_features"]
    chosen_specs = [parse(n) for n in chosen]
    from .manual import MANUAL_SPECS
    overlap = sorted(set(chosen_specs) & set(MANUAL_SPECS), key=lambda s: s.name)
    report = {
        "test_table_markdown": table, "contrasts": contrasts, "k_trace": [{"k": t["k"], "selection_ndcg_10": t["selection"]["ndcg_10"], "fit_seconds": t["wall_seconds"]} for t in selection["k_trace"]],
        "reference_on_selection": {k: v["ndcg_10"] for k, v in selection["reference_on_selection"].items()},
        "auto_importance_by_axis": {a: dict(sorted(v.items(), key=lambda x: -x[1])) for a, v in axes.items()},
        "manual_importance_by_group": dict(sorted(manual_groups.items(), key=lambda x: -x[1])),
        "auto_chosen_k": selection["chosen_k"], "auto_overlap_with_manual": [s.name for s in overlap],
        "auto_window_counts": {str(w): sum(s.window == w for s in chosen_specs) for w in (1, 3, 7, 14, 30, 90, None)},
        "auto_key_counts": {k: sum(s.key == k for s in chosen_specs) for k in sorted({s.key for s in chosen_specs})},
        "screen": {k: screen[k] for k in ("grammar_size", "nonzero_importance", "best_iteration", "feature_resources", "fit_resources")},
        "slices": slices, "stage_resources": stages,
        "top_auto_features": chosen[:20],
    }
    write_json(EXPERIMENT / "reports" / f"results{suffix}.json", report)
    (EXPERIMENT / "reports" / f"results-table{suffix}.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(json.dumps({k: report[k] for k in ("auto_importance_by_axis", "manual_importance_by_group", "auto_window_counts", "auto_key_counts", "k_trace", "reference_on_selection")}, ensure_ascii=False, indent=1))
    print(len(overlap), "overlap")


if __name__ == "__main__":
    main(EXPERIMENT / "artifacts" / "main", sys.argv[1] if len(sys.argv) > 1 else "final.json")
