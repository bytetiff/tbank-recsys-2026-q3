"""Figures for the report, built from saved artifacts only (no training).

    uv run --project ../.. --with matplotlib python -m strongfeat.figures

Writes SVG files to reports/figures/.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .common import EXPERIMENT

OUT = EXPERIMENT / "reports" / "figures"
ART = EXPERIMENT / "artifacts" / "main"
DARK, LIGHT = "0.25", "0.7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none", "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True})


def pipeline():
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.6); ax.axis("off")
    boxes = {"events": (0.0, 1.0, "события\n(1000 польз.)"), "cand": (2.0, 1.0, "кандидаты\n(≈430 на польз.)"),
             "manual": (4.4, 1.75, "ручные признаки (59)"), "auto": (4.4, 0.25, "грамматика: 938 → 120"),
             "model": (7.8, 1.0, "CatBoost")}
    w, h = {"events": 1.6, "cand": 2.1, "manual": 2.6, "auto": 2.6, "model": 1.6}, 0.6
    for key, (x, y, text) in boxes.items():
        ax.add_patch(Rectangle((x, y), w[key], h, fill=False, ec="black", lw=0.8))
        ax.text(x + w[key] / 2, y + h / 2, text, ha="center", va="center")
    kw = dict(arrowstyle="->", color="black", lw=0.8, shrinkA=0, shrinkB=0)
    ax.annotate("", (2.0, 1.3), (1.6, 1.3), arrowprops=kw)
    ax.annotate("", (4.4, 2.05), (4.1, 1.45), arrowprops=kw); ax.annotate("", (4.4, 0.55), (4.1, 1.15), arrowprops=kw)
    ax.annotate("", (7.8, 1.45), (7.0, 2.05), arrowprops=kw); ax.annotate("", (7.8, 1.15), (7.0, 0.55), arrowprops=kw)
    fig.savefig(OUT / "pipeline.svg", bbox_inches="tight"); plt.close(fig)


def results():
    final = json.loads((ART / "final-tuned.json").read_text())
    c = final["contrasts_ndcg_10"]
    mean = {v: np.mean([r["metrics"]["ndcg_10"] for r in runs]) for v, runs in final["variants"].items()}
    std = {v: np.std([r["metrics"]["ndcg_10"] for r in runs]) for v, runs in final["variants"].items()}
    mean |= {k: v["metrics"]["ndcg_10"] for k, v in final["baselines"].items()}
    order = [("popular", "популярность"), ("history_count", "история пары"), ("old_manual", "базовые ручные"),
             ("manual", "ручные"), ("auto", "авто"), ("auto_manual", "ручные + авто")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"wspace": 0.6, "width_ratios": [1.3, 1]})
    xs = np.arange(len(order))
    a1.bar(xs, [mean[k] for k, _ in order], yerr=[std.get(k, 0) for k, _ in order], width=0.6, color=LIGHT, ec="black", lw=0.6, capsize=3)
    a1.set_xticks(xs, [lab for _, lab in order], rotation=30, ha="right"); a1.tick_params(axis="x", length=0); a1.set_ylabel("NDCG@10 (test)"); a1.set_ylim(0, 0.5)
    a1.set_title("(а)", loc="left")
    pairs = [("old_manual", "базовые ручные"), ("auto", "авто"), ("auto_manual", "ручные + авто")]
    for i, (k, lab) in enumerate(pairs):
        d = c[f"{k}-manual"]
        a2.errorbar(d["mean"], i, xerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]], fmt="o", color="black", ms=5, capsize=3, lw=0.9)
    a2.axvline(0, color="black", lw=0.6, ls="--")
    a2.set_yticks(range(len(pairs)), [lab for _, lab in pairs]); a2.set_ylim(-0.6, len(pairs) - 0.4)
    a2.set_xlabel("Δ NDCG@10 относительно «ручные»"); a2.set_title("(б)", loc="left"); a2.tick_params(axis="y", length=0)
    fig.savefig(OUT / "results.svg", bbox_inches="tight"); plt.close(fig)


def analysis():
    selection = json.loads((ART / "selection.json").read_text())
    ablation = json.loads((EXPERIMENT / "reports" / "manual-ablation-selection.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"wspace": 0.55})
    ks = [t["k"] for t in selection["k_trace"]]; vals = [t["selection"]["ndcg_10"] for t in selection["k_trace"]]
    ref = selection["reference_on_selection"]
    a1.plot(ks, vals, "-o", color="black", lw=0.9, ms=4, label="авто, top-K")
    a1.axhline(ref["manual"]["ndcg_10"], color=DARK, lw=0.8, ls="--", label="ручные")
    a1.axhline(ref["old_manual"]["ndcg_10"], color=DARK, lw=0.8, ls=":", label="базовые ручные")
    a1.set_xticks(ks); a1.set_xlabel("K, число отобранных признаков"); a1.set_ylabel("NDCG@10 (selection)")
    a1.legend(frameon=False, fontsize=10, loc="lower right"); a1.set_title("(а)", loc="left")
    names = {"extras": "вне грамматики", "ratios_and_shares": "доли и средние", "source_flags": "флаги источника",
             "event_types": "типы событий", "indirect": "артист и альбом"}
    base = ablation["none"]["selection_ndcg_10"]
    items = sorted(names, key=lambda k: ablation[k]["selection_ndcg_10"] - base, reverse=True)
    a2.barh(range(len(items)), [ablation[k]["selection_ndcg_10"] - base for k in items], height=0.6, color=LIGHT, ec="black", lw=0.6)
    a2.set_yticks(range(len(items)), [names[k] for k in items]); a2.axvline(0, color="black", lw=0.6)
    a2.set_xlabel("Δ NDCG@10 без группы (selection)"); a2.set_title("(б)", loc="left"); a2.tick_params(axis="y", length=0)
    fig.savefig(OUT / "analysis.svg", bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline(); results(); analysis()
    print("written", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
