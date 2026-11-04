"""Plot for experiment 04: accuracy of each selection rule at N=4."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ACCENT, GRAY = "#2a78d6", "#9aa0a6"


def selection(runs, path):
    groups = [("normal members", "n4"), ("weak members", "weak_n4")]
    rules = [("one member", "single"), ("majority vote", "vote"), ("judge", "judge"),
             ("shuffled judge", "shuffled_judge"), ("best possible\n(pass@4)", "pass")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (title, key) in zip(axes, groups):
        vals = [100 * np.mean([r[key][k] for r in runs]) for _, k in rules]
        cols = [ACCENT if k == "judge" else GRAY for _, k in rules]
        bars = ax.bar(range(len(rules)), vals, color=cols)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
        if key == "n4":
            m = 100 * np.mean([r["matched_acc"] for r in runs])
            ax.axhline(m, color="black", lw=1, ls="--")
            ax.text(len(rules) - 0.5, m + 0.3, f"one model, same training budget: {m:.1f}", ha="right", fontsize=8)
        ax.set_xticks(range(len(rules)), [n for n, _ in rules], fontsize=8)
        ax.set_title(f"{title}, N=4")
        lo = min(vals) - 5
        ax.set_ylim(max(0, lo), 100.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy (%)")
    fig.suptitle("Does a tiny judge pick the right answer better than a vote?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
