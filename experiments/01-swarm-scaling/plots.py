"""Plots for experiment 01. 150 dpi, phone-readable, one message per plot.
The method (diverse swarm) is the blue accent; baselines are gray."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ACCENT, GRAY, LIGHT, INK = "#2a78d6", "#7a7974", "#b8b7b1", "#0b0b0b"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": LIGHT, "axes.grid": True,
    "grid.color": "#ebeae6", "axes.spines.top": False, "axes.spines.right": False, "font.size": 12,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 12, "legend.fontsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.frameon": False, "lines.linewidth": 2.2,
    "lines.markersize": 7,
})


def _mean_curve(med, key):
    seeds = list(med)
    ns = [r["N"] for r in med[seeds[0]]["rows"]]
    ys = np.array([[r[key] for r in med[s]["rows"]] for s in seeds])
    return np.array(ns), ys.mean(0), ys.std(0)


def _logx(ax, ns):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Models in the swarm (N)")


def make_all(curves, longer, out: Path):
    med = curves["medium"]
    ns, vote, vsd = _mean_curve(med, "vote")
    _, homo, hsd = _mean_curve(med, "homo")
    _, ind, _ = _mean_curve(med, "independent")
    single = vote[0]

    # 1. accuracy vs N
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(ns, homo, "-o", color=GRAY, label="1 model asked N times")
    ax.plot(ns, ind, ":", color=INK, linewidth=1.6, label="If mistakes were independent")
    ax.errorbar(ns, vote, yerr=vsd, fmt="-o", color=ACCENT, capsize=3, label="N different models, majority vote")
    _logx(ax, ns)
    ax.set_ylabel("Accuracy (500 held-out sums)" if len(vote) > 5 else "Accuracy (held-out sums)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(min(0.5, single - 0.1), 1.01)
    ax.set_title(f"Different models voting: {single:.0%} → {vote[-1]:.0%}, then it flattens")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_n.png", dpi=150)
    plt.close(fig)

    # 2. error decay, log-log
    fig, ax = plt.subplots(figsize=(7, 4.8))
    err, ierr = np.clip(1 - vote, 1e-4, 1), np.clip(1 - ind, 1e-4, 1)
    ax.plot(ns, ierr, ":", color=GRAY, linewidth=1.8, marker="o", label="If mistakes were independent (10⁻⁴ = zero errors)")
    ax.plot(ns, err, "-o", color=ACCENT, label="Real swarm")
    m = ns >= 2
    b, a = np.polyfit(np.log(ns[m]), np.log(err[m]), 1)
    ax.plot(ns[m], np.exp(a) * ns[m] ** b, "--", color=ACCENT, linewidth=1.2, alpha=0.7,
            label=f"Power-law fit: error ∝ N^{b:.2f}")
    _logx(ax, ns)
    ax.set_yscale("log")
    ax.set_ylabel("Error rate (log scale)")
    ax.set_title("Error shrinks as a power law, not exponentially")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "error_decay.png", dpi=150)
    plt.close(fig)

    # 3. compute-matched
    if longer:
        seeds = list(med)
        sw4 = [next(r["greedy_vote"] for r in med[s]["rows"] if r["N"] == 4) for s in seeds]
        sw1 = [next(r["greedy_vote"] for r in med[s]["rows"] if r["N"] == 1) for s in seeds]
        lg = [longer[s] for s in seeds]
        fig, ax = plt.subplots(figsize=(7, 4.8))
        labels = ["1 small model", "4 small models,\nmajority vote", "1 small model,\ntrained 4× longer"]
        vals, cols = [sw1, sw4, lg], [LIGHT, ACCENT, GRAY]
        for i, (v, c) in enumerate(zip(vals, cols)):
            ax.bar(i, np.mean(v), color=c, width=0.6)
            ax.scatter([i] * len(v), v, color=INK, s=18, zorder=3)
            ax.text(i, max(v) + 0.025, f"{np.mean(v):.1%}", ha="center", fontsize=12)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("Accuracy (dots = seeds)")
        ax.set_title(f"Same training compute: training longer wins by {100 * (np.mean(lg) - np.mean(sw4)):.1f} points")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        fig.savefig(out / "compute_matched.png", dpi=150)
        plt.close(fig)
