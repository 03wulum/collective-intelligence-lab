"""Plot an LLM sweep:  python llm/plot_llm.py llm/results/qwen2.5_0.5b.jsonl"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plots  # noqa: E402  (shared style)
from swarm import analysis  # noqa: E402

plt = plots.plt
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4"]


def main(path: Path):
    acc = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        r = json.loads(line)
        s = r["setting"]
        acc[s["pattern"] + (f" ({s['topology']})" if "topology" in s else "")][s["N"]].append(r["correct"])
    fig, ax = plt.subplots(figsize=(7, 4.8))
    summary = {}
    for i, (name, byn) in enumerate(sorted(acc.items())):
        ns = sorted(byn)
        y = [float(np.mean(byn[n])) for n in ns]
        se = [float(np.std(byn[n]) / np.sqrt(len(byn[n]))) for n in ns]
        color = plots.GRAY if name == "vote" else COLORS[i % len(COLORS)]
        ax.errorbar(ns, y, yerr=se, fmt="-o", color=color, capsize=3, label=name)
        summary[name] = {"N": ns, "acc": y, "fit": analysis.fit_scaling(ns, y) if len(ns) >= 4 else None}
    plots._logx(ax, sorted({n for v in acc.values() for n in v}))
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Small-LLM swarm: {path.stem}")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    json.dump(summary, open(path.with_suffix(".summary.json"), "w"), indent=1)
    print("wrote", path.with_suffix(".png"))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
