"""Pre-registered hypothesis, pass criteria, controls and budgets for experiment 04.

Frozen on 2026-11-04 before any run of this experiment (plan doc:
https://claude.ai/artifact/RZR6Z3pPwEiwEVNDV5vPF3). Thresholds come from experiment 01's
numbers and the cited papers, never from this experiment's results.
Do not edit thresholds after results exist; add a new experiment instead.
"""
from __future__ import annotations

HYPOTHESIS = (
    "A tiny judge (no bigger than one swarm member) trained to tell right sums from wrong ones "
    "picks the correct candidate from a 4-member swarm far more often than a majority vote, "
    "closing at least half of the vote -> pass@4 gap."
)

ASSUMPTIONS = [
    "3-digit addition stands in for any task with one checkable answer.",
    "The judge only sees the candidate strings 'a+b=c', never the members' confidences.",
]

CRITERIA = [
    {"id": "C1", "name": "The judge closes the gap (core)",
     "desc": "Share of the vote@4 -> pass@4 gap closed by judge selection at N=4",
     "metric": "gap_closed_n4", "op": ">=", "threshold": 0.50,
     "source": "Cobbe et al. 2021: trained verifiers recover most of best-of-N coverage"},
    {"id": "C2", "name": "Worth the compute",
     "desc": "Swarm of 4 + judge minus one model trained on the same total tokens (judge counted)",
     "metric": "gain_judge4_vs_matched", "op": ">=", "threshold": -0.01,
     "source": "Experiment 01 C4 (failed by 1.6 pts); 1-point tolerance"},
    {"id": "C3", "name": "Tiny and cheap",
     "desc": "Judge parameters / member parameters (and one forward pass per distinct candidate)",
     "metric": "judge_size_ratio", "op": "<=", "threshold": 1.0,
     "source": "Experiment design: the judge must be no bigger than one member"},
    {"id": "C4", "name": "Transfers to weaker members",
     "desc": "Gap closed at N=4 on weak members (fewer training steps), judge not retrained",
     "metric": "gap_closed_weak_n4", "op": ">=", "threshold": 0.30,
     "source": "Generalisation check; a lower bar than C1 because the judge never saw these members"},
]

CORE = ["C1"]

CONTROLS = {
    "baseline": "Majority vote, single member, oracle pass@N, one model trained on matched tokens.",
    "negative": "Judge trained on shuffled right/wrong labels. It must FAIL C1.",
    "perturbations": ["N=8 members", "sampling temperature 1.0", "3 independent seeds"],
}

BUDGET = {"peak_ram_gb": 2.0, "disk_mb": 50.0, "train_minutes": 10.0, "quick_seconds": 60.0}


def check(v: float, op: str, t: float) -> bool:
    return {">=": v >= t, ">": v > t, "<=": v <= t, "==": v == t}[op]


def verdict(main: dict[str, bool], negative: dict[str, bool]) -> str:
    """SOP v0.6. INCONCLUSIVE if the shuffled-label judge passes C1; FAIL if C1 fails;
    PASS if every criterion passes; INCONCLUSIVE for any other mix."""
    if negative.get("C1"):
        return "INCONCLUSIVE"
    if not all(main[c] for c in CORE):
        return "FAIL"
    return "PASS" if all(main.values()) else "INCONCLUSIVE"
