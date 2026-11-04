"""Pre-registered hypothesis, pass criteria, controls and budgets for experiment 01.

Frozen on 2026-11-04 before the confirmatory run (verdict words updated to SOP v0.6 later the
same day; thresholds and logic unchanged: the old NOT CONFIRMED is the new FAIL) (see README, "Deviations from SOP":
an exploratory run on different seeds and a different test set came first).
Thresholds come from the cited papers or from theory, never from this experiment's results.
Do not edit thresholds after results exist; add a new experiment instead.
"""
from __future__ import annotations

HYPOTHESIS = (
    "A swarm of N diverse ~100k-parameter models gains capability exponentially in N: "
    "majority-vote error falls like exp(-cN) rather than as a power law N^-b."
)

ASSUMPTIONS = [
    "Arithmetic (3-digit addition) is a fair stand-in for tasks with one checkable answer.",
    "Seed + bootstrap-resampled training data is enough to make members diverse.",
]

# Each criterion: id, description, metric key in the run's metrics dict, comparison, threshold, source.
CRITERIA = [
    {"id": "C1", "name": "Swarm beats its members",
     "desc": "Majority vote of 8 different members minus mean single-member accuracy",
     "metric": "gain_vote8_vs_single", "op": ">=", "threshold": 0.10,
     "source": "Li et al. 2024 (More Agents Is All You Need) report gains of this size on arithmetic/GSM8K"},
    {"id": "C2", "name": "Error falls exponentially (the core claim)",
     "desc": "Exponential fit of log error vs N beats power-law fit (AIC difference, power - exp)",
     "metric": "aic_power_minus_exp", "op": ">", "threshold": 0.0,
     "source": "Condorcet jury theorem: independent voters give exponential error decay"},
    {"id": "C3", "name": "Diversity, not count, drives it",
     "desc": "8 different members voting minus 1 member sampled 8 times",
     "metric": "gain_hetero8_vs_homo8", "op": ">=", "threshold": 0.05,
     "source": "Krogh & Vedelsby 1995: ensemble gain = member diversity (ambiguity)"},
    {"id": "C4", "name": "Worth the compute",
     "desc": "Swarm of 4 (greedy vote) minus one member trained 4x longer (same training FLOPs)",
     "metric": "gain_swarm4_vs_longer4", "op": ">=", "threshold": -0.01,
     "source": "Compute-matched baseline (Hoffmann et al. 2022 framing); 1-point tolerance"},
    {"id": "C5", "name": "Adaptive stopping saves calls",
     "desc": "Adaptive (margin 2) accuracy within 1 pt of fixed N=8, using <= 4 calls/question",
     "metric": "adaptive_ok", "op": "==", "threshold": 1,
     "source": "Aggarwal et al. 2023 (Adaptive-Consistency) report 3-6x fewer samples at <0.1 pt loss"},
]

CORE = ["C2"]  # the hypothesis itself

CONTROLS = {
    "baseline": "One member trained N x longer (same training FLOPs as an N-swarm); single member alone.",
    "negative": "Clone swarm: N copies of one member, greedy decoding. No diversity, so C1 and C3 must FAIL.",
    "perturbations": ["weak members (fewer training steps)", "sampling temperature 1.0", "3 independent seeds"],
}

BUDGET = {"peak_ram_gb": 2.0, "disk_mb": 50.0, "full_minutes": 5.0, "quick_seconds": 30.0}


def check(metric_value: float, op: str, threshold: float) -> bool:
    return {">=": metric_value >= threshold, ">": metric_value > threshold,
            "==": metric_value == threshold}[op]


def verdict(main: dict[str, bool], negative: dict[str, bool]) -> str:
    """SOP v0.6 verdicts. INCONCLUSIVE if the negative control passes C1 or C3 (the test can't
    tell). FAIL if the core criterion (C2, the hypothesis itself) fails. PASS only if every
    criterion passes; any other mix is INCONCLUSIVE."""
    if negative.get("C1") or negative.get("C3"):
        return "INCONCLUSIVE"
    if not all(main[c] for c in CORE):
        return "FAIL"
    return "PASS" if all(main.values()) else "INCONCLUSIVE"
