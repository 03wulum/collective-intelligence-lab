"""Scaling-law fits and error-correlation diagnostics.

The question "does capability grow exponentially, linearly or saturate in N?" is
answered by fitting competing curves to accuracy(N) and comparing them by AIC.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import binom


def _aic(y, yhat, k):
    n = len(y)
    rss = float(np.sum((np.asarray(y) - np.asarray(yhat)) ** 2)) + 1e-12
    return n * math.log(rss / n) + 2 * k


MODELS = {
    # acc = a + b*N
    "linear": (lambda n, a, b: a + b * n, [0.5, 0.01]),
    # acc = a + b*log2(N): constant gain per doubling
    "log": (lambda n, a, b: a + b * np.log2(n), [0.5, 0.05]),
    # acc = c - (c-a)*N^-b: saturates at c
    "saturating": (lambda n, a, b, c: c - (c - a) * np.power(n, -b), [0.5, 0.5, 0.9]),
    # super-linear: acc = a + b*N^g, g>1
    "superlinear": (lambda n, a, b, g: a + b * np.power(n, g), [0.5, 0.01, 1.5]),
}


def fit_scaling(ns, accs) -> dict:
    ns, accs = np.asarray(ns, float), np.asarray(accs, float)
    out = {}
    for name, (f, p0) in MODELS.items():
        try:
            bounds = (-np.inf, np.inf)
            if name == "saturating":
                bounds = ([0, 0, 0], [1, 5, 1.0])
            if name == "superlinear":
                bounds = ([-1, -1, 1.0], [1, 1, 4])
            p, _ = curve_fit(f, ns, accs, p0=p0, bounds=bounds, maxfev=20000)
            yhat = f(ns, *p)
            out[name] = {"params": [float(x) for x in p], "aic": _aic(accs, yhat, len(p)),
                         "rmse": float(np.sqrt(np.mean((accs - yhat) ** 2)))}
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": str(e)}
    ok = {k: v for k, v in out.items() if "aic" in v}
    out["best"] = min(ok, key=lambda k: ok[k]["aic"]) if ok else None
    return out


def gain_per_doubling(ns, accs) -> list[float]:
    """Accuracy added each time N doubles. Shrinking = saturating, constant = log-linear,
    growing = super-linear."""
    d = dict(zip(ns, accs))
    return [d[n * 2] - d[n] for n in ns if n * 2 in d]


def condorcet(p: float, n: int) -> float:
    """Majority-vote accuracy of n *independent* binary voters, each right w.p. p
    (ties broken by coin flip). The textbook 'exponential' case."""
    if n % 2:
        return float(1 - binom.cdf(n // 2, n, p))
    return float(1 - binom.cdf(n // 2, n, p) + 0.5 * binom.pmf(n // 2, n, p))


def error_correlation(correct: np.ndarray) -> float:
    """Mean pairwise Pearson correlation of per-question correctness across models.
    correct: [models, questions] bool. 0 = independent errors, 1 = identical models."""
    c = correct.astype(float)
    c = c[c.std(1) > 0]
    if len(c) < 2:
        return float("nan")
    m = np.corrcoef(c)
    iu = np.triu_indices(len(c), 1)
    return float(np.nanmean(m[iu]))


def error_decay(ns, accs, floor: float = 1e-3) -> dict:
    """Is the *error rate* falling exponentially in N (Condorcet, independent voters) or as a
    power law (heterogeneous difficulty: a few questions most members get wrong)?
    Fits log(err) linearly against N (exponential) and against log N (power law)."""
    ns = np.asarray(ns, float)
    err = np.clip(1 - np.asarray(accs, float), floor, 1)
    le = np.log(err)
    out = {}
    for name, x in [("exponential", ns), ("power_law", np.log(ns))]:
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, le, rcond=None)
        out[name] = {"slope": float(coef[1]), "aic": _aic(le, A @ coef, 2)}
    out["best"] = min(("exponential", "power_law"), key=lambda k: out[k]["aic"])
    return out
