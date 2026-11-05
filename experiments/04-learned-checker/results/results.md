# Experiment 04 results (full)

**Verdict: FAIL**

| ID | Criterion | Value (mean ± sd over seeds) | Need | Result |
|---|---|---|---|---|
| C1 | The judge closes the gap (core) | -1.807 ± 0.553 | >= 0.5 | FAIL |
| C2 | Worth the compute | -0.622 ± 0.037 | >= -0.01 | FAIL |
| C3 | Tiny and cheap | 0.992 ± 0.000 | <= 1.0 | PASS |
| C4 | Transfers to weaker members | -0.666 ± 0.153 | >= 0.3 | FAIL |

Accuracy at N=4 (normal members, T=0.7):

| Rule | Accuracy |
|---|---|
| single | 52.0% ± 4.2 |
| vote | 73.4% ± 7.3 |
| judge | 37.7% ± 3.7 |
| shuffled_judge | 36.5% ± 3.1 |
| pass | 94.5% ± 2.6 |
| matched single model | 99.9% |

Weak members, N=4:

| Rule | Accuracy |
|---|---|
| single | 35.2% ± 4.1 |
| vote | 47.9% ± 5.9 |
| judge | 26.0% ± 1.8 |
| pass | 81.2% ± 4.9 |

Negative control (shuffled-label judge) C1: FAIL (gap closed -1.871)
Judge AUC 0.638, shuffled 0.455

Perturbations (C1 metric):
- N=8: gap closed -11.066 -> C1 FAIL
- temperature 1.0: gap closed -1.319 -> C1 FAIL

Budget: training 1.9 min on cuda NVIDIA GeForce RTX 4080 SUPER, peak RAM 1.55 GB, models 23.6 MB, wall 2.5 min
