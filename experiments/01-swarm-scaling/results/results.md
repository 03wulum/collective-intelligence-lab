# Results · experiment 01 · full

- seeds: [0, 1, 2] · quick: False · runtime: 48 s
- peak RAM: 0.57 GB · disk (model cache): 28.9 MB
- Python 3.13.18 · torch 2.15.0+cu132 · numpy 2.5.3
- CPU: Intel(R) Xeon(R) Processor @ 2.10GHz (4 threads) · Linux-6.18.44-fc-v236-x86_64-with-glibc2.39

**Verdict: FAIL** ✨ INTRIGUING: Error follows a clean power law (~N^-1.4) whose floor is set by shared mistakes: the same members with independent errors would be perfect by N=8.

```
MAIN (medium members, T=0.7, mean ± sd over 3 seeds)
  criterion                                               value  threshold  result
  C1 Swarm beats its members                     +0.158 ± 0.035 >=     0.1  PASS
  C2 Error falls exponentially (the core claim)   -6.121 ± 0.374 >     0.0  FAIL
  C3 Diversity, not count, drives it             +0.125 ± 0.029 >=    0.05  PASS
  C4 Worth the compute                           -0.016 ± 0.014 >=   -0.01  FAIL
  C5 Adaptive stopping saves calls                            1 ==       1  PASS

NEGATIVE CONTROL (clone swarm; C1 and C3 must FAIL)
  criterion                                               value  threshold  result
  C1 Swarm beats its members                     +0.000 ± 0.000 >=     0.1  FAIL
  C2 Error falls exponentially (the core claim)   +3.634 ± 3.964 >     0.0  PASS
  C3 Diversity, not count, drives it             +0.000 ± 0.000 >=    0.05  FAIL
  C4 Worth the compute                                      n/a >=   -0.01  n/a
  C5 Adaptive stopping saves calls                            1 ==       1  PASS

PERTURBATION: weak members
  criterion                                               value  threshold  result
  C1 Swarm beats its members                     +0.360 ± 0.000 >=     0.1  PASS
  C2 Error falls exponentially (the core claim)  -12.279 ± 0.000 >     0.0  FAIL
  C3 Diversity, not count, drives it             +0.313 ± 0.000 >=    0.05  PASS
  C4 Worth the compute                                      n/a >=   -0.01  n/a
  C5 Adaptive stopping saves calls                            0 ==       1  FAIL

PERTURBATION: temperature 1.0
  criterion                                               value  threshold  result
  C1 Swarm beats its members                     +0.250 ± 0.000 >=     0.1  PASS
  C2 Error falls exponentially (the core claim)  -10.419 ± 0.000 >     0.0  FAIL
  C3 Diversity, not count, drives it             +0.167 ± 0.000 >=    0.05  PASS
  C4 Worth the compute                                      n/a >=   -0.01  n/a
  C5 Adaptive stopping saves calls                            1 ==       1  PASS
```

## Accuracy vs N (medium members, mean over seeds)

| N | vote | 1 member × N samples | pass@N | independent-errors counterfactual | clone swarm |
|---|---|---|---|---|---|
| 1 | 0.824 | 0.823 | 0.824 | 0.824 | 0.860 |
| 2 | 0.831 | 0.830 | 0.964 | 0.831 | 0.865 |
| 4 | 0.964 | 0.835 | 0.996 | 0.989 | 0.852 |
| 8 | 0.986 | 0.861 | 1.000 | 1.000 | 0.869 |
| 16 | 0.992 | 0.851 | 1.000 | 1.000 | 0.863 |

Mean single-member accuracy (sampled): 0.824
One member trained 4× longer (greedy), per seed: {0: 1.0, 1: 0.9825, 2: 0.9925}
