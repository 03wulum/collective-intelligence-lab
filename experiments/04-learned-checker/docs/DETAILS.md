# 04 · Details

Plan and frozen criteria: [plan doc](https://claude.ai/artifact/RZR6Z3pPwEiwEVNDV5vPF3), [`criteria.py`](../criteria.py). Raw numbers: [`results/results.md`](../results/results.md), [`results/verdict.json`](../results/verdict.json), [`results/console.txt`](../results/console.txt).

## Hypothesis

Experiment 01 found that a swarm of 4 tiny models almost always *contains* the right answer (pass@4) but the vote often fails to pick it. Hypothesis: a judge the size of one member, trained to tell right sums from wrong ones on the members' own answers, closes at least half of that gap.

## Setup

- **Members:** 8 per seed, TinyGPT (2 layers, d=64, 103k parameters, 0.4 MB), 1,250 steps each on its own random slice of 4,000 problems. Weak members: 1,000 steps.
- **Judge:** the same body with one output, read at the end-of-answer token. Trained 1,500 steps (batch 128, class-balanced) on ~64,000 of the members' sampled answers to 2,000 fresh problems, labelled right or wrong by exact match.
- **Selection:** among the distinct candidates of 4 members (one sample each, T=0.7), take the highest judge score; the vote count breaks ties.
- **Compute-matched baseline:** one member trained on the same total tokens as 4 members plus the judge (8,000 steps).
- **Speed:** all 48 members, 3 baselines and 6 judges train at once as stacked models in one GPU process (`checker/tinylm.py`).
- **Seeds:** 3. Each sweep averages 100 random member subsets over 400 test problems.

## Results (mean over 3 seeds)

| Rule, N=4 | Normal members | Weak members |
|---|---|---|
| One member | 52.0% | 35.2% |
| Majority vote | 73.4% | 47.9% |
| **Judge** | **37.7%** | **26.0%** |
| Shuffled-label judge | 36.5% | 24.7% |
| Best possible (pass@4) | 94.5% | 81.2% |
| One model, same training budget | 99.9% | — |

- Judge AUC on test candidates: 0.64 (shuffled-label judge: 0.46). It has a faint signal, far too weak to rank candidates.
- The judge selects *below a single member*. Choosing among distinct candidates gives each wrong answer the same weight as the right one, which is usually the most repeated. A near-random judge throws away the vote's main advantage.
- Perturbations: N=8 and temperature 1.0 fail the same way.
- Budget: 1.9 min training, 2.5 min wall, 1.55 GB peak RAM, 23.6 MB of models (deleted after the run).

## Why it failed

1. **For addition, checking costs as much as solving.** To know whether `417+389=806` is right, the judge has to do the sum, carries included. A model of the same size trained for a similar number of steps does it no better than the members, so its scores carry little information.
2. **Training longer beats everything again.** One model trained on the swarm's plus the judge's token budget reaches 99.9%. That is the same lesson as experiment 01's C4, now much more decisive.

## Deviations from the plan

- **Members are weaker than in experiment 01** (52% vs 82% single-member accuracy). The stacked GPU rewrite (needed for the 5-minute budget) uses small (std 0.02) initial weights where 01 used PyTorch defaults, which slows learning at the same step count. The gap the judge had to close was larger (73% → 95%), so C1 was easier to pass, not harder. The verdict does not hinge on it.
- Training budget lowered from 15 to 10 minutes before the run, after Obi's new rule.
- `results.md` was first written in the Windows default encoding; it was re-encoded to UTF-8 and `run.py` now writes UTF-8.
- The `--quick` smoke run adds true sums to the judge data when untrained members produce no right answers. That only happens in `--quick`.

## Next

A verifier only helps when checking is cheaper than solving. Options: give the judge structure that makes checking easy (subtraction: c−b=a), test on a task with that property (constraint puzzles, unit tests for code), or let the judge see the members' confidences.
