# 04 · A tiny learned checker for swarms

**Goal:** a judge no bigger than one tiny model picks the swarm's right answer better than a majority vote.

**Verdict:** pending (criteria frozen in [`criteria.py`](criteria.py) before any run).

| ID | Criterion | Pass if |
|---|---|---|
| C1 (core) | Share of the vote@4 → pass@4 gap the judge closes | ≥ 50% |
| C2 | Swarm of 4 + judge vs one model on the same training budget | ≥ −1 pt |
| C3 | Judge size vs one member | ≤ 1× |
| C4 | Gap closed on weaker members, judge not retrained | ≥ 30% |

Negative control: a judge trained on shuffled labels must fail C1.

```bash
python run.py          # full run, NVIDIA GPU only (~1-3 min of training)
python run.py --quick  # CPU smoke check of the pipeline (not a result)
python run.py --clean  # delete cached models
```
