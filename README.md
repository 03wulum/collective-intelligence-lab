# collective-intelligence-lab

Can many tiny models working together match a big one? Small, one-command experiments with pass/fail criteria fixed in advance. They run on a laptop, with no cloud. Rules: [`docs/SOP.md`](docs/SOP.md) (v0.6).

## Results

| # | Goal | Verdict | Criteria | Finding in one line | |
|---|---|---|---|---|---|
| 01 | Exponential gains from a swarm of tiny models | FAIL ✨ INTRIGUING | 3/5 | Voting takes 82% to 99%, but error falls as a power law (N^−1.44); diversity, not count, drives the gains | [details](experiments/01-swarm-scaling/) |

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python run_all.py           # every experiment in --quick mode, one verdict table
python run_all.py --full    # full runs
python run_all.py --clean   # delete caches and report the space freed
```

MIT licensed.
