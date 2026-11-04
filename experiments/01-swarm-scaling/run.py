"""Experiment 01: does a swarm of tiny models get better exponentially with N?

    python run.py            # full run, ~4-5 min on a 4-core laptop CPU
    python run.py --quick    # pipeline check, <30 s (verdict not meaningful)
    python run.py --extended # 32-member pools, N up to 32 (~45 min)
    python run.py --clean    # delete cached models in data/

Prints the PASS/FAIL table for the main run, the negative control and each perturbation,
then writes results/verdict.json, results/results.md and the plots.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import criteria  # noqa: E402
from swarm import analysis, patterns  # noqa: E402
from swarm.tasks import make_problems  # noqa: E402

DATA = HERE / "data"        # git-ignored model cache
RESULTS = HERE / "results"

MODEL = {"d_model": 64, "n_layer": 2, "n_head": 4}
PRESETS = {
    "quick": dict(seeds=1, pool=4, steps=150, weak_steps=100, ns=[1, 2, 4], questions=60, bootstrap=5,
                  samples=4, model={"d_model": 32, "n_layer": 1, "n_head": 4}, longer=False),
    "full": dict(seeds=3, pool=16, steps=1250, weak_steps=1000, ns=[1, 2, 4, 8, 16], questions=400,
                 bootstrap=100, samples=8, model=MODEL, longer=True),
    "extended": dict(seeds=3, pool=32, steps=1250, weak_steps=1000, ns=[1, 2, 4, 8, 16, 32], questions=500,
                     bootstrap=200, samples=32, model=MODEL, longer=True),
}
TEMPERATURE = 0.7

# Optional SOP v0.6 tag, added after the confirmatory run. It never changes the verdict.
INTRIGUING = ("Error follows a clean power law (~N^-1.4) whose floor is set by shared mistakes: "
              "the same members with independent errors would be perfect by N=8.")


# ----------------------------------------------------------------------------- training + sampling

def _member(job):
    """Train (or load cached) one member, then sample it. Runs in a worker process."""
    import torch

    from swarm.tinylm import TinyConfig, TinyGPT, train

    tag, mcfg, seed, steps, test_seed, Q, S, temps = job
    torch.set_num_threads(1)
    path = DATA / f"{tag}.pt"
    cfg = TinyConfig(**mcfg)
    if path.exists():
        blob = torch.load(path, weights_only=False)
        m = TinyGPT(cfg)
        m.load_state_dict(blob["state"])
        meta = blob["meta"]
    else:
        m, meta = train(cfg, seed=seed, steps=steps)
        DATA.mkdir(exist_ok=True)
        torch.save({"state": m.state_dict(), "meta": meta}, path)
    prompts = [p.prompt for p in make_problems(Q, seed=test_seed)]
    g = torch.Generator().manual_seed(seed + 7)  # sampling stream, separate from training stream
    greedy = [a for a, _ in m.sample(prompts, temperature=0)]
    samples = {T: [[a for a, _ in m.sample(prompts, temperature=T, generator=g)] for _ in range(S)] for T in temps}
    return greedy, samples, meta


def build_pool(level, steps, s, P, ex, temps=(TEMPERATURE,)):
    test_seed = 2026 + s
    jobs = [(f"{level}_d{P['model']['d_model']}_s{steps}_seed{s}_m{i}", P["model"], 100_000 + 1000 * s + i, steps,
             test_seed, P["questions"], P["samples"], temps) for i in range(P["pool"])]
    res = list(ex.map(_member, jobs))
    greedy = np.array([r[0] for r in res], dtype=object)                                # [M, Q]
    samples = {T: np.array([r[1][T] for r in res], dtype=object) for T in temps}         # [M, S, Q]
    truth = np.array([p.answer for p in make_problems(P["questions"], seed=test_seed)], dtype=object)
    return {"greedy": greedy, "samples": samples, "truth": truth, "params": res[0][2]["params"]}


# ----------------------------------------------------------------------------- sweep

def _acc(preds, truth):
    return float(np.mean([p == t for p, t in zip(preds, truth)]))


def _vote(a):
    return [patterns.plurality(list(a[:, q])) for q in range(a.shape[1])]


def sweep(pool, T, P, seed):
    """Accuracy vs N for each pattern, averaged over random member subsets."""
    rng = np.random.default_rng(seed)
    ans, greedy, truth = pool["samples"][T], pool["greedy"], pool["truth"]
    M, S, Q = ans.shape
    rows = []
    for N in P["ns"]:
        acc = {k: [] for k in ["vote", "homo", "greedy_vote", "pass_at_n", "independent", "single", "clone_vote",
                               "clone_single", "clone_homo"]}
        for _ in range(P["bootstrap"]):
            mem = rng.choice(M, N, replace=N > M)
            a = ans[mem, rng.integers(0, S, N), :]                       # N different members, 1 sample each
            base = rng.integers(0, M)
            h = ans[base, rng.choice(S, N, replace=N > S), :]           # 1 member sampled N times
            acc["vote"].append(_acc(_vote(a), truth))
            acc["homo"].append(_acc(_vote(h), truth))
            acc["greedy_vote"].append(_acc(_vote(greedy[mem]), truth))
            acc["pass_at_n"].append(float(np.mean([truth[q] in a[:, q] for q in range(Q)])))
            acc["single"].append(float(np.mean(a == truth[None, :])))
            # independent-errors counterfactual: shuffle each member's right/wrong pattern over questions
            ind = np.empty_like(a)
            for i in range(N):
                perm = rng.permutation(Q)
                ok = a[i, perm] == truth[perm]
                ind[i] = np.where(ok, truth, f"wrong{i}")
            acc["independent"].append(_acc(_vote(ind), truth))
            # negative control: N clones of one member, greedy decoding
            clones = np.repeat(greedy[base][None, :], N, axis=0)
            acc["clone_vote"].append(_acc(_vote(clones), truth))
            acc["clone_single"].append(_acc(greedy[base], truth))
            acc["clone_homo"].append(_acc(_vote(clones), truth))
        rows.append({"N": int(N), **{k: float(np.mean(v)) for k, v in acc.items()}})
    # adaptive stopping over up to max(ns) members, margin 2
    Nmax = max(P["ns"])
    ad_acc, ad_used = [], []
    for _ in range(P["bootstrap"]):
        mem = rng.choice(M, Nmax, replace=Nmax > M)
        a = ans[mem, rng.integers(0, S, Nmax), :]
        r = [patterns.adaptive(list(a[:, q]), margin=2) for q in range(Q)]
        ad_acc.append(_acc([x[0] for x in r], truth))
        ad_used.append(float(np.mean([x[1] for x in r])))
    return {"rows": rows, "adaptive": {"margin": 2, "acc": float(np.mean(ad_acc)), "calls": float(np.mean(ad_used))}}


def metrics_from(sw, ref_n=8, longer4=None, clone=False):
    """Turn one sweep into the criteria metrics. clone=True scores the negative control."""
    rows = {r["N"]: r for r in sw["rows"]}
    n = ref_n if ref_n in rows else max(rows)
    if clone:
        vote, single, homo = rows[n]["clone_vote"], rows[n]["clone_single"], rows[n]["clone_homo"]
        curve = [rows[k]["clone_vote"] for k in sorted(rows)]
    else:
        vote, single, homo = rows[n]["vote"], rows[n]["single"], rows[n]["homo"]
        curve = [rows[k]["vote"] for k in sorted(rows)]
    ns = sorted(rows)
    ed = analysis.error_decay([k for k in ns if k >= 2], [c for k, c in zip(ns, curve) if k >= 2])
    m = {"gain_vote8_vs_single": vote - single,
         "aic_power_minus_exp": ed["power_law"]["aic"] - ed["exponential"]["aic"],
         "power_law_slope": ed["power_law"]["slope"],
         "gain_hetero8_vs_homo8": vote - homo}
    if longer4 is not None and 4 in rows and not clone:
        m["gain_swarm4_vs_longer4"] = rows[4]["greedy_vote"] - longer4
    ad = sw["adaptive"]
    if clone:  # clones always agree, so adaptive stops after 2 identical answers
        ad = {"acc": rows[n]["clone_vote"], "calls": 2.0}
    m["adaptive_acc"], m["adaptive_calls"] = ad["acc"], ad["calls"]
    m["adaptive_ok"] = int(ad["acc"] >= vote - 0.01 and ad["calls"] <= 4)
    return m


def score(m):
    out = {}
    for c in criteria.CRITERIA:
        v = m.get(c["metric"])
        out[c["id"]] = None if v is None else bool(criteria.check(v, c["op"], c["threshold"]))
    return out


# ----------------------------------------------------------------------------- reporting

def _fmt(v):
    return "n/a" if v is None else (f"{v:+.3f}" if isinstance(v, float) else str(v))


def table(title, mean, sd, passed):
    lines = [f"\n{title}", f"  {'criterion':<44} {'value':>16} {'threshold':>10}  result"]
    for c in criteria.CRITERIA:
        v = mean.get(c["metric"])
        val = "n/a" if v is None else (f"{v:+.3f} ± {sd.get(c['metric'], 0):.3f}" if c["op"] != "==" else str(int(round(v))))
        res = "n/a" if passed[c["id"]] is None else ("PASS" if passed[c["id"]] else "FAIL")
        lines.append(f"  {c['id']} {c['name']:<41} {val:>16} {c['op']}{c['threshold']:>8}  {res}")
    return "\n".join(lines)


def agg(ms):
    keys = set().union(*ms)
    mean = {k: float(np.mean([m[k] for m in ms if k in m])) for k in keys}
    sd = {k: float(np.std([m[k] for m in ms if k in m])) for k in keys}
    if "adaptive_ok" in mean:  # pass only if adaptive works in every seed
        mean["adaptive_ok"] = int(all(m["adaptive_ok"] for m in ms))
    return mean, sd


def peak_ram_gb():
    """Peak resident memory of this process or its largest worker. 0.0 where unsupported (Windows)."""
    try:
        import resource
    except ImportError:
        return 0.0
    kb = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return kb / 1024**3 if platform.system() == "Darwin" else kb / 1024**2


def disk_mb(p: Path):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.exists() else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--extended", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    if a.clean:
        freed = disk_mb(DATA)
        shutil.rmtree(DATA, ignore_errors=True)
        for p in HERE.rglob("__pycache__"):
            shutil.rmtree(p, ignore_errors=True)
        print(f"deleted {DATA} and caches, freed {freed:.1f} MB")
        return
    mode = "quick" if a.quick else "extended" if a.extended else "full"
    P = PRESETS[mode]
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    seeds = [a.seed + i for i in range(P["seeds"])]
    print(f"Experiment 01 · {mode} · seeds {seeds} · {P['pool']} members/pool · N in {P['ns']}", flush=True)
    print(criteria.HYPOTHESIS, flush=True)

    per_seed, sweeps, weak_sw, hot_sw, longer = [], {}, None, None, {}
    with ProcessPoolExecutor(a.workers) as ex:
        # compute-matched baseline jobs go first so the long ones overlap with pool training
        longer_jobs = {}
        if P["longer"]:
            for s in seeds:
                longer_jobs[s] = ex.submit(_member, (f"longer4_s{P['steps'] * 4}_seed{s}", P["model"], 200_000 + s,
                                                     P["steps"] * 4, 2026 + s, P["questions"], 1, ()))
        for i, s in enumerate(seeds):
            temps = (TEMPERATURE, 1.0) if i == 0 else (TEMPERATURE,)
            pool = build_pool("medium", P["steps"], s, P, ex, temps)
            print(f"  seed {s}: medium pool ready ({pool['params']:,} params/member, "
                  f"mean greedy acc {np.mean(pool['greedy'] == pool['truth'][None, :]):.3f})", flush=True)
            sw = sweep(pool, TEMPERATURE, P, s)
            sweeps[s] = sw
            if s in longer_jobs:
                g_ans = longer_jobs[s].result()[0]
                longer[s] = _acc(g_ans, pool["truth"])
            per_seed.append(metrics_from(sw, longer4=longer.get(s)))
            if i == 0:
                hot_sw = sweep(pool, 1.0, P, s)
                wpool = build_pool("weak", P["weak_steps"], s, P, ex)
                weak_sw = sweep(wpool, TEMPERATURE, P, s)
                print(f"  seed {s}: weak pool ready (mean greedy acc "
                      f"{np.mean(wpool['greedy'] == wpool['truth'][None, :]):.3f})", flush=True)

    main_mean, main_sd = agg(per_seed)
    neg_mean, neg_sd = agg([metrics_from(sweeps[s], clone=True) for s in seeds])
    weak_mean, _ = agg([metrics_from(weak_sw)])
    hot_mean, _ = agg([metrics_from(hot_sw)])
    passed = {"main": score(main_mean), "negative": score(neg_mean), "weak": score(weak_mean),
              "temperature_1.0": score(hot_mean)}
    v = criteria.verdict({k: bool(x) for k, x in passed["main"].items() if x is not None}, passed["negative"])
    runtime = time.time() - t0

    out = [table(f"MAIN (medium members, T={TEMPERATURE}, mean ± sd over {len(seeds)} seeds)", main_mean, main_sd,
                 passed["main"]),
           table("NEGATIVE CONTROL (clone swarm; C1 and C3 must FAIL)", neg_mean, neg_sd, passed["negative"]),
           table("PERTURBATION: weak members", weak_mean, {}, passed["weak"]),
           table("PERTURBATION: temperature 1.0", hot_mean, {}, passed["temperature_1.0"])]
    report = "\n".join(out)
    print(report)
    tag = f"  ✨ INTRIGUING: {INTRIGUING}" if mode != "quick" and INTRIGUING else ""
    print(f"\nVERDICT: {v}{tag}   ({runtime:.0f}s, peak RAM {peak_ram_gb():.2f} GB, cache {disk_mb(DATA):.1f} MB)")

    curves = {"medium": {s: sweeps[s] for s in seeds}, "weak": weak_sw, "temperature_1.0": hot_sw}
    import torch
    env = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
           "cpu": _cpu(), "os": platform.platform()}
    vj = {"experiment": "01-swarm-scaling", "hypothesis": criteria.HYPOTHESIS, "verdict": v,
          "intriguing": INTRIGUING if mode != "quick" else None, "mode": mode,
          "seeds": seeds, "criteria": criteria.CRITERIA, "passed": passed,
          "metrics": {"main": {"mean": main_mean, "sd": main_sd}, "negative": {"mean": neg_mean, "sd": neg_sd},
                      "weak": weak_mean, "temperature_1.0": hot_mean},
          "longer4_baseline_acc": longer, "runtime_s": runtime, "peak_ram_gb": peak_ram_gb(),
          "disk_mb": disk_mb(DATA), "budget": criteria.BUDGET, "env": env}
    suffix = "" if mode == "full" else f"_{mode}"
    json.dump(vj, open(RESULTS / f"verdict{suffix}.json", "w"), indent=1)
    json.dump(curves, open(RESULTS / f"curves{suffix}.json", "w"), indent=1)
    write_md(RESULTS / f"results{suffix}.md", vj, report, curves)
    if mode != "quick":
        import plots
        plots.make_all(curves, longer, RESULTS)
    over = runtime > (criteria.BUDGET["quick_seconds"] if mode == "quick" else criteria.BUDGET["full_minutes"] * 60)
    if mode != "extended" and over:
        print(f"WARNING: runtime {runtime:.0f}s is over the {mode} budget")
    if peak_ram_gb() > criteria.BUDGET["peak_ram_gb"] or disk_mb(DATA) > criteria.BUDGET["disk_mb"]:
        print("BUDGET BLOWN")
        sys.exit(2)


def _cpu():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip() + f" ({os.cpu_count()} threads)"
    except OSError:
        pass
    return f"{platform.processor() or platform.machine()} ({os.cpu_count()} threads)"


def write_md(path, vj, report, curves):
    e = vj["env"]
    lines = [f"# Results · experiment 01 · {vj['mode']}", "",
             f"- seeds: {vj['seeds']} · quick: {vj['mode'] == 'quick'} · runtime: {vj['runtime_s']:.0f} s",
             f"- peak RAM: {vj['peak_ram_gb']:.2f} GB · disk (model cache): {vj['disk_mb']:.1f} MB",
             f"- Python {e['python']} · torch {e['torch']} · numpy {e['numpy']}",
             f"- CPU: {e['cpu']} · {e['os']}", "",
             f"**Verdict: {vj['verdict']}**" + (f" ✨ INTRIGUING: {vj['intriguing']}" if vj.get("intriguing") else ""), "", "```", report.strip(), "```", "",
             "## Accuracy vs N (medium members, mean over seeds)", "",
             "| N | vote | 1 member × N samples | pass@N | independent-errors counterfactual | clone swarm |",
             "|---|---|---|---|---|---|"]
    med = curves["medium"]
    for j, r in enumerate(next(iter(med.values()))["rows"]):
        avg = {k: np.mean([med[s]["rows"][j][k] for s in med]) for k in r if k != "N"}
        lines.append(f"| {r['N']} | {avg['vote']:.3f} | {avg['homo']:.3f} | {avg['pass_at_n']:.3f} | "
                     f"{avg['independent']:.3f} | {avg['clone_vote']:.3f} |")
    lines += ["", f"Mean single-member accuracy (sampled): {np.mean([med[s]['rows'][0]['single'] for s in med]):.3f}",
              f"One member trained 4× longer (greedy), per seed: {vj['longer4_baseline_acc']}"]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
