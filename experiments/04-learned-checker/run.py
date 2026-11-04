"""Experiment 04: can a tiny learned judge beat the swarm's majority vote?

    python run.py            # full run; trains on an NVIDIA GPU (refuses CPU)
    python run.py --quick    # pipeline smoke check, <60 s, CPU allowed, verdict not meaningful
    python run.py --clean    # delete cached models in data/

Prints the PASS/FAIL table for the main run, the negative control and the perturbations,
then writes results/verdict.json, results/results.md, results/curves.json and the plot.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import criteria  # noqa: E402
from checker.tasks import make_problems  # noqa: E402
from checker.tinylm import Judge, TinyConfig, TinyGPT, device, train_judge, train_member  # noqa: E402

DATA = HERE / "data"        # git-ignored model cache
RESULTS = HERE / "results"

MODEL = {"d_model": 64, "n_layer": 2, "n_head": 4}
PRESETS = {
    "quick": dict(seeds=1, pool=8, steps=60, weak_steps=40, questions=60, bootstrap=5, samples=2,
                  judge_problems=200, judge_samples=2, judge_steps=60, judge_bs=128,
                  model={"d_model": 32, "n_layer": 1, "n_head": 4}),
    "full": dict(seeds=3, pool=8, steps=1250, weak_steps=1000, questions=400, bootstrap=100, samples=8,
                 judge_problems=2000, judge_samples=4, judge_steps=1500, judge_bs=128, model=MODEL),
}
MEMBER_BS = 64
TEMPERATURE = 0.7
INTRIGUING = None  # optional SOP v0.6 tag, set only after results exist; never changes the verdict


# ----------------------------------------------------------------------------- cached training

def _cached(path: Path, build, cls, cfg):
    if path.exists():
        blob = torch.load(path, weights_only=False, map_location=device())
        m = cls(cfg).to(device())
        m.load_state_dict(blob["state"])
        return m, {**blob["meta"], "cached": True}
    m, meta = build()
    DATA.mkdir(exist_ok=True)
    torch.save({"state": m.state_dict(), "meta": meta}, path)
    return m, {**meta, "cached": False}


def member(tag, cfg, seed, steps):
    return _cached(DATA / f"{tag}.pt", lambda: train_member(cfg, seed=seed, steps=steps, bs=MEMBER_BS), TinyGPT, cfg)


def judge(tag, cfg, strings, labels, seed, P):
    return _cached(DATA / f"{tag}.pt",
                   lambda: train_judge(cfg, strings, labels, seed=seed, steps=P["judge_steps"], bs=P["judge_bs"]),
                   Judge, cfg)


def samples_of(m, prompts, S, T, seed):
    g = torch.Generator().manual_seed(seed)
    return np.array([m.sample(prompts, temperature=T, generator=g) for _ in range(S)], dtype=object)  # [S, Q]


# ----------------------------------------------------------------------------- selection

def select(cands: list[str], scores: list[float]) -> str:
    """Highest judge score among distinct candidates; the vote count breaks ties."""
    best, key = None, None
    for c in set(cands):
        k = (max(s for x, s in zip(cands, scores) if x == c), cands.count(c), c)
        if key is None or k > key:
            best, key = c, k
    return best


def vote(cands: list[str]) -> str:
    return max(set(cands), key=lambda c: (cands.count(c), c))


def sweep(ans, sc, sc_neg, truth, N, B, seed):
    """Mean accuracy of each selection rule over B random N-member subsets (1 sample each)."""
    rng = np.random.default_rng(seed)
    M, S, Q = ans.shape
    acc = {k: [] for k in ["single", "vote", "judge", "shuffled_judge", "pass"]}
    for _ in range(B):
        mem = rng.choice(M, N, replace=False)
        smp = rng.integers(0, S, N)
        a, s, sn = ans[mem, smp, :], sc[mem, smp, :], sc_neg[mem, smp, :]
        acc["single"].append(float(np.mean(a == truth[None, :])))
        acc["vote"].append(np.mean([vote(list(a[:, q])) == truth[q] for q in range(Q)]))
        acc["judge"].append(np.mean([select(list(a[:, q]), list(s[:, q])) == truth[q] for q in range(Q)]))
        acc["shuffled_judge"].append(np.mean([select(list(a[:, q]), list(sn[:, q])) == truth[q] for q in range(Q)]))
        acc["pass"].append(np.mean([truth[q] in a[:, q] for q in range(Q)]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def gap_closed(r, key="judge"):
    gap = r["pass"] - r["vote"]
    return (r[key] - r["vote"]) / gap if gap > 1e-9 else 0.0


def auc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels, bool)
    if y.all() or not y.any():
        return float("nan")
    ranks = s.argsort().argsort() + 1
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


# ----------------------------------------------------------------------------- one seed

def run_seed(s, P):
    cfg = TinyConfig(**P["model"])
    d = P["model"]["d_model"]
    test = make_problems(P["questions"], seed=2026 + s)
    prompts = [p.prompt for p in test]
    truth = np.array([p.answer for p in test], dtype=object)
    jprob = make_problems(P["judge_problems"], seed=50_000 + s)
    train_s = 0.0

    pools = {}
    for level, steps in [("normal", P["steps"]), ("weak", P["weak_steps"])]:
        ms = []
        for i in range(P["pool"]):
            m, meta = member(f"{level}_d{d}_s{steps}_seed{s}_m{i}", cfg, 100_000 + 1000 * s + i, steps)
            train_s += 0 if meta["cached"] else meta["train_seconds"]
            ms.append((m, meta))
        pools[level] = ms
    member_params = pools["normal"][0][1]["params"]

    # judge training data: the normal members' own sampled answers to fresh problems
    strings, labels = [], []
    for i, (m, _) in enumerate(pools["normal"]):
        sm = samples_of(m, [p.prompt for p in jprob], P["judge_samples"], TEMPERATURE, 400_000 + 1000 * s + i)
        for row in sm:
            for p, a in zip(jprob, row):
                strings.append(p.prompt + a)
                labels.append(int(a == p.answer))
    rng = np.random.default_rng(s)
    shuffled = list(rng.permutation(labels))
    jcfg = TinyConfig(**P["model"])
    J, jmeta = judge(f"judge_d{d}_seed{s}", jcfg, strings, labels, 300_000 + s, P)
    Jn, jnmeta = judge(f"judge_shuffled_d{d}_seed{s}", jcfg, strings, shuffled, 310_000 + s, P)
    train_s += sum(0 if m["cached"] else m["train_seconds"] for m in (jmeta, jnmeta))

    # compute-matched single model: 4 members' tokens + the judge's tokens
    matched_steps = math.ceil((4 * P["steps"] * MEMBER_BS + P["judge_steps"] * P["judge_bs"]) / MEMBER_BS)
    Mm, mmeta = member(f"matched_d{d}_s{matched_steps}_seed{s}", cfg, 200_000 + s, matched_steps)
    train_s += 0 if mmeta["cached"] else mmeta["train_seconds"]
    matched_acc = float(np.mean(np.array(Mm.sample(prompts, temperature=0), dtype=object) == truth))

    def scored(ms, T, off):
        ans = np.array([samples_of(m, prompts, P["samples"], T, 500_000 + off + 1000 * s + i)
                        for i, (m, _) in enumerate(ms)], dtype=object)       # [M, S, Q]
        flat = [p + a for row in ans.reshape(-1, len(prompts)) for p, a in zip(prompts, row)]
        sc = np.array(J.score(flat)).reshape(ans.shape)
        sn = np.array(Jn.score(flat)).reshape(ans.shape)
        return ans, sc, sn

    out = {"seed": s}
    a, sc, sn = scored(pools["normal"], TEMPERATURE, 0)
    out["n4"] = sweep(a, sc, sn, truth, 4, P["bootstrap"], s)
    out["n8"] = sweep(a, sc, sn, truth, min(8, P["pool"]), P["bootstrap"], s + 1)
    out["judge_auc"] = auc(sc.ravel(), (a == truth[None, None, :]).ravel())
    out["shuffled_auc"] = auc(sn.ravel(), (a == truth[None, None, :]).ravel())
    a, sc, sn = scored(pools["normal"], 1.0, 50_000)
    out["t1_n4"] = sweep(a, sc, sn, truth, 4, P["bootstrap"], s + 2)
    a, sc, sn = scored(pools["weak"], TEMPERATURE, 90_000)
    out["weak_n4"] = sweep(a, sc, sn, truth, 4, P["bootstrap"], s + 3)
    out["matched_acc"] = matched_acc
    out["matched_steps"] = matched_steps
    out["judge_train_pos_neg"] = [jmeta.get("n_pos"), jmeta.get("n_neg")]
    out["metrics"] = {
        "gap_closed_n4": gap_closed(out["n4"]),
        "gain_judge4_vs_matched": out["n4"]["judge"] - matched_acc,
        "judge_size_ratio": jmeta["params"] / member_params,
        "gap_closed_weak_n4": gap_closed(out["weak_n4"]),
    }
    out["negative"] = {**out["metrics"], "gap_closed_n4": gap_closed(out["n4"], "shuffled_judge"),
                       "gain_judge4_vs_matched": out["n4"]["shuffled_judge"] - matched_acc,
                       "gap_closed_weak_n4": gap_closed(out["weak_n4"], "shuffled_judge")}
    out["perturb"] = {"N=8": {**out["metrics"], "gap_closed_n4": gap_closed(out["n8"])},
                      "temperature 1.0": {**out["metrics"], "gap_closed_n4": gap_closed(out["t1_n4"]),
                                          "gain_judge4_vs_matched": out["t1_n4"]["judge"] - matched_acc}}
    out["params"] = {"member": member_params, "judge": jmeta["params"]}
    out["train_seconds"] = train_s
    return out


# ----------------------------------------------------------------------------- scoring + output

def mean_std(xs):
    return float(np.mean(xs)), float(np.std(xs))


def score(metrics: dict) -> dict:
    return {c["id"]: criteria.check(metrics[c["metric"]], c["op"], c["threshold"]) for c in criteria.CRITERIA}


def agg(runs, key):
    keys = runs[0][key].keys()
    return {k: float(np.mean([r[key][k] for r in runs])) for k in keys}


def table(title, metrics, passed):
    lines = [f"\n{title}", f"{'ID':<4}{'criterion':<34}{'value':>10}{'need':>12}  result"]
    for c in criteria.CRITERIA:
        lines.append(f"{c['id']:<4}{c['name']:<34}{metrics[c['metric']]:>10.3f}{c['op'] + str(c['threshold']):>12}  "
                     f"{'PASS' if passed[c['id']] else 'FAIL'}")
    return "\n".join(lines)


def peak_ram_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except ImportError:  # Windows
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1e9
        except Exception:
            return float("nan")


def disk_mb():
    return sum(f.stat().st_size for f in DATA.glob("*.pt")) / 1e6 if DATA.exists() else 0.0


def write_md(path, v, runs):
    n4 = {k: mean_std([r["n4"][k] for r in runs]) for k in runs[0]["n4"]}
    wk = {k: mean_std([r["weak_n4"][k] for r in runs]) for k in runs[0]["weak_n4"]}
    L = [f"# Experiment 04 results ({v['preset']})", "", f"**Verdict: {v['verdict']}**"
         + (f" ✨ INTRIGUING: {v['intriguing']}" if v.get("intriguing") else ""), "",
         "| ID | Criterion | Value (mean ± sd over seeds) | Need | Result |", "|---|---|---|---|---|"]
    for c in criteria.CRITERIA:
        m, sd = v["metric_stats"][c["metric"]]
        L.append(f"| {c['id']} | {c['name']} | {m:.3f} ± {sd:.3f} | {c['op']} {c['threshold']} | "
                 f"{'PASS' if v['main'][c['id']] else 'FAIL'} |")
    L += ["", "Accuracy at N=4 (normal members, T=0.7):", "", "| Rule | Accuracy |", "|---|---|"]
    for k in ["single", "vote", "judge", "shuffled_judge", "pass"]:
        L.append(f"| {k} | {100 * n4[k][0]:.1f}% ± {100 * n4[k][1]:.1f} |")
    L.append(f"| matched single model | {100 * np.mean([r['matched_acc'] for r in runs]):.1f}% |")
    L += ["", "Weak members, N=4:", "", "| Rule | Accuracy |", "|---|---|"]
    for k in ["single", "vote", "judge", "pass"]:
        L.append(f"| {k} | {100 * wk[k][0]:.1f}% ± {100 * wk[k][1]:.1f} |")
    L += ["", f"Negative control (shuffled-label judge) C1: {'PASS' if v['negative']['C1'] else 'FAIL'} "
          f"(gap closed {v['negative_metrics']['gap_closed_n4']:.3f})",
          f"Judge AUC {np.mean([r['judge_auc'] for r in runs]):.3f}, shuffled {np.mean([r['shuffled_auc'] for r in runs]):.3f}",
          "", "Perturbations (C1 metric):"]
    for name, m in v["perturbations"].items():
        L.append(f"- {name}: gap closed {m['metrics']['gap_closed_n4']:.3f} -> C1 {'PASS' if m['passed']['C1'] else 'FAIL'}")
    b = v["budget"]
    L += ["", f"Budget: training {b['train_minutes']:.1f} min on {b['device']}, peak RAM {b['peak_ram_gb']:.2f} GB, "
          f"models {b['disk_mb']:.1f} MB, wall {b['wall_minutes']:.1f} min"]
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    if args.clean:
        mb = disk_mb()
        shutil.rmtree(DATA, ignore_errors=True)
        print(f"removed {DATA} ({mb:.1f} MB)")
        return
    preset = "quick" if args.quick else "full"
    if preset == "full" and not torch.cuda.is_available():
        sys.exit("Full runs train on an NVIDIA GPU only (project rule). Use --quick for a CPU smoke check.")
    P = PRESETS[preset]
    t0 = time.time()
    dev = device()
    print(f"experiment 04 · {preset} · {dev}" + (f" ({torch.cuda.get_device_name()})" if dev.type == "cuda" else ""))
    runs = []
    for s in range(P["seeds"]):
        r = run_seed(s, P)
        runs.append(r)
        print(f"seed {s}: vote {r['n4']['vote']:.3f} judge {r['n4']['judge']:.3f} pass {r['n4']['pass']:.3f} "
              f"matched {r['matched_acc']:.3f} weak vote {r['weak_n4']['vote']:.3f} judge {r['weak_n4']['judge']:.3f} "
              f"| trained {r['train_seconds']:.0f}s")

    metrics, neg = agg(runs, "metrics"), agg(runs, "negative")
    main_p, neg_p = score(metrics), score(neg)
    perturb = {}
    for name in runs[0]["perturb"]:
        m = {k: float(np.mean([r["perturb"][name][k] for r in runs])) for k in metrics}
        perturb[name] = {"metrics": m, "passed": score(m)}
    verdict = criteria.verdict(main_p, neg_p)
    train_min = sum(r["train_seconds"] for r in runs) / 60
    budget = {"train_minutes": train_min, "peak_ram_gb": peak_ram_gb(), "disk_mb": disk_mb(),
              "wall_minutes": (time.time() - t0) / 60, "device": str(dev) + (f" {torch.cuda.get_device_name()}" if dev.type == "cuda" else "")}
    budget_ok = {"train_minutes": train_min <= criteria.BUDGET["train_minutes"],
                 "peak_ram_gb": not budget["peak_ram_gb"] > criteria.BUDGET["peak_ram_gb"],
                 "disk_mb": budget["disk_mb"] <= criteria.BUDGET["disk_mb"]}
    if preset == "quick":
        budget_ok["quick_seconds"] = budget["wall_minutes"] * 60 <= criteria.BUDGET["quick_seconds"]

    print(table("MAIN (mean over seeds)", metrics, main_p))
    print(table("NEGATIVE CONTROL: shuffled-label judge (C1 must FAIL)", neg, neg_p))
    for name, p in perturb.items():
        print(table(f"PERTURBATION: {name}", p["metrics"], p["passed"]))
    print(f"\nbudget: {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in budget.items()})} ok={budget_ok}")
    print(f"\nVERDICT: {verdict}" + (" (quick smoke run, not meaningful)" if preset == "quick" else ""))

    RESULTS.mkdir(exist_ok=True)
    sfx = "_quick" if preset == "quick" else ""
    v = {"experiment": "04-learned-checker", "preset": preset, "verdict": verdict, "intriguing": INTRIGUING,
         "hypothesis": criteria.HYPOTHESIS, "main": main_p, "metrics": metrics,
         "metric_stats": {k: mean_std([r["metrics"][k] for r in runs]) for k in metrics},
         "negative": neg_p, "negative_metrics": neg, "perturbations": perturb,
         "budget": budget, "budget_ok": budget_ok, "criteria": criteria.CRITERIA,
         "python": platform.python_version(), "torch": torch.__version__}
    (RESULTS / f"verdict{sfx}.json").write_text(json.dumps(v, indent=2))
    (RESULTS / f"curves{sfx}.json").write_text(json.dumps(runs, indent=2, default=float))
    write_md(RESULTS / f"results{sfx}.md", v, runs)
    try:
        import plots
        plots.selection(runs, RESULTS / f"selection_n4{sfx}.png")
    except Exception as e:  # plots never break a run
        print(f"plot skipped: {e}")
    if not all(budget_ok.values()):
        sys.exit("budget exceeded")


if __name__ == "__main__":
    main()
