"""Experiment 04: can a tiny learned judge beat the swarm's majority vote?

    python run.py            # full run; trains on an NVIDIA GPU (refuses CPU), ~1-3 min of training
    python run.py --quick    # pipeline smoke check, <60 s, CPU allowed, verdict not meaningful
    python run.py --clean    # delete cached models in data/

All members of all seeds train at once in one process (stacked models, see checker/tinylm.py),
so the GPU is used fully and the run fits Obi's ~5-minute training budget.
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
from checker.tinylm import Stacked, TinyConfig, VOCAB, device, train_judges, train_members  # noqa: E402

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
INTRIGUING = None  # optional SOP v0.6 tag; none for this run


# ----------------------------------------------------------------------------- cached training

def _cached(tag: str, build, cfg, out):
    path = DATA / f"{tag}.pt"
    if path.exists():
        blob = torch.load(path, weights_only=False, map_location=device())
        m = Stacked(cfg, blob["K"], out).to(device())
        m.load_state_dict(blob["state"])
        return m, {**blob["meta"], "cached": True}
    m, meta = build()
    DATA.mkdir(exist_ok=True)
    torch.save({"state": m.state_dict(), "meta": meta, "K": m.K}, path)
    return m, {**meta, "cached": False}


def members(tag, cfg, seeds, steps):
    return _cached(tag, lambda: train_members(cfg, seeds, steps, bs=MEMBER_BS), cfg, len(VOCAB))


def samples_of(m: Stacked, prompts, S, T, seed):
    """[K, S, Q] sampled answers."""
    g = torch.Generator(device=m.tok.device).manual_seed(seed)
    return np.array([m.sample(prompts, T, generator=g) for _ in range(S)], dtype=object).transpose(1, 0, 2)
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




# ----------------------------------------------------------------------------- all seeds

def run_seeds(P):
    cfg = TinyConfig(**P["model"])
    d, S, Ps = P["model"]["d_model"], P["seeds"], P["pool"]
    seeds = range(S)
    mseeds = [100_000 + 1000 * s + i for s in seeds for i in range(Ps)]          # same seeds as experiment 01
    train_s = 0.0

    normal, nmeta = members(f"normal_d{d}_s{P['steps']}_S{S}_p{Ps}", cfg, mseeds, P["steps"])
    weak, wmeta = members(f"weak_d{d}_s{P['weak_steps']}_S{S}_p{Ps}", cfg, mseeds, P["weak_steps"])
    matched_steps = math.ceil((4 * P["steps"] * MEMBER_BS + P["judge_steps"] * P["judge_bs"]) / MEMBER_BS)
    matched, mmeta = members(f"matched_d{d}_s{matched_steps}_S{S}", cfg, [200_000 + s for s in seeds], matched_steps)
    train_s += sum(0 if m["cached"] else m["train_seconds"] for m in (nmeta, wmeta, mmeta))
    print(f"members trained: normal {nmeta['train_seconds']:.0f}s, weak {wmeta['train_seconds']:.0f}s, "
          f"matched ({matched_steps} steps) {mmeta['train_seconds']:.0f}s")
    member_params = nmeta["params"]

    # judge training data: each seed's normal members' own sampled answers to fresh problems
    datasets, shuffled = [], []
    for s in seeds:
        jprob = make_problems(P["judge_problems"], seed=50_000 + s)
        pool = normal.subset(list(range(s * Ps, (s + 1) * Ps)))
        sm = samples_of(pool, [p.prompt for p in jprob], P["judge_samples"], TEMPERATURE, 400_000 + s)
        strings, labels = [], []
        for row in sm.reshape(-1, len(jprob)):
            for p, a in zip(jprob, row):
                strings.append(p.prompt + a)
                labels.append(int(a == p.answer))
        if not any(labels) or all(labels):  # only happens with untrained --quick members
            print("  (smoke fallback: adding true sums so the judge has both classes)")
            strings += [p.prompt + p.answer for p in jprob] + [p.prompt + "0" for p in jprob]
            labels += [1] * len(jprob) + [0] * len(jprob)
        datasets.append((strings, labels))
        shuffled.append((strings, [int(v) for v in np.random.default_rng(s).permutation(labels)]))
    jtag = f"judges_d{d}_s{P['judge_steps']}_S{S}"
    judges, jmeta = _cached(jtag, lambda: train_judges(cfg, datasets + shuffled, 300_000, P["judge_steps"],
                                                       P["judge_bs"]), cfg, 1)
    train_s += 0 if jmeta["cached"] else jmeta["train_seconds"]
    print(f"judges trained ({2 * S}): {jmeta['train_seconds']:.0f}s")

    runs = []
    for s in seeds:
        test = make_problems(P["questions"], seed=2026 + s)
        prompts = [p.prompt for p in test]
        truth = np.array([p.answer for p in test], dtype=object)
        J = judges.subset([s, S + s])                         # this seed's real and shuffled-label judge
        matched_acc = float(np.mean(np.array(matched.subset([s]).sample(prompts, 0)[0], dtype=object) == truth))
        ks = list(range(s * Ps, (s + 1) * Ps))

        def scored(m, T, off):
            ans = samples_of(m.subset(ks), prompts, P["samples"], T, 500_000 + off + s)   # [M, S, Q]
            flat = [p + a for row in ans.reshape(-1, len(prompts)) for p, a in zip(prompts, row)]
            sc = J.score(flat).numpy().reshape(2, *ans.shape)
            return ans, sc[0], sc[1]

        out = {"seed": s}
        a, sc, sn = scored(normal, TEMPERATURE, 0)
        out["n4"] = sweep(a, sc, sn, truth, 4, P["bootstrap"], s)
        out["n8"] = sweep(a, sc, sn, truth, min(8, Ps), P["bootstrap"], s + 1)
        right = (a == truth[None, None, :]).ravel()
        out["judge_auc"], out["shuffled_auc"] = auc(sc.ravel(), right), auc(sn.ravel(), right)
        a, sc, sn = scored(normal, 1.0, 50_000)
        out["t1_n4"] = sweep(a, sc, sn, truth, 4, P["bootstrap"], s + 2)
        a, sc, sn = scored(weak, TEMPERATURE, 90_000)
        out["weak_n4"] = sweep(a, sc, sn, truth, 4, P["bootstrap"], s + 3)
        out["matched_acc"] = matched_acc
        out["matched_steps"] = matched_steps
        out["judge_train_pos_neg"] = [jmeta["n_pos"][s], jmeta["n_neg"][s]] if "n_pos" in jmeta else None
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
        out["train_seconds"] = train_s / S
        runs.append(out)
    return runs, train_s

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
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


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
    runs, _ = run_seeds(P)
    for r in runs:
        print(f"seed {r['seed']}: vote {r['n4']['vote']:.3f} judge {r['n4']['judge']:.3f} pass {r['n4']['pass']:.3f} "
              f"matched {r['matched_acc']:.3f} weak vote {r['weak_n4']['vote']:.3f} judge {r['weak_n4']['judge']:.3f} "
              f"")

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
         "runtime_s": budget["wall_minutes"] * 60, "peak_ram_gb": budget["peak_ram_gb"],
         "budget": budget, "budget_ok": budget_ok, "criteria": criteria.CRITERIA,
         "python": platform.python_version(), "torch": torch.__version__}
    (RESULTS / f"verdict{sfx}.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    (RESULTS / f"curves{sfx}.json").write_text(json.dumps(runs, indent=2, default=float), encoding="utf-8")
    write_md(RESULTS / f"results{sfx}.md", v, runs)
    try:
        import plots
        plots.selection(runs, RESULTS / f"selection_n4{sfx}.png")
    except Exception as e:  # plots never break a run
        print(f"plot skipped: {e}")
    if not all(budget_ok.values()):
        print("budget exceeded", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
