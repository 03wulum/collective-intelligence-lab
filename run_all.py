"""Run every experiment and print one verdict table.

    python run_all.py          # --quick for each experiment (CI)
    python run_all.py --full   # full runs
    python run_all.py --clean  # delete every experiment's caches and report the space freed
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    if a.clean:
        for exp in sorted((ROOT / "experiments").glob("*/")):
            if (exp / "run.py").exists():
                subprocess.run([sys.executable, "run.py", "--clean"], cwd=exp)
        return
    rows, failed = [], False
    for exp in sorted((ROOT / "experiments").glob("*/")):
        if not (exp / "run.py").exists():
            continue
        cmd = [sys.executable, "run.py"] + ([] if a.full else ["--quick"])
        print(f"\n=== {exp.name}: {' '.join(cmd[1:])}", flush=True)
        r = subprocess.run(cmd, cwd=exp)
        vfile = exp / "results" / ("verdict.json" if a.full else "verdict_quick.json")
        if r.returncode != 0 or not vfile.exists():
            rows.append((exp.name, "CRASH" if r.returncode != 2 else "BUDGET BLOWN", "-"))
            failed = True
            continue
        v = json.load(open(vfile))
        rows.append((exp.name, v["verdict"] + (" ✨" if v.get("intriguing") else ""), f"{v['runtime_s']:.0f}s, {v['peak_ram_gb']:.2f} GB RAM"))
    print("\n" + "-" * 72)
    print(f"{'experiment':<28} {'verdict':<16} cost   ({'full' if a.full else 'quick'} mode)")
    for name, verdict, cost in rows:
        print(f"{name:<28} {verdict:<16} {cost}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
