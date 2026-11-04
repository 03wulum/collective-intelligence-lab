"""Sweep swarm size N x pattern x topology on real small LLMs served locally.

Works with anything that speaks the OpenAI chat API on your machine:

    # Ollama
    ollama pull qwen2.5:0.5b
    python llm/run_llm_sweep.py

    # llama.cpp
    llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080 -np 8
    python llm/run_llm_sweep.py --base-url http://localhost:8080/v1

    # no server? dry-run the whole pipeline on a mock model
    python llm/run_llm_sweep.py --mock

Results are appended to llm/results/<model>.jsonl (one line per question x setting),
so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import patterns  # noqa: E402
from swarm.backends import MockLLMBackend, OpenAICompatBackend  # noqa: E402
from swarm.tasks import extract_answer, make_problems  # noqa: E402


def settings(cfg):
    for N in cfg["swarm_sizes"]:
        yield {"pattern": "vote", "N": N}
        if N >= 2:
            for topo in cfg["debate"]["topologies"]:
                if topo == "star" and N < 3:
                    continue
                yield {"pattern": "debate", "N": N, "topology": topo, "rounds": cfg["debate"]["rounds"]}
            if cfg["moa"]["enabled"]:
                yield {"pattern": "moa", "N": N, "layers": cfg["moa"]["layers"]}


def run_one(s, agents, problem, cfg, qi):
    T = cfg["temperature"]
    if s["pattern"] == "vote":
        texts = [patterns._ask(agents[i], problem.as_question(), T, qi * 1000 + i) for i in range(s["N"])]
        ans = [extract_answer(t) for t in texts]
        return {"answer": patterns.plurality(ans), "members": ans, "calls": s["N"]}
    if s["pattern"] == "debate":
        r = patterns.debate(agents[: s["N"]], problem, rounds=s["rounds"], topo=s["topology"], temperature=T, seed=qi)
        return {"answer": r["answer"], "members": r["history"][-1], "round0": r["history"][0], "calls": r["calls"]}
    if s["pattern"] == "moa":
        r = patterns.mixture_of_agents(agents[: s["N"]], agents[0], problem, layers=s["layers"], temperature=T, seed=qi)
        return {"answer": r["answer"], "members": r["proposals"], "calls": r["calls"]}
    raise ValueError(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "llm.yaml"))
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    models = [a.model] if a.model else cfg["models"]
    base = a.base_url or cfg["base_url"]
    t = cfg["task"]
    probs = make_problems(t["questions"], t["digits"], t["operands"], seed=t["seed"])
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_n = max(cfg["swarm_sizes"])
    for model in models:
        # agent i = same weights, different sampling seed. Add more models to `models`
        # and set heterogeneous: true to cycle different models across agents.
        pool = models if cfg.get("heterogeneous") else [model]
        if a.mock:
            agents = [MockLLMBackend(p=0.55, conform=0.5, seed=i) for i in range(max_n)]
            tag = "mock"
        else:
            agents = [OpenAICompatBackend(pool[i % len(pool)], base, max_tokens=cfg["max_tokens"])
                      for i in range(max_n)]
            tag = "+".join(pool) if cfg.get("heterogeneous") else model
        path = out_dir / f"{tag.replace(':', '_').replace('/', '_')}.jsonl"
        done = set()
        if path.exists():
            for line in open(path):
                r = json.loads(line)
                done.add((json.dumps(r["setting"], sort_keys=True), r["q"]))
        todo = [(s, qi) for s in settings(cfg) for qi in range(len(probs))
                if (json.dumps(s, sort_keys=True), qi) not in done]
        print(f"[{tag}] {len(todo)} runs to do ({len(done)} cached)", flush=True)
        t0 = time.time()

        def work(item):
            s, qi = item
            r = run_one(s, agents, probs[qi], cfg, qi)
            return {"setting": s, "q": qi, "truth": probs[qi].answer, **r,
                    "correct": r["answer"] == probs[qi].answer}

        with open(path, "a") as f, ThreadPoolExecutor(cfg["concurrency"]) as ex:
            for i, row in enumerate(ex.map(work, todo), 1):
                f.write(json.dumps(row) + "\n")
                if i % 25 == 0:
                    f.flush()
                    print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
        print(f"[{tag}] wrote {path}")


if __name__ == "__main__":
    main()
