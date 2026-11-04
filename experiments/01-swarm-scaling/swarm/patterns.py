"""Collaboration patterns and communication topologies.

Aggregators (pure functions over answers)
  plurality         – "More Agents Is All You Need" (Li et al., 2024): sample & vote.
  weighted          – confidence-weighted vote (sum of sequence probabilities).
  adaptive          – stop adding agents once the leader is ``margin`` votes ahead
                      (Adaptive-Consistency, Aggarwal et al., 2023). Same accuracy, less compute.
  oracle (pass@N)   – any agent correct. The ceiling for a perfect verifier/aggregator.

Interactive patterns (need agents that can read peers, i.e. LLM backends)
  debate            – Du et al., 2023 multi-agent debate, over an arbitrary topology
                      (full / ring / star / random-k: Li et al., 2024 "sparse topology").
  mixture_of_agents – Wang et al., 2024: layers of proposers, then one aggregator.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Callable, Sequence

from .tasks import Problem, extract_answer

# ----------------------------------------------------------------------------- aggregators


def plurality(answers: Sequence[str | None]) -> str | None:
    c = Counter(a for a in answers if a)
    if not c:
        return None
    best = max(c.values())
    # deterministic tie-break: first answer to reach the max count
    return next(a for a in answers if a and c[a] == best)


def weighted(answers: Sequence[str | None], logprobs: Sequence[float]) -> str | None:
    score: dict[str, float] = {}
    for a, lp in zip(answers, logprobs):
        if a:
            score[a] = score.get(a, 0.0) + math.exp(lp)
    return max(score, key=score.get) if score else None


def adaptive(answers: Sequence[str | None], margin: int = 2, min_agents: int = 1) -> tuple[str | None, int]:
    """Consume agents in order; stop when the leader leads the runner-up by ``margin``.
    Returns (answer, agents_used)."""
    c: Counter = Counter()
    for i, a in enumerate(answers, 1):
        if a:
            c[a] += 1
        top = c.most_common(2) + [(None, 0), (None, 0)]
        if i >= min_agents and top[0][1] - top[1][1] >= margin:
            return top[0][0], i
    return plurality(answers), len(answers)


def oracle(answers: Sequence[str | None], truth: str) -> bool:
    return truth in answers


# ----------------------------------------------------------------------------- topologies


def topology(name: str, n: int, k: int = 2, seed: int = 0) -> list[list[int]]:
    """Return neighbors[i] = agents whose messages agent i reads each round."""
    if name == "none":
        return [[] for _ in range(n)]
    if name == "full":
        return [[j for j in range(n) if j != i] for i in range(n)]
    if name == "ring":
        return [sorted({(i - 1) % n, (i + 1) % n} - {i}) for i in range(n)]
    if name == "star":  # agent 0 is the hub
        return [[j for j in range(1, n)]] + [[0] for _ in range(1, n)]
    if name == "random":
        rng = random.Random(seed)
        return [rng.sample([j for j in range(n) if j != i], min(k, n - 1)) for i in range(n)]
    raise ValueError(f"unknown topology {name!r}")


# ----------------------------------------------------------------------------- interactive patterns

SYSTEM = "You are a careful assistant. Think briefly, then give the final integer on the last line."


def _ask(agent, prompt: str, temperature: float, seed: int) -> str:
    return agent.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                      temperature=temperature, seed=seed)


def _peer_block(texts: list[str], limit: int = 400) -> str:
    return "\n".join(f"- {t.strip()[-limit:]}" for t in texts)


def debate(agents: list, problem: Problem, rounds: int = 2, topo: str = "full", temperature: float = 0.7,
           seed: int = 0, on_call: Callable | None = None) -> dict:
    n = len(agents)
    nbrs = topology(topo, n, seed=seed)
    q = problem.as_question()
    texts = [_ask(a, q, temperature, seed * 1000 + i) for i, a in enumerate(agents)]
    history = [[extract_answer(t) for t in texts]]
    for r in range(1, rounds + 1):
        new = []
        for i, a in enumerate(agents):
            if not nbrs[i]:
                new.append(texts[i])
                continue
            prompt = (f"{q}\n\nOther agents answered:\n{_peer_block([texts[j] for j in nbrs[i]])}\n\n"
                      f"Your previous answer: {history[-1][i]}\n"
                      "Check the reasoning, then give your updated final integer on the last line.")
            new.append(_ask(a, prompt, temperature, seed * 1000 + r * 100 + i))
        texts = new
        history.append([extract_answer(t) for t in texts])
    return {"answer": plurality(history[-1]), "history": history,
            "calls": n * (rounds + 1) - sum(1 for x in nbrs if not x) * rounds}


def mixture_of_agents(proposers: list, aggregator, problem: Problem, layers: int = 2,
                      temperature: float = 0.7, seed: int = 0) -> dict:
    q = problem.as_question()
    texts = [_ask(a, q, temperature, seed * 1000 + i) for i, a in enumerate(proposers)]
    calls = len(proposers)
    for layer in range(1, layers):
        ctx = f"{q}\n\nResponses from other models:\n{_peer_block(texts)}\n\nWrite an improved answer."
        texts = [_ask(a, ctx, temperature, seed * 1000 + layer * 100 + i) for i, a in enumerate(proposers)]
        calls += len(proposers)
    final = _ask(aggregator, f"{q}\n\nCandidate responses:\n{_peer_block(texts)}\n\n"
                 "Synthesize them into the single most likely correct final integer.", 0.0, seed)
    return {"answer": extract_answer(final), "proposals": [extract_answer(t) for t in texts], "calls": calls + 1}
