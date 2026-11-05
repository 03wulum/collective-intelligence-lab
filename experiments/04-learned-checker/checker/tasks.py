"""Auto-graded arithmetic benchmark.

Generated (not downloaded) so there is no contamination and no network needed.
Each problem is a string like ``"417+389="`` with integer answer ``"806"``.
Difficulty is controlled by digit count and number of operands.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    prompt: str  # e.g. "417+389="
    answer: str  # e.g. "806"

    def as_question(self) -> str:
        """Natural-language form for LLM backends."""
        expr = self.prompt.rstrip("=")
        return f"What is {expr}? Reply with only the final integer."


def make_problems(n: int, digits: int = 3, operands: int = 2, seed: int = 0) -> list[Problem]:
    rng = random.Random(seed)
    lo, hi = 10 ** (digits - 1), 10**digits - 1
    out = []
    for _ in range(n):
        xs = [rng.randint(lo, hi) for _ in range(operands)]
        expr = "+".join(map(str, xs))
        out.append(Problem(prompt=expr + "=", answer=str(sum(xs))))
    return out


_INT = re.compile(r"-?\d[\d,]*")


def extract_answer(text: str) -> str | None:
    """Last integer in the text (LLMs often restate the question first)."""
    m = _INT.findall(text or "")
    return m[-1].replace(",", "") if m else None


def is_correct(pred: str | None, problem: Problem) -> bool:
    return pred is not None and pred == problem.answer
