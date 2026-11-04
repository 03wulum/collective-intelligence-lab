"""Model backends. Every swarm member is an `Agent` wrapping one backend.

* ``TinyLMBackend``  – a from-scratch TinyGPT (no network, trains in ~12 s on CPU).
* ``OpenAICompatBackend`` – any local server speaking ``/v1/chat/completions``:
  Ollama (``http://localhost:11434/v1``), llama.cpp ``llama-server``, LM Studio, vLLM.
  This is how you run the swarm on real small LLMs (Qwen2.5-0.5B, SmolLM2-360M, ...).
* ``MockLLMBackend`` – deterministic fake used in tests.
"""
from __future__ import annotations

import json
import random
import re
import urllib.request
import zlib
from dataclasses import dataclass, field

from .tasks import Problem, extract_answer


@dataclass
class Reply:
    text: str
    answer: str | None
    logprob: float | None = None


class TinyLMBackend:
    """Tiny models answer the raw prompt; they cannot read peers' messages."""

    can_read_context = False

    def __init__(self, model, name: str = "tiny"):
        self.model, self.name = model, name

    def answer_batch(self, problems: list[Problem], temperature: float, generator=None) -> list[Reply]:
        outs = self.model.sample([p.prompt for p in problems], temperature=temperature, generator=generator)
        return [Reply(text=a, answer=a or None, logprob=lp) for a, lp in outs]


class OpenAICompatBackend:
    can_read_context = True

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", api_key: str = "local",
                 max_tokens: int = 256, timeout: float = 120.0):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.max_tokens, self.timeout = max_tokens, timeout
        self.name = model

    def chat(self, messages: list[dict], temperature: float, seed: int | None = None) -> str:
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": self.max_tokens}
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"]


@dataclass
class MockLLMBackend:
    """Answers correctly with probability ``p``; otherwise a plausible wrong answer.
    If shown peers' answers it adopts the majority with probability ``conform``."""

    p: float = 0.6
    conform: float = 0.5
    seed: int = 0
    name: str = "mock"
    can_read_context: bool = True
    calls: int = field(default=0, init=False)

    def chat(self, messages: list[dict], temperature: float, seed: int | None = None) -> str:
        self.calls += 1
        rng = random.Random(zlib.crc32(f"{self.seed}|{seed}|{messages[-1]['content']}".encode()))
        content = messages[-1]["content"]
        m = re.search(r"What is ([\d+]+)\?", content)
        truth = str(sum(map(int, m.group(1).split("+")))) if m else "0"
        peers = [extract_answer(line) for line in content.splitlines() if line.startswith("- ")]
        peers = [x for x in peers if x]
        if peers and rng.random() < self.conform:
            return max(set(peers), key=peers.count)
        return truth if rng.random() < self.p else str(int(truth) + rng.choice([-10, -1, 1, 10, 100]))
