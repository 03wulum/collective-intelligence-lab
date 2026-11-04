"""A tiny character-level GPT, trainable on a laptop CPU in under a minute.

These are the "mini models" of the swarm when no pretrained LLM is available.
They are real neural nets with real, independent errors, which is exactly what
the scaling question depends on.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import Problem, make_problems

VOCAB = list("0123456789+=$")  # '$' = end of answer
STOI = {c: i for i, c in enumerate(VOCAB)}
EOS = STOI["$"]


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join(VOCAB[i] for i in ids)


@dataclass
class TinyConfig:
    d_model: int = 64
    n_layer: int = 2
    n_head: int = 4
    block: int = 24


class Block(nn.Module):
    def __init__(self, c: TinyConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.attn = nn.MultiheadAttention(c.d_model, c.n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.mlp = nn.Sequential(nn.Linear(c.d_model, 4 * c.d_model), nn.GELU(), nn.Linear(4 * c.d_model, c.d_model))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, c: TinyConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(len(VOCAB), c.d_model)
        self.pos = nn.Embedding(c.block, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, len(VOCAB))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx):
        T = idx.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))

    @torch.no_grad()
    def sample(self, prompts: list[str], temperature: float = 1.0, max_new: int = 6,
               generator: torch.Generator | None = None) -> list[tuple[str, float]]:
        """Sample one completion per prompt. Returns (answer, total logprob)."""
        self.eval()
        # all prompts in a benchmark share a length, so batch them directly
        idx = torch.tensor([encode(p) for p in prompts])
        B = idx.shape[0]
        out = [[] for _ in range(B)]
        lp = torch.zeros(B)
        done = torch.zeros(B, dtype=torch.bool)
        for _ in range(max_new):
            logits = self(idx[:, -self.c.block:])[:, -1, :]
            logp = F.log_softmax(logits, -1)
            if temperature <= 0:
                nxt = logp.argmax(-1)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=generator).squeeze(-1)
            lp += torch.where(done, torch.zeros(B), logp.gather(1, nxt[:, None]).squeeze(1))
            for i in range(B):
                if not done[i]:
                    if nxt[i].item() == EOS:
                        done[i] = True
                    else:
                        out[i].append(nxt[i].item())
            if done.all():
                break
            idx = torch.cat([idx, nxt[:, None]], 1)
        return [(decode(o), float(l)) for o, l in zip(out, lp)]


def _batch(problems: list[Problem], rng: random.Random, bs: int):
    ps = [problems[rng.randrange(len(problems))] for _ in range(bs)]
    seqs = [encode(p.prompt + p.answer + "$") for p in ps]
    L = max(map(len, seqs))
    x = torch.full((bs, L - 1), EOS)
    y = torch.full((bs, L - 1), -100)  # ignore loss on the prompt
    for i, (s, p) in enumerate(zip(seqs, ps)):
        x[i, : len(s) - 1] = torch.tensor(s[:-1])
        a0 = len(p.prompt) - 1
        y[i, a0 : len(s) - 1] = torch.tensor(s[a0 + 1 :])
    return x, y


def train(cfg: TinyConfig, seed: int, steps: int = 1500, bs: int = 64, lr: float = 3e-3,
          n_train: int = 4000, digits: int = 3, operands: int = 2, verbose: bool = False) -> tuple[TinyGPT, dict]:
    """Train one tiny model. Each seed sees a different random training subset (bagging),
    so swarm members genuinely disagree."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    data = make_problems(n_train, digits, operands, seed=10_000 + seed)
    m = TinyGPT(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1)
    t0 = time.time()
    for step in range(steps):
        lr_t = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr_t
        x, y = _batch(data, rng, bs)
        loss = F.cross_entropy(m(x).reshape(-1, len(VOCAB)), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if verbose and step % 250 == 0:
            print(f"  step {step} loss {loss.item():.3f}")
    meta = {"seed": seed, "steps": steps, "params": m.n_params(), "train_seconds": time.time() - t0,
            "train_flops": 6 * m.n_params() * steps * bs * (len(data[0].prompt) + 4), **asdict(cfg)}
    return m, meta
