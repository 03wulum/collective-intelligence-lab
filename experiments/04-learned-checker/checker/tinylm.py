"""Tiny character-level GPT members and the tiny judge, both GPU-aware.

Members are the same architecture as experiment 01 (2 layers, d=64, ~103k parameters).
The judge reuses the same body and swaps the next-character head for a single
"is this sum right?" logit read at the end-of-answer token.
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
ALLOWED = set(VOCAB) - {"$"}


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids) -> str:
    return "".join(VOCAB[i] for i in ids)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class Body(nn.Module):
    def __init__(self, c: TinyConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(len(VOCAB), c.d_model)
        self.pos = nn.Embedding(c.block, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln = nn.LayerNorm(c.d_model)

    def forward(self, idx):
        T = idx.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x, mask)
        return self.ln(x)


class TinyGPT(nn.Module):
    """A swarm member: predicts the next character."""

    def __init__(self, c: TinyConfig):
        super().__init__()
        self.c = c
        self.body = Body(c)
        self.head = nn.Linear(c.d_model, len(VOCAB))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx):
        return self.head(self.body(idx))

    @torch.no_grad()
    def sample(self, prompts: list[str], temperature: float = 1.0, max_new: int = 6,
               generator: torch.Generator | None = None) -> list[str]:
        """One completion per prompt (all prompts share a length). Fully batched, GPU-friendly."""
        self.eval()
        dev = next(self.parameters()).device
        idx = torch.tensor([encode(p) for p in prompts], device=dev)
        B = idx.shape[0]
        out = torch.full((B, max_new), EOS, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(max_new):
            logits = self(idx[:, -self.c.block:])[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                probs = F.softmax(logits.float() / temperature, -1)
                nxt = torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(-1).to(dev)
            nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
            out[:, t] = nxt
            done |= nxt == EOS
            if bool(done.all()):
                break
            idx = torch.cat([idx, nxt[:, None]], 1)
        res = []
        for row in out.tolist():
            s = []
            for i in row:
                if i == EOS:
                    break
                s.append(i)
            res.append(decode(s))
        return res


class Judge(nn.Module):
    """Reads 'a+b=c$' and outputs one logit: is c the right sum?"""

    def __init__(self, c: TinyConfig):
        super().__init__()
        self.c = c
        self.body = Body(c)
        self.head = nn.Linear(c.d_model, 1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, last):
        h = self.body(idx)
        return self.head(h[torch.arange(idx.shape[0], device=idx.device), last]).squeeze(-1)

    @torch.no_grad()
    def score(self, strings: list[str], bs: int = 4096) -> list[float]:
        """P(correct) for each 'a+b=c' string. Unparseable strings score 0."""
        self.eval()
        dev = next(self.parameters()).device
        out = [0.0] * len(strings)
        ok = [i for i, s in enumerate(strings) if 0 < len(s) < self.c.block and set(s) <= ALLOWED]
        for k in range(0, len(ok), bs):
            ids = ok[k:k + bs]
            x, last = pack([strings[i] for i in ids], dev)
            p = torch.sigmoid(self(x, last).float()).tolist()
            for i, v in zip(ids, p):
                out[i] = v
        return out


def pack(strings: list[str], dev) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [encode(s + "$") for s in strings]
    L = max(map(len, seqs))
    x = torch.full((len(seqs), L), EOS, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    last = torch.tensor([len(s) - 1 for s in seqs])
    return x.to(dev), last.to(dev)


def _lm_batch(problems: list[Problem], rng: random.Random, bs: int):
    ps = [problems[rng.randrange(len(problems))] for _ in range(bs)]
    seqs = [encode(p.prompt + p.answer + "$") for p in ps]
    L = max(map(len, seqs))
    x = torch.full((bs, L - 1), EOS)
    y = torch.full((bs, L - 1), -100)  # no loss on the prompt
    for i, (s, p) in enumerate(zip(seqs, ps)):
        x[i, : len(s) - 1] = torch.tensor(s[:-1])
        a0 = len(p.prompt) - 1
        y[i, a0 : len(s) - 1] = torch.tensor(s[a0 + 1 :])
    return x, y


def _cosine(opt, lr, step, steps):
    for g in opt.param_groups:
        g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))


def train_member(cfg: TinyConfig, seed: int, steps: int = 1250, bs: int = 64, lr: float = 3e-3,
                 n_train: int = 4000, dev=None) -> tuple[TinyGPT, dict]:
    """Same recipe as experiment 01: each seed sees its own random training subset (bagging)."""
    dev = dev or device()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    data = make_problems(n_train, seed=10_000 + seed)
    m = TinyGPT(cfg).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1)
    t0 = time.time()
    m.train()
    for step in range(steps):
        _cosine(opt, lr, step, steps)
        x, y = _lm_batch(data, rng, bs)
        x, y = x.to(dev), y.to(dev)
        loss = F.cross_entropy(m(x).reshape(-1, len(VOCAB)), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    tokens = steps * bs * 12  # ~12 positions per "abc+def=ghij$" sequence
    meta = {"seed": seed, "steps": steps, "bs": bs, "params": m.n_params(), "tokens": tokens,
            "train_flops": 6 * m.n_params() * tokens, "train_seconds": time.time() - t0,
            "device": str(dev), **asdict(cfg)}
    return m, meta


def train_judge(cfg: TinyConfig, strings: list[str], labels: list[int], seed: int, steps: int = 1500,
                bs: int = 128, lr: float = 3e-3, dev=None) -> tuple[Judge, dict]:
    """Class-balanced binary training on members' own sampled answers."""
    dev = dev or device()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    pos = [s for s, y in zip(strings, labels) if y]
    neg = [s for s, y in zip(strings, labels) if not y]
    if not pos or not neg:
        raise ValueError("judge needs both right and wrong examples")
    j = Judge(cfg).to(dev)
    opt = torch.optim.AdamW(j.parameters(), lr=lr, weight_decay=0.1)
    t0 = time.time()
    j.train()
    half = bs // 2
    for step in range(steps):
        _cosine(opt, lr, step, steps)
        batch = [pos[rng.randrange(len(pos))] for _ in range(half)] + [neg[rng.randrange(len(neg))] for _ in range(half)]
        x, last = pack(batch, dev)
        y = torch.tensor([1.0] * half + [0.0] * half, device=dev)
        loss = F.binary_cross_entropy_with_logits(j(x, last), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    tokens = steps * bs * 12
    meta = {"seed": seed, "steps": steps, "bs": bs, "params": j.n_params(), "tokens": tokens,
            "train_flops": 6 * j.n_params() * tokens, "train_seconds": time.time() - t0,
            "n_pos": len(pos), "n_neg": len(neg), "device": str(dev)}
    return j, meta
