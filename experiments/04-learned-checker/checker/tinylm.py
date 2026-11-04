"""Tiny character-level GPT members and the tiny judge, trained K at a time on one GPU.

Every tensor carries a leading model axis K, so a whole swarm (e.g. 24 members across
3 seeds) trains in one process as one batched computation. The models never share
weights or data; this is only a faster way to run K independent trainings, which keeps
the GPU busy instead of launching thousands of tiny kernels per model.

Members match experiment 01's shape (2 layers, d=64, 4 heads, ~103k parameters).
The judge has the same body with a single "is this sum right?" output read at '$'.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import make_problems

VOCAB = list("0123456789+=$")  # '$' = end of answer (also padding)
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


def _ln(x, w, b):
    return F.layer_norm(x, x.shape[-1:]) * w[:, None, None, :] + b[:, None, None, :]


def _lin(x, w, b):  # x [K,B,T,i], w [K,i,o], b [K,o]
    return torch.einsum("kbti,kio->kbto", x, w) + b[:, None, None, :]


class Stacked(nn.Module):
    """K independent transformers. out=len(VOCAB) for members, out=1 for judges."""

    def __init__(self, c: TinyConfig, K: int, out: int):
        super().__init__()
        self.c, self.K, self.out = c, K, out
        d, V = c.d_model, len(VOCAB)

        def P(*shape, std=0.02):
            return nn.Parameter(torch.randn(K, *shape) * std)

        def Z(*shape):
            return nn.Parameter(torch.zeros(K, *shape))

        def O(*shape):
            return nn.Parameter(torch.ones(K, *shape))

        self.tok, self.pos = P(V, d), P(c.block, d)
        self.layers = nn.ModuleList()
        for _ in range(c.n_layer):
            L = nn.ParameterDict({"ln1w": O(d), "ln1b": Z(d), "qkv": P(d, 3 * d), "qkvb": Z(3 * d),
                                  "proj": P(d, d, std=0.02 / math.sqrt(2 * c.n_layer)), "projb": Z(d),
                                  "ln2w": O(d), "ln2b": Z(d), "fc1": P(d, 4 * d), "fc1b": Z(4 * d),
                                  "fc2": P(4 * d, d, std=0.02 / math.sqrt(2 * c.n_layer)), "fc2b": Z(d)})
            self.layers.append(L)
        self.lnw, self.lnb = O(d), Z(d)
        self.head, self.headb = P(d, out), Z(out)

    def params_per_model(self) -> int:
        return sum(p.numel() for p in self.parameters()) // self.K

    def subset(self, ks: list[int]) -> "Stacked":
        m = Stacked(self.c, len(ks), self.out).to(self.tok.device)
        idx = torch.tensor(ks, device=self.tok.device)
        with torch.no_grad():
            for (_, p_new), (_, p_old) in zip(m.named_parameters(), self.named_parameters()):
                p_new.copy_(p_old.index_select(0, idx))
        return m

    def hidden(self, idx):  # idx [K,B,T] -> [K,B,T,d]
        K, B, T = idx.shape
        d, H = self.c.d_model, self.c.n_head
        kk = torch.arange(K, device=idx.device)[:, None, None]
        x = self.tok[kk, idx] + self.pos[:, None, :T, :]
        for L in self.layers:
            h = _ln(x, L["ln1w"], L["ln1b"])
            q, k, v = _lin(h, L["qkv"], L["qkvb"]).split(d, -1)
            q, k, v = (t.reshape(K * B, T, H, d // H).transpose(1, 2) for t in (q, k, v))
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(K, B, T, d)
            x = x + _lin(a, L["proj"], L["projb"])
            h = _ln(x, L["ln2w"], L["ln2b"])
            x = x + _lin(F.gelu(_lin(h, L["fc1"], L["fc1b"])), L["fc2"], L["fc2b"])
        return _ln(x, self.lnw, self.lnb)

    def forward(self, idx):
        return _lin(self.hidden(idx), self.head, self.headb)

    # ------------------------------------------------------------------ member sampling
    @torch.no_grad()
    def sample(self, prompts: list[str], temperature: float, generator: torch.Generator | None = None,
               max_new: int = 6) -> list[list[str]]:
        """One completion per prompt per model -> [K][B] strings. Prompts share a length."""
        self.eval()
        dev = self.tok.device
        idx = torch.tensor([encode(p) for p in prompts], device=dev)[None].expand(self.K, -1, -1).contiguous()
        K, B, _ = idx.shape
        out = torch.full((K, B, max_new), EOS, device=dev)
        done = torch.zeros(K, B, dtype=torch.bool, device=dev)
        for t in range(max_new):
            logits = self(idx[:, :, -self.c.block:])[:, :, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                u = torch.rand(logits.shape, device=dev, generator=generator).clamp_(1e-10, 1.0)
                nxt = (logits / temperature - torch.log(-torch.log(u))).argmax(-1)  # Gumbel-max sampling
            nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
            out[:, :, t] = nxt
            done |= nxt == EOS
            if bool(done.all()):
                break
            idx = torch.cat([idx, nxt[:, :, None]], 2)
        res = []
        for mk in out.tolist():
            row = []
            for seq in mk:
                s = []
                for i in seq:
                    if i == EOS:
                        break
                    s.append(i)
                row.append(decode(s))
            res.append(row)
        return res

    # ------------------------------------------------------------------ judge scoring
    @torch.no_grad()
    def score(self, strings: list[str], bs: int = 8192) -> torch.Tensor:
        """P(correct) from each of the K judges for each 'a+b=c' string -> [K, N]. Junk scores 0."""
        self.eval()
        dev = self.tok.device
        res = torch.zeros(self.K, len(strings))
        ok = [i for i, s in enumerate(strings) if 0 < len(s) < self.c.block and set(s) <= ALLOWED]
        for j in range(0, len(ok), bs):
            ids = ok[j:j + bs]
            x, last = pack([strings[i] for i in ids], dev)
            h = self.hidden(x[None].expand(self.K, -1, -1))
            logit = _lin(h, self.head, self.headb)[..., 0]                       # [K,B,T]
            p = torch.sigmoid(logit.gather(2, last[None, :, None].expand(self.K, -1, 1))[..., 0].float())
            res[:, ids] = p.cpu()
        return res


def pack(strings: list[str], dev) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [encode(s + "$") for s in strings]
    L = max(map(len, seqs))
    x = torch.full((len(seqs), L), EOS, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x.to(dev), torch.tensor([len(s) - 1 for s in seqs], device=dev)


def _cosine(opt, lr, step, steps):
    for g in opt.param_groups:
        g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))


def _member_data(seed: int, n_train: int):
    """Teacher-forcing tensors for one member's own bagged training set (no loss on the prompt)."""
    ps = make_problems(n_train, seed=10_000 + seed)
    L = 12  # 'abc+def=' (8) + up to 4 answer digits + '$' -> 13 tokens -> 12 transitions
    x = torch.full((n_train, L), EOS)
    y = torch.full((n_train, L), -100)
    for i, p in enumerate(ps):
        s = encode(p.prompt + p.answer + "$")
        x[i, :len(s) - 1] = torch.tensor(s[:-1])
        a0 = len(p.prompt) - 1
        y[i, a0:len(s) - 1] = torch.tensor(s[a0 + 1:])
    return x, y


def train_members(cfg: TinyConfig, seeds: list[int], steps: int, bs: int = 64, lr: float = 3e-3,
                  n_train: int = 4000, dev=None) -> tuple[Stacked, dict]:
    """Train one member per seed, all at once. Each seed has its own init, data subset and batches."""
    dev = dev or device()
    K = len(seeds)
    torch.manual_seed(seeds[0])
    m = Stacked(cfg, K, len(VOCAB)).to(dev)
    xs, ys = zip(*(_member_data(s, n_train) for s in seeds))
    X, Y = torch.stack(xs).to(dev), torch.stack(ys).to(dev)                   # [K,n,L]
    g = torch.Generator(device=dev).manual_seed(seeds[0])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1)
    kk = torch.arange(K, device=dev)[:, None]
    t0 = time.time()
    m.train()
    for step in range(steps):
        _cosine(opt, lr, step, steps)
        i = torch.randint(n_train, (K, bs), device=dev, generator=g)
        x, y = X[kk, i], Y[kk, i]
        # sum over models of each model's mean loss: gradients stay independent per model
        loss = F.cross_entropy(m(x).reshape(K * bs * x.shape[2], -1), y.reshape(-1), ignore_index=-100,
                               reduction="none").reshape(K, -1)
        loss = (loss.sum(1) / (y != -100).reshape(K, -1).sum(1)).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    tokens = steps * bs * 12
    meta = {"seeds": list(seeds), "steps": steps, "bs": bs, "params": m.params_per_model(), "tokens_each": tokens,
            "train_seconds": time.time() - t0, "device": str(dev), **asdict(cfg)}
    return m, meta


def train_judges(cfg: TinyConfig, datasets: list[tuple[list[str], list[int]]], seed: int, steps: int = 1500,
                 bs: int = 128, lr: float = 3e-3, dev=None) -> tuple[Stacked, dict]:
    """One class-balanced judge per (strings, labels) dataset, all trained at once."""
    dev = dev or device()
    K = len(datasets)
    torch.manual_seed(seed)
    j = Stacked(cfg, K, 1).to(dev)
    allx, alll = pack([s for strings, _ in datasets for s in strings], "cpu")
    pos_idx, neg_idx, off = [], [], 0
    for strings, labels in datasets:
        lab = torch.tensor(labels, dtype=torch.bool)
        if lab.all() or not lab.any():
            raise ValueError("each judge needs both right and wrong examples")
        r = torch.arange(len(strings)) + off
        pos_idx.append(r[lab])
        neg_idx.append(r[~lab])
        off += len(strings)
    pad = lambda ts: torch.nn.utils.rnn.pad_sequence(ts, batch_first=True).to(dev)  # noqa: E731
    P, N = pad(pos_idx), pad(neg_idx)
    nP = torch.tensor([len(t) for t in pos_idx], device=dev)[:, None]
    nN = torch.tensor([len(t) for t in neg_idx], device=dev)[:, None]
    allx, alll = allx.to(dev), alll.to(dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    opt = torch.optim.AdamW(j.parameters(), lr=lr, weight_decay=0.1)
    kk = torch.arange(K, device=dev)[:, None]
    half = bs // 2
    y = torch.cat([torch.ones(half), torch.zeros(half)]).to(dev)[None].expand(K, -1)
    t0 = time.time()
    j.train()
    for step in range(steps):
        _cosine(opt, lr, step, steps)
        ip = P[kk, (torch.rand(K, half, device=dev, generator=g) * nP).long()]
        ineg = N[kk, (torch.rand(K, half, device=dev, generator=g) * nN).long()]
        rows = torch.cat([ip, ineg], 1)                                          # [K, bs] dataset rows
        x, last = allx[rows], alll[rows]
        logit = _lin(j.hidden(x), j.head, j.headb)[..., 0]
        logit = logit.gather(2, last[..., None])[..., 0]
        loss = F.binary_cross_entropy_with_logits(logit, y, reduction="none").mean(1).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    meta = {"seed": seed, "steps": steps, "bs": bs, "params": j.params_per_model(), "tokens_each": steps * bs * 12,
            "train_seconds": time.time() - t0, "n_pos": [len(t) for t in pos_idx],
            "n_neg": [len(t) for t in neg_idx], "device": str(dev)}
    return j, meta
