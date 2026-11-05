import numpy as np
import torch

import criteria
from checker.tasks import make_problems
from checker.tinylm import VOCAB, Stacked, TinyConfig, train_judges, train_members
from run import auc, gap_closed, select, sweep, vote

CFG = TinyConfig(d_model=16, n_layer=1, n_head=2)


def test_select_prefers_judge_then_vote():
    assert select(["5", "6", "6"], [0.9, 0.2, 0.3]) == "5"
    assert select(["5", "6", "6"], [0.5, 0.5, 0.1]) == "6"  # tie on score -> vote
    assert vote(["1", "2", "2"]) == "2"


def test_gap_closed():
    assert gap_closed({"pass": 1.0, "vote": 0.9, "judge": 0.95}) == 0.5 or abs(
        gap_closed({"pass": 1.0, "vote": 0.9, "judge": 0.95}) - 0.5) < 1e-9
    assert gap_closed({"pass": 0.9, "vote": 0.9, "judge": 0.9}) == 0.0


def test_auc():
    assert auc([0.1, 0.9], [0, 1]) == 1.0
    assert auc([0.9, 0.1], [0, 1]) == 0.0


def test_sweep_perfect_judge_closes_gap():
    truth = np.array(["1", "2", "3"], dtype=object)
    ans = np.array([[["1", "9", "3"]], [["8", "2", "7"]], [["8", "9", "7"]], [["8", "2", "3"]]], dtype=object)
    sc = (ans == truth[None, None, :]).astype(float)
    r = sweep(ans, sc, np.zeros_like(sc), truth, 4, 3, 0)
    assert r["pass"] == 1.0 and r["judge"] == 1.0 and r["vote"] < 1.0
    assert gap_closed(r) == 1.0 and gap_closed(r, "shuffled_judge") < 1.0


def test_models_shapes_and_sizes():
    m, j = Stacked(CFG, 3, len(VOCAB)), Stacked(CFG, 2, 1)
    assert j.params_per_model() <= m.params_per_model()
    out = m.sample(["123+456="] * 4, temperature=0.7, generator=torch.Generator().manual_seed(0))
    assert len(out) == 3 and len(out[0]) == 4 and all(isinstance(s, str) for s in out[0])
    sc = j.score(["123+456=579", "bad!", ""])
    assert sc.shape == (2, 3) and sc[0, 1] == 0.0 and sc[0, 2] == 0.0
    sub = m.subset([2])
    assert torch.equal(sub.tok[0], m.tok[2])


def test_stacked_models_are_independent():
    """Training K models together must equal training each alone (no leakage across the model axis)."""
    torch.manual_seed(0)
    m = Stacked(CFG, 2, len(VOCAB))
    x = torch.randint(0, 12, (2, 3, 5))
    m(x)[0].sum().backward()
    assert m.tok.grad[1].abs().sum() == 0 and m.tok.grad[0].abs().sum() > 0


def test_members_learn_and_judge_learns_simple_rule():
    m, meta = train_members(CFG, [1, 2], steps=30, n_train=200, dev=torch.device("cpu"))
    assert meta["params"] == m.params_per_model() and m.K == 2
    ps = make_problems(200, seed=1)
    strings = [p.prompt + p.answer for p in ps] + [p.prompt + "0" for p in ps]
    labels = [1] * 200 + [0] * 200
    j, jm = train_judges(CFG, [(strings, labels), (strings, labels[::-1])], seed=0, steps=150, bs=64,
                         dev=torch.device("cpu"))
    s = j.score(strings)
    assert auc(s[0].tolist(), labels) > 0.9 and auc(s[1].tolist(), labels) < 0.1


def test_verdict_logic():
    allp = {c["id"]: True for c in criteria.CRITERIA}
    assert criteria.verdict(allp, {"C1": False}) == "PASS"
    assert criteria.verdict({**allp, "C1": False}, {"C1": False}) == "FAIL"
    assert criteria.verdict({**allp, "C2": False}, {"C1": False}) == "INCONCLUSIVE"
    assert criteria.verdict(allp, {"C1": True}) == "INCONCLUSIVE"
