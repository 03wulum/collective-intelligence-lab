import numpy as np
import torch

import criteria
from checker.tasks import make_problems
from checker.tinylm import Judge, TinyConfig, TinyGPT, train_judge
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
    m, j = TinyGPT(CFG), Judge(CFG)
    assert j.n_params() <= m.n_params()
    out = m.sample(["123+456="] * 3, temperature=0)
    assert len(out) == 3 and all(isinstance(s, str) for s in out)
    sc = j.score(["123+456=579", "bad!", ""])
    assert len(sc) == 3 and sc[1] == 0.0 and sc[2] == 0.0


def test_judge_learns_simple_rule():
    torch.manual_seed(0)
    ps = make_problems(200, seed=1)
    strings = [p.prompt + p.answer for p in ps] + [p.prompt + "0" for p in ps]
    labels = [1] * 200 + [0] * 200
    j, meta = train_judge(CFG, strings, labels, seed=0, steps=150, bs=64, dev=torch.device("cpu"))
    s = j.score(strings)
    assert auc(s, labels) > 0.9


def test_verdict_logic():
    allp = {c["id"]: True for c in criteria.CRITERIA}
    assert criteria.verdict(allp, {"C1": False}) == "PASS"
    assert criteria.verdict({**allp, "C1": False}, {"C1": False}) == "FAIL"
    assert criteria.verdict({**allp, "C2": False}, {"C1": False}) == "INCONCLUSIVE"
    assert criteria.verdict(allp, {"C1": True}) == "INCONCLUSIVE"
