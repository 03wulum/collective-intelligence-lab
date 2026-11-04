import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swarm import analysis, patterns  # noqa: E402
from swarm.backends import MockLLMBackend  # noqa: E402
from swarm.tasks import Problem, extract_answer, make_problems  # noqa: E402


def test_tasks_are_deterministic_and_correct():
    a, b = make_problems(5, seed=1), make_problems(5, seed=1)
    assert a == b
    for p in a:
        assert sum(map(int, p.prompt.rstrip("=").split("+"))) == int(p.answer)


def test_extract_answer_takes_last_integer():
    assert extract_answer("123+456 = 579") == "579"
    assert extract_answer("The answer is 1,234.") == "1234"
    assert extract_answer("no idea") is None


def test_plurality_and_ties():
    assert patterns.plurality(["1", "2", "2", None]) == "2"
    assert patterns.plurality(["3", "4"]) == "3"  # tie -> first seen
    assert patterns.plurality([None, None]) is None


def test_weighted_vote_prefers_confident_minority():
    assert patterns.weighted(["1", "2", "2"], [0.0, -5.0, -5.0]) == "1"


def test_adaptive_stops_early():
    ans, used = patterns.adaptive(["7", "7", "7", "8", "9"], margin=2)
    assert ans == "7" and used == 2


def test_topologies():
    assert patterns.topology("ring", 4) == [[1, 3], [0, 2], [1, 3], [0, 2]]
    assert patterns.topology("star", 3) == [[1, 2], [0], [0]]
    assert all(len(n) == 4 for n in patterns.topology("full", 5))
    assert all(len(n) == 2 for n in patterns.topology("random", 6, k=2))


def test_debate_and_moa_run_on_mock():
    p = make_problems(1, seed=3)[0]
    agents = [MockLLMBackend(seed=i) for i in range(4)]
    r = patterns.debate(agents, p, rounds=2, topo="ring")
    assert len(r["history"]) == 3 and r["calls"] == 12
    m = patterns.mixture_of_agents(agents[:3], agents[0], p, layers=2)
    assert m["calls"] == 7 and m["answer"] is not None


def test_mock_is_deterministic():
    p = Problem("12+30=", "42")
    msg = [{"role": "user", "content": p.as_question()}]
    assert MockLLMBackend(seed=1).chat(msg, 0.7, 5) == MockLLMBackend(seed=1).chat(msg, 0.7, 5)


def test_condorcet_is_exponential_for_independent_voters():
    assert analysis.condorcet(0.6, 1) == 0.6
    assert analysis.condorcet(0.6, 101) > 0.97
    assert analysis.condorcet(0.4, 101) < 0.03


def test_fit_recognises_saturation_and_log():
    ns = np.array([1, 2, 4, 8, 16, 32])
    sat = 0.9 - 0.4 * ns ** -1.0
    assert analysis.fit_scaling(ns, sat)["best"] == "saturating"
    assert analysis.gain_per_doubling(list(ns), list(sat))[0] > analysis.gain_per_doubling(list(ns), list(sat))[-1]


def test_error_correlation_bounds():
    same = np.array([[1, 0, 1, 0], [1, 0, 1, 0]], bool)
    assert abs(analysis.error_correlation(same) - 1) < 1e-9


def test_tiny_model_trains_and_samples():
    from swarm.tinylm import TinyConfig, train
    m, meta = train(TinyConfig(d_model=16, n_layer=1, n_head=2), seed=0, steps=5, bs=8, n_train=50)
    out = m.sample(["123+456="], temperature=0.0)
    assert isinstance(out[0][0], str) and meta["params"] > 0


def test_error_decay_distinguishes_power_law_from_exponential():
    ns = [2, 4, 8, 16, 32]
    assert analysis.error_decay(ns, [1 - 0.5 * n ** -1.5 for n in ns])["best"] == "power_law"
    assert analysis.error_decay(ns, [1 - 0.5 * np.exp(-0.2 * n) for n in ns])["best"] == "exponential"


def test_verdict_logic():
    import criteria
    ok = {c["id"]: True for c in criteria.CRITERIA}
    assert criteria.verdict(ok, {"C1": False, "C3": False}) == "PASS"
    assert criteria.verdict({**ok, "C4": False}, {}) == "INCONCLUSIVE"
    assert criteria.verdict({**ok, "C2": False}, {}) == "FAIL"
    assert criteria.verdict(ok, {"C1": True}) == "INCONCLUSIVE"


def test_clone_negative_control_has_no_gain():
    import run
    pool = {"samples": {0.7: np.array([[["1", "2"]], [["1", "3"]]], dtype=object)},
            "greedy": np.array([["1", "2"], ["1", "3"]], dtype=object),
            "truth": np.array(["1", "3"], dtype=object)}
    P = {"ns": [1, 2], "bootstrap": 3}
    m = run.metrics_from(run.sweep(pool, 0.7, P, 0), ref_n=2, clone=True)
    assert m["gain_vote8_vs_single"] == 0 and m["gain_hetero8_vs_homo8"] == 0
