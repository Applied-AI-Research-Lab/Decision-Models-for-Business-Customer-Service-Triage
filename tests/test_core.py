import math

import numpy as np
import pytest

from bp01 import metrics as M
from bp01 import stats as S
from bp01.prompts import verbalized_probs
from bp01.readout import score_options
from bp01.runners.api import parse_response

CIDS = {c: i for i, c in enumerate("0123456789]")}


def make_backend(target: dict[str, float]):
    inv = {v: k for k, v in CIDS.items()}

    def fn(batch):
        out = []
        for ids in batch:
            s = "".join(inv[i] for i in ids[1:])

            mass = {}
            for lab, p in target.items():
                full = lab + "]"
                if full.startswith(s) and len(full) > len(s):
                    ch = full[len(s)]
                    mass[ch] = mass.get(ch, 0.0) + p
            tot = sum(mass.values())
            out.append({CIDS[ch]: math.log(m / tot) for ch, m in mass.items()})
        return out

    return fn


@pytest.mark.parametrize("k", [3, 12, 77, 151])
def test_trie_recovers_distribution(k):
    rng = np.random.default_rng(k)
    w = rng.dirichlet(np.ones(k))
    target = {str(i + 1): float(w[i]) for i in range(k)}
    probs, diags = score_options([[999]], [k], make_backend(target), CIDS, prune=0.0)
    np.testing.assert_allclose(probs[0], w, atol=1e-9)
    assert abs(diags[0]["mass"] - 1.0) < 1e-9


def test_trie_pruning_keeps_mass():
    target = {"1": 0.9, "2": 0.0999999, "12": 1e-7}
    probs, _ = score_options([[999]], [12], make_backend(target), CIDS, prune=1e-5)
    assert probs[0][0] == pytest.approx(0.9 / 0.9999999, rel=1e-6)


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1] * 50)
    P = np.eye(2)[y]
    assert M.ece_equal_mass(P, y) == pytest.approx(0.0)
    Pw = np.tile([0.99, 0.01], (100, 1))
    assert M.ece_equal_mass(Pw, y) == pytest.approx(0.49, abs=1e-9)


def test_brier_nll_accuracy():
    y = np.array([0, 1])
    P = np.array([[0.8, 0.2], [0.4, 0.6]])
    assert M.accuracy(P, y) == 1.0
    assert M.brier(P, y) == pytest.approx(((0.2**2 + 0.2**2) + (0.4**2 + 0.4**2)) / 2)
    assert M.nll(P, y) == pytest.approx(-(math.log(0.8) + math.log(0.6)) / 2)


def test_aurc_perfect_ranking():
    y = np.array([0, 0, 0, 1])
    P = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    cov, risk = M.risk_coverage(P, y)
    assert risk[-1] == 0.25 and risk[0] == 0.0
    assert M.coverage_at_risk(P, y, 0.0) == 0.75


def test_holm_and_mcnemar():
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    a = np.array([1] * 20 + [0] * 0, bool); b = np.array([0] * 20, bool)
    assert S.mcnemar_exact(a, b) < 1e-4
    assert S.mcnemar_exact(a, a) == 1.0


def test_tost():
    assert S.tost_equivalent(-0.01, 0.015, 0.02)
    assert not S.tost_equivalent(-0.03, 0.01, 0.02)


def test_jev_parse_and_verbalized():
    rec = {"options": ["a", "b", "c"]}
    data = {"model": "jev-1.13.0", "answers": {"intent": {"type": "choice", "choice": "b",
            "probabilities": {"a": 0.1, "b": 0.7, "c": 0.2}, "confidence": 0.6}}, "usage": {"input_tokens": 50}}
    probs, meta = parse_response(rec, data)
    assert max(probs, key=probs.get) == "b" and meta["model_returned"] == "jev-1.13.0"
    v = verbalized_probs("c", 70, rec["options"])
    assert v["c"] == pytest.approx(0.7) and sum(v.values()) == pytest.approx(1.0)
    u = verbalized_probs("zzz", 70, rec["options"])
    assert u["a"] == pytest.approx(1 / 3)
