"""Testsuite fuer P03-final (Put-that-there). pytest fehlt -> __main__-Runner.
    /Users/.../.venv/bin/python test_putthatthere.py
"""
import numpy as np
from putthatthere import (make_scene, make_command, resolve, resolve_naive_at_word,
                         _object_evidence, TYPES)

MU, SIG, TAU = -0.15, 0.035, 0.18


def _pool(n=400, base=500):
    cmds = []
    for k in range(n):
        sc = make_scene(7, seed=base + k)
        cmds.append(make_command(sc, seed=base + 50000 + k,
                                 mu_true=MU, sigma_point=SIG, tau_true=TAU))
    return cmds


def _acc(cmds, **kw):
    return np.mean([resolve(c, mu_hat=MU, sigma_hat=SIG, tau_hat=TAU, **kw)[1] == c["target"]
                    for c in cmds])


def test_scene_and_command_structure():
    sc = make_scene(7, seed=1)
    assert sc["pos"].shape == (7, 2)
    cmd = make_command(sc, seed=2, mu_true=MU, sigma_point=SIG, tau_true=TAU)
    n = cmd["scene_pos"].shape[0]
    assert n in (7, 8)                       # 8, falls Zwilling hinzugefuegt
    assert 0 <= cmd["target"] < n
    assert cmd["decoy"] != cmd["target"]
    assert cmd["ptr_xy"].shape[0] == cmd["ptr_t"].shape[0]
    assert np.isclose(cmd["q_type"].sum(), 1.0)


def test_posterior_valid():
    sc = make_scene(7, seed=3)
    cmd = make_command(sc, seed=4, mu_true=MU, sigma_point=SIG, tau_true=TAU)
    p, pred = resolve(cmd, mu_hat=MU, sigma_hat=SIG, tau_hat=TAU)
    assert np.isclose(p.sum(), 1.0) and np.all(p >= 0)
    assert p.argmax() == pred


def test_target_timing_matches_mu():
    # Der Zeiger erreicht das Ziel um t_word + mu_true -> delta_target ~ mu_true
    deltas = []
    for c in _pool(200):
        _, t_near = _object_evidence(c, c["target"])
        deltas.append(t_near - c["t_word"])
    med = np.median(deltas)
    assert abs(med - MU) < 0.08, f"Median delta_target {med:.3f} weit von mu_true {MU}"


def test_full_fusion_beats_all_ablations():
    cmds = _pool(500)
    full = _acc(cmds)
    point = _acc(cmds, use_sem=False, use_point=True, use_temp=False)
    sem = _acc(cmds, use_sem=True, use_point=False, use_temp=False)
    no_time = _acc(cmds, use_sem=True, use_point=True, use_temp=False)
    no_sem = _acc(cmds, use_sem=False, use_point=True, use_temp=True)
    assert full > point + 0.15
    assert full > sem + 0.15
    assert full >= no_time and full >= no_sem
    assert full > 0.80


def test_temporal_kills_decoy():
    # Zeigen allein ist mehrdeutig (Decoy nah); der Zeitfaktor hebt es klar an
    cmds = _pool(500)
    point = _acc(cmds, use_sem=False, use_point=True, use_temp=False)
    no_sem = _acc(cmds, use_sem=False, use_point=True, use_temp=True)
    assert no_sem > point + 0.20, f"Zeit sollte Zeigen deutlich verbessern ({point:.3f}->{no_sem:.3f})"


def test_mu_zero_worse_than_true():
    cmds = _pool(500)
    acc_true = np.mean([resolve(c, mu_hat=MU, sigma_hat=SIG, tau_hat=TAU)[1] == c["target"]
                        for c in cmds])
    acc_zero = np.mean([resolve(c, mu_hat=0.0, sigma_hat=SIG, tau_hat=TAU)[1] == c["target"]
                        for c in cmds])
    assert acc_true > acc_zero + 0.03, f"mu=wahr {acc_true:.3f} sollte mu=0 {acc_zero:.3f} schlagen"


def test_naive_baseline_weaker():
    cmds = _pool(500)
    full = _acc(cmds)
    naive = np.mean([resolve_naive_at_word(c) == c["target"] for c in cmds])
    assert full > naive + 0.20, f"Volle Fusion {full:.3f} sollte naiv {naive:.3f} klar schlagen"


def test_mutual_disambiguation_positive():
    cmds = _pool(500)
    saved = 0; wrong = 0
    for c in cmds:
        p_only = resolve(c, use_sem=False, use_point=True, use_temp=False,
                         mu_hat=MU, sigma_hat=SIG, tau_hat=TAU)[1]
        p_full = resolve(c, mu_hat=MU, sigma_hat=SIG, tau_hat=TAU)[1]
        if p_only != c["target"]:
            wrong += 1
            saved += (p_full == c["target"])
    assert wrong > 0 and saved / wrong > 0.5


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} Tests bestanden.")
