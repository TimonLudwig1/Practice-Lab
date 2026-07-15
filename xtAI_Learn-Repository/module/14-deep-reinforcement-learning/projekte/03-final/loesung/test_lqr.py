"""Tests fuer LQR-Referenz, Umgebung und modellfreien Policy Gradient.

Aufruf:  python test_lqr.py     (kein pytest noetig)
"""
import numpy as np
import torch

from lqr_env import LinearQuadraticSystem
from lqr_reference import solve_dare, lqr_gain, closed_loop_eigenvalues, optimal_cost
from policy_gradient_continuous import (LinearGaussianPolicy, discounted_returns,
                                        train_policy_gradient)


# ------------------------------ Umgebung ------------------------------
def test_dynamik_doppelter_integrator():
    env = LinearQuadraticSystem(dt=0.1)
    env.reset(x0=[0.0, 1.0])            # Position 0, Geschwindigkeit 1
    x, r, done = env.step(0.0)          # keine Kraft -> konstante Geschwindigkeit
    assert np.allclose(x, [0.1, 1.0])   # Position waechst um dt*v
    assert not done


def test_kosten_quadratisch_und_nichtnegativ():
    env = LinearQuadraticSystem(q_pos=1.0, q_vel=0.1, r_ctrl=0.1)
    assert env.cost(np.zeros(2), 0.0) == 0.0                 # Ursprung, keine Kraft: 0
    c = env.cost(np.array([2.0, 0.0]), 0.0)
    assert np.isclose(c, 4.0)                                # 1*2^2
    # Kosten sind quadratisch: doppelter Zustand -> vierfache Kosten
    assert np.isclose(env.cost(np.array([4.0, 0.0]), 0.0), 16.0)
    assert env.cost(np.array([1.0, 1.0]), 3.0) > 0


def test_horizon_beendet_episode():
    env = LinearQuadraticSystem(horizon=5)
    env.reset()
    for i in range(4):
        _, _, done = env.step(0.0)
        assert not done
    _, _, done = env.step(0.0)
    assert done


# ------------------------------ LQR-Referenz ------------------------------
def test_riccati_fixpunkt_erfuellt_dare():
    env = LinearQuadraticSystem()
    P, K, iters = lqr_gain(env)
    A, B, Q, R = env.A, env.B, env.Q, env.R
    residual = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) - P
    assert np.max(np.abs(residual)) < 1e-8, np.max(np.abs(residual))
    assert iters < 10_000


def test_P_symmetrisch_positiv_definit():
    env = LinearQuadraticSystem()
    P, _, _ = lqr_gain(env)
    assert np.allclose(P, P.T)
    assert np.all(np.linalg.eigvals(P) > 0)


def test_closed_loop_ist_stabil():
    env = LinearQuadraticSystem()
    _, K, _ = lqr_gain(env)
    ev = closed_loop_eigenvalues(env, K)
    assert np.all(np.abs(ev) < 1.0), np.abs(ev)      # diskrete Zeit: Betrag < 1


def test_optimale_kosten_stimmen_mit_simulation():
    # J*(x0) = x0^T P x0 muss die simulierten Kosten von u=-Kx reproduzieren
    env = LinearQuadraticSystem()
    P, K, _ = lqr_gain(env)
    for x0 in ([1.0, 0.0], [0.0, 1.0], [-2.0, 0.5]):
        sim, _ = env.rollout_linear(-K.ravel(), x0, horizon=500)
        assert np.isclose(sim, optimal_cost(P, x0), rtol=1e-6), (x0, sim, optimal_cost(P, x0))


def test_lqr_schlaegt_beliebige_andere_lineare_regler():
    # Optimalitaet: keine zufaellige lineare Rueckfuehrung darf besser sein als -K
    env = LinearQuadraticSystem()
    _, K, _ = lqr_gain(env)
    rng = np.random.default_rng(0)
    X0 = [rng.normal(0, 1, 2) for _ in range(50)]
    c_opt = np.mean([env.rollout_linear(-K.ravel(), x0, horizon=200)[0] for x0 in X0])
    for _ in range(20):
        w = rng.normal(0, 3, 2)
        c = np.mean([env.rollout_linear(w, x0, horizon=200)[0] for x0 in X0])
        assert c >= c_opt - 1e-6, f"w={w} schlaegt LQR?! {c} < {c_opt}"


# ------------------------------ Policy Gradient ------------------------------
def test_discounted_returns():
    assert discounted_returns([1.0, 1.0, 1.0], 1.0) == [3.0, 2.0, 1.0]
    assert np.allclose(discounted_returns([0.0, 0.0, 1.0], 0.5), [0.25, 0.5, 1.0])


def test_policy_std_wird_geklemmt():
    p = LinearGaussianPolicy(init_log_std=-10.0)       # absurd klein
    assert float(p.std().detach()) >= np.exp(-1.5) - 1e-6   # untere Klemme greift


def test_select_action_liefert_float_und_logprob():
    p = LinearGaussianPolicy(seed=1)
    u, lp = p.select_action(np.array([1.0, 0.0]))
    assert isinstance(u, float)
    assert lp.requires_grad


def test_policy_gradient_naehert_sich_lqr():
    # Kurzes Training: die gelernte Politik soll deutlich besser als "nichts tun" (w=0)
    # sein und in der Naehe des LQR-Optimums landen.
    env = LinearQuadraticSystem(seed=0)
    _, K, _ = lqr_gain(env)
    policy = LinearGaussianPolicy(lr=0.05, seed=0)
    train_policy_gradient(env, policy, n_updates=150, batch=32)
    w = policy.weights()

    rng = np.random.default_rng(7)
    X0 = [rng.normal(0, 1, 2) for _ in range(50)]
    c_learn = np.mean([env.rollout_linear(w, x0)[0] for x0 in X0])
    c_opt = np.mean([env.rollout_linear(-K.ravel(), x0)[0] for x0 in X0])
    c_zero = np.mean([env.rollout_linear(np.zeros(2), x0)[0] for x0 in X0])

    assert c_learn < c_zero, "gelernte Politik ist nicht besser als Nichtstun"
    assert c_learn < 1.5 * c_opt, f"zu weit vom Optimum: {c_learn:.2f} vs {c_opt:.2f}"
    # Vorzeichen: die Rueckfuehrung muss NEGATIV sein (gegensteuern!)
    assert np.all(w < 0), w


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Starte {len(tests)} Tests ...")
    for name, t in tests:
        t(); print(f"  {name} ... OK")
    print("Alle Tests bestanden.")
