"""Tests for the LQR reference, the environment and the model-free policy gradient.

Call:  python test_lqr.py     (no pytest needed)
"""
import numpy as np
import torch

from lqr_env import LinearQuadraticSystem
from lqr_reference import solve_dare, lqr_gain, closed_loop_eigenvalues, optimal_cost
from policy_gradient_continuous import (LinearGaussianPolicy, discounted_returns,
                                        train_policy_gradient)


# ------------------------------ environment ------------------------------
def test_dynamics_double_integrator():
    env = LinearQuadraticSystem(dt=0.1)
    env.reset(x0=[0.0, 1.0])            # position 0, velocity 1
    x, r, done = env.step(0.0)          # no force -> constant velocity
    assert np.allclose(x, [0.1, 1.0])   # position grows by dt*v
    assert not done


def test_cost_quadratic_and_nonnegative():
    env = LinearQuadraticSystem(q_pos=1.0, q_vel=0.1, r_ctrl=0.1)
    assert env.cost(np.zeros(2), 0.0) == 0.0                 # origin, no force: 0
    c = env.cost(np.array([2.0, 0.0]), 0.0)
    assert np.isclose(c, 4.0)                                # 1*2^2
    # the cost is quadratic: double the state -> quadruple the cost
    assert np.isclose(env.cost(np.array([4.0, 0.0]), 0.0), 16.0)
    assert env.cost(np.array([1.0, 1.0]), 3.0) > 0


def test_horizon_ends_episode():
    env = LinearQuadraticSystem(horizon=5)
    env.reset()
    for i in range(4):
        _, _, done = env.step(0.0)
        assert not done
    _, _, done = env.step(0.0)
    assert done


# ------------------------------ LQR reference ------------------------------
def test_riccati_fixed_point_satisfies_dare():
    env = LinearQuadraticSystem()
    P, K, iters = lqr_gain(env)
    A, B, Q, R = env.A, env.B, env.Q, env.R
    residual = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) - P
    assert np.max(np.abs(residual)) < 1e-8, np.max(np.abs(residual))
    assert iters < 10_000


def test_P_symmetric_positive_definite():
    env = LinearQuadraticSystem()
    P, _, _ = lqr_gain(env)
    assert np.allclose(P, P.T)
    assert np.all(np.linalg.eigvals(P) > 0)


def test_closed_loop_is_stable():
    env = LinearQuadraticSystem()
    _, K, _ = lqr_gain(env)
    ev = closed_loop_eigenvalues(env, K)
    assert np.all(np.abs(ev) < 1.0), np.abs(ev)      # discrete time: magnitude < 1


def test_optimal_cost_matches_simulation():
    # J*(x0) = x0^T P x0 must reproduce the simulated cost of u=-Kx
    env = LinearQuadraticSystem()
    P, K, _ = lqr_gain(env)
    for x0 in ([1.0, 0.0], [0.0, 1.0], [-2.0, 0.5]):
        sim, _ = env.rollout_linear(-K.ravel(), x0, horizon=500)
        assert np.isclose(sim, optimal_cost(P, x0), rtol=1e-6), (x0, sim, optimal_cost(P, x0))


def test_lqr_beats_any_other_linear_controller():
    # optimality: no random linear feedback may be better than -K
    env = LinearQuadraticSystem()
    _, K, _ = lqr_gain(env)
    rng = np.random.default_rng(0)
    X0 = [rng.normal(0, 1, 2) for _ in range(50)]
    c_opt = np.mean([env.rollout_linear(-K.ravel(), x0, horizon=200)[0] for x0 in X0])
    for _ in range(20):
        w = rng.normal(0, 3, 2)
        c = np.mean([env.rollout_linear(w, x0, horizon=200)[0] for x0 in X0])
        assert c >= c_opt - 1e-6, f"w={w} beats LQR?! {c} < {c_opt}"


# ------------------------------ policy gradient ------------------------------
def test_discounted_returns():
    assert discounted_returns([1.0, 1.0, 1.0], 1.0) == [3.0, 2.0, 1.0]
    assert np.allclose(discounted_returns([0.0, 0.0, 1.0], 0.5), [0.25, 0.5, 1.0])


def test_policy_std_is_clamped():
    p = LinearGaussianPolicy(init_log_std=-10.0)       # absurdly small
    assert float(p.std().detach()) >= np.exp(-1.5) - 1e-6   # the lower clamp kicks in


def test_select_action_returns_float_and_logprob():
    p = LinearGaussianPolicy(seed=1)
    u, lp = p.select_action(np.array([1.0, 0.0]))
    assert isinstance(u, float)
    assert lp.requires_grad


def test_policy_gradient_approaches_lqr():
    # A short training: the learned policy should be markedly better than "do nothing" (w=0)
    # and land in the vicinity of the LQR optimum.
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

    assert c_learn < c_zero, "the learned policy is not better than doing nothing"
    assert c_learn < 1.5 * c_opt, f"too far from the optimum: {c_learn:.2f} vs {c_opt:.2f}"
    # sign: the feedback must be NEGATIVE (counter-steer!)
    assert np.all(w < 0), w


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests ...")
    for name, t in tests:
        t(); print(f"  {name} ... OK")
    print("All tests passed.")
