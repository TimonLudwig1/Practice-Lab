"""Test suite P02-medium (auto-scaling). pytest is missing -> __main__ runner.
    /Users/.../.venv/bin/python test_scaling.py
"""
import numpy as np
from scaling import (erlang_c, response_quantile_mmc, min_replicas, workload,
                     policy_reactive, policy_control, policy_predictive, simulate, oracle)


def test_erlang_c_and_quantile_reduce_to_mm1():
    # for c=1 the approximate p95 must be close to the exact M/M/1 value ln(20)/(mu-lam)
    mu = 1.0
    for rho in [0.7, 0.8, 0.9]:
        lam = rho * mu
        approx = response_quantile_mmc(lam, mu, 1, p=0.95)
        exact = np.log(20) / (mu - lam)
        assert abs(approx - exact) / exact < 0.05, f"rho={rho}: {approx} vs {exact}"
    # unstable configuration -> infinite latency
    assert np.isinf(response_quantile_mmc(20.0, 1.0, 5))
    assert erlang_c(5, 20.0, 1.0) == 1.0


def test_min_replicas_is_minimal_and_sufficient():
    mu, slo = 10.0, 0.5
    for lam in [20.0, 45.0, 80.0]:
        c = min_replicas(lam, mu, slo)
        assert response_quantile_mmc(lam, mu, c) < slo, "must meet the SLO"
        assert response_quantile_mmc(lam, mu, c - 1) >= slo, "must be the SMALLEST such c"
    # more load never needs fewer replicas (monotone)
    needs = [min_replicas(l, mu, slo) for l in [10, 30, 50, 70, 90]]
    assert all(needs[i] <= needs[i + 1] for i in range(len(needs) - 1))


def test_reactive_policy_thresholds_and_hysteresis():
    st = dict(c=5, util=0.9)
    assert policy_reactive(st, knob=0.75) == 6, "high utilisation must scale out"
    st = dict(c=5, util=0.1)
    assert policy_reactive(st, knob=0.75) == 4, "low utilisation must scale in"
    # inside the hysteresis band nothing happens
    for u in [0.40, 0.55, 0.70]:
        assert policy_reactive(dict(c=5, util=u), knob=0.75) == 5


def test_predictive_uses_the_model_and_anticipates():
    base = dict(c=3, util=0.5, mu=10.0, slo=0.5, delay=3)
    # rising load: forecast is above the current level -> more capacity than the load alone needs
    rising = dict(base, ewma=50.0, prev_ewma=44.0)
    steady = dict(base, ewma=50.0, prev_ewma=50.0)
    assert policy_predictive(rising, knob=1.0) > policy_predictive(steady, knob=1.0), \
        "a rising trend must be anticipated"
    # with a steady load the answer is exactly the model inversion (times the margin 1.0)
    assert policy_predictive(steady, knob=1.0) == min_replicas(50.0, 10.0, 0.5)


def test_control_policy_moves_toward_target_utilisation():
    # utilisation well above target -> scale out; well below -> scale in
    assert policy_control(dict(c=4, util=0.95, integ=0.0), knob=0.70) > 4
    assert policy_control(dict(c=8, util=0.20, integ=0.0), knob=0.70) < 8


def test_provisioning_delay_is_respected():
    # with a huge cooldown only ONE adaptation can ever be issued
    r = simulate("reactive", cooldown=10**6)
    assert r["adaptations"] <= 1


def test_predictive_is_cost_efficient_vs_oracle():
    # the predictive policy should provision close to the theoretical minimum
    r = simulate("predictive", knob=1.0)
    o = oracle()
    assert r["mean_replicas"] < 1.15 * o["mean_replicas"], "predictive should be near-optimal in cost"
    # and it must beat a naive static over-provisioning of the same violation level in cost
    assert r["mean_replicas"] < simulate("reactive", knob=0.75)["mean_replicas"]


def test_elasticity_metrics_are_consistent():
    r = simulate("reactive")
    assert 0.0 <= r["slo_violations"] <= 1.0
    assert r["mean_replicas"] > 0 and r["adaptations"] >= 0
    # under- and over-provisioning cannot both be large at the same interval by construction
    assert r["under"] >= 0 and r["over"] >= 0
    assert len(r["c"]) == len(r["needed"]) == len(r["lam"])


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
    print(f"\n{passed}/{len(tests)} tests passed.")
