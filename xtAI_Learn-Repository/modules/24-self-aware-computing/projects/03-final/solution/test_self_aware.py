"""Test suite P03-final (self-aware system). pytest is missing -> __main__ runner.
    /Users/.../.venv/bin/python test_self_aware.py
"""
import numpy as np
from self_aware import erlang_c, latency_p95, min_replicas, Service, workload, run


def test_model_basics():
    # unstable configurations report infinite latency and certain queueing
    assert np.isinf(latency_p95(100.0, 10.0, 5))
    assert erlang_c(5, 100.0, 10.0) == 1.0
    # more capacity never increases predicted latency (monotone in c)
    lats = [latency_p95(50.0, 10.0, c) for c in range(6, 12)]
    assert all(lats[i] >= lats[i + 1] for i in range(len(lats) - 1))


def test_min_replicas_minimal_and_sufficient():
    mu, slo = 10.0, 0.5
    for lam in [20.0, 45.0, 70.0]:
        c = min_replicas(lam, mu, slo)
        assert latency_p95(lam, mu, c) < slo
        assert latency_p95(lam, mu, c - 1) >= slo


def test_service_sickness_and_restart():
    svc = Service(n_replicas=4, mu_nominal=10.0, seed=0, p_sick=0.0)
    assert svc.capacity(4) == 40.0
    svc.sick[1] = True                                   # a sick replica cuts capacity
    assert 30.0 < svc.capacity(4) < 40.0
    justified = svc.restart(1)
    assert justified is True and svc.sick[1] == np.False_
    assert svc.available(4)[1] == np.False_, "a restarting replica is unavailable"
    assert svc.capacity(4) < 40.0, "and contributes no capacity while restarting"
    # only the first `c` provisioned replicas count toward capacity
    assert svc.capacity(2) < svc.capacity(4)
    # an unjustified restart is reported as such
    assert svc.restart(2) is False


def test_deployment_changes_true_rate_only():
    svc = Service(n_replicas=4, mu_nominal=10.0, seed=0, p_sick=0.0)
    svc.deploy(0.7)
    assert abs(svc.mu_true - 7.0) < 1e-9
    assert abs(svc.capacity(4) - 28.0) < 1e-9


def test_online_estimation_tracks_drift():
    r = run(adaptive_model=True, healing=False, deploy_at=150, T=300, p_sick=0.0)
    # the estimate must converge to the new true rate after the deployment
    assert abs(r["mu_hat"][-1] - r["mu_true"][-1]) / r["mu_true"][-1] < 0.05
    # and the static model must NOT
    r_static = run(adaptive_model=False, healing=False, deploy_at=150, T=300, p_sick=0.0)
    assert abs(r_static["mu_hat"][-1] - r_static["mu_true"][-1]) > 2.0


def test_static_model_fails_after_drift():
    static = run(adaptive_model=False, healing=False, deploy_at=150, T=300, p_sick=0.0)
    adaptive = run(adaptive_model=True, healing=False, deploy_at=150, T=300, p_sick=0.0)
    assert static["slo_after"] > 0.8, "a stale model should violate the SLO almost always"
    assert adaptive["slo_after"] < static["slo_after"] / 3, "re-estimation must recover"
    assert adaptive["pred_error_after"] < static["pred_error_after"] / 3


def test_persistence_defeats_the_base_rate_trap():
    naive = run(healing=True, persistence=1, T=300)
    careful = run(healing=True, persistence=3, T=300)
    # the naive healer is dominated by false positives
    assert naive["precision"] < 0.2, "naive alerting must be mostly false alarms"
    assert careful["precision"] > 3 * naive["precision"], "persistence must raise precision"
    assert careful["restarts"] < naive["restarts"] / 5, "and cut the restart count sharply"
    # the careful healer genuinely improves the system; the naive one makes it WORSE than
    # not healing at all, because every false restart removes a healthy replica
    none = run(healing=False, T=300)
    assert careful["slo_violations"] < 0.8 * none["slo_violations"], "careful healing must help"
    assert naive["slo_violations"] > none["slo_violations"], "naive healing must backfire"


def test_healing_and_model_are_complementary():
    both = run(adaptive_model=True, healing=True, persistence=3, deploy_at=150, T=300)
    only_heal = run(adaptive_model=False, healing=True, persistence=3, deploy_at=150, T=300)
    only_model = run(adaptive_model=True, healing=False, deploy_at=150, T=300)
    assert both["slo_violations"] < only_heal["slo_violations"], "healing cannot fix a stale model"
    assert both["slo_violations"] < only_model["slo_violations"], "a model cannot fix sick replicas"


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
