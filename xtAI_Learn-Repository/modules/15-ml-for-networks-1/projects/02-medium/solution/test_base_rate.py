"""Tests for the base-rate computation.  Call:  python test_base_rate.py"""
import numpy as np

from base_rate import (ppv_at_base_rate, required_fpr, alarms_per_day,
                       expected_cost, best_operating_point)


# ----------------------- PPV / Bayes -----------------------
def test_axelsson_example():
    # The numerical example from the script (3.1): TPR .99, FPR .001, pi 1e-4 -> ~9 %
    p = ppv_at_base_rate(0.99, 0.001, 1e-4)
    assert abs(p - 0.0901) < 0.001, p


def test_ppv_perfect_detector_without_false_alarms():
    # FPR = 0 -> every alarm is real, no matter how small the base rate
    assert ppv_at_base_rate(0.9, 0.0, 1e-9) == 1.0


def test_ppv_falls_monotonically_with_smaller_base_rate():
    pis = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    values = [ppv_at_base_rate(0.99, 0.01, p) for p in pis]
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1)), values


def test_ppv_at_base_rate_one():
    # pi = 1: everything is an attack -> every alarm is necessarily real
    assert abs(ppv_at_base_rate(0.5, 0.5, 1.0) - 1.0) < 1e-12


def test_ppv_accepts_arrays():
    out = ppv_at_base_rate(0.99, 0.001, np.array([1e-4, 1e-2]))
    assert out.shape == (2,) and out[1] > out[0]


def test_ppv_useless_detector():
    # TPR == FPR (coin flip) -> PPV == base rate (no information gain)
    for pi in (0.01, 0.2, 0.5):
        assert abs(ppv_at_base_rate(0.5, 0.5, pi) - pi) < 1e-12


# ----------------------- inversion -----------------------
def test_required_fpr_is_inverse_of_ppv():
    tpr, pi, target = 0.99, 1e-4, 0.5
    f = required_fpr(tpr, pi, target)
    assert abs(ppv_at_base_rate(tpr, f, pi) - target) < 1e-9


def test_required_fpr_stricter_at_smaller_base_rate():
    f_large = required_fpr(0.99, 1e-2, 0.5)
    f_small = required_fpr(0.99, 1e-4, 0.5)
    assert f_small < f_large


def test_required_fpr_rejects_nonsense():
    for target in (0.0, 1.0, 1.5, -0.2):
        try:
            required_fpr(0.99, 1e-4, target)
            assert False, f"target_ppv={target} should have been rejected"
        except ValueError:
            pass


# ----------------------- absolute numbers -----------------------
def test_alarms_per_day():
    true_alarms, false_alarms = alarms_per_day(1e7, tpr=0.99, fpr=0.001, pi=1e-4)
    assert abs(true_alarms - 990.0) < 1.0          # 0.99 * 1e-4 * 1e7
    assert abs(false_alarms - 9999.0) < 1.0        # 0.001 * 0.9999 * 1e7
    assert false_alarms > 10 * true_alarms         # the heart of the problem


def test_alarms_consistent_with_ppv():
    # The PPV must also follow from the absolute numbers
    tpr, fpr, pi = 0.9, 0.02, 1e-3
    true_alarms, false_alarms = alarms_per_day(1e6, tpr, fpr, pi)
    assert abs(true_alarms / (true_alarms + false_alarms)
               - ppv_at_base_rate(tpr, fpr, pi)) < 1e-9


# ----------------------- costs -----------------------
def test_expected_cost_sums_both_error_types():
    # 1e6 flows, pi=1e-3 -> 1000 attacks; TPR=0.9 -> 100 missed
    # FPR=0.01 -> ~9990 false alarms
    c = expected_cost(1e6, tpr=0.9, fpr=0.01, pi=1e-3,
                      cost_false_alarm=1.0, cost_missed=100.0)
    expected = 0.01 * 0.999 * 1e6 * 1.0 + 0.1 * 1e-3 * 1e6 * 100.0
    assert abs(c - expected) < 1e-6


def test_best_operating_point_picks_the_minimum():
    # Artificial ROC with three points; 1e6 flows, pi=1e-3 -> 1000 attacks.
    #   point 0 (never alarm)     : 1000 missed * 100              = 100_000
    #   point 1 (compromise)      : ~999 false*1 + 100 missed*100  =  10_999  <- minimum
    #   point 2 (permanent alarm) : ~499_500 false alarms * 1      = 499_500
    fpr = np.array([0.0, 0.001, 0.5])
    tpr = np.array([0.0, 0.90, 1.0])
    thr = np.array([1.0, 0.5, 0.0])
    i, s, c = best_operating_point(fpr, tpr, thr, 1e6, 1e-3,
                                   cost_false_alarm=1.0, cost_missed=100.0)
    assert i == 1 and s == 0.5, (i, s, c)
    assert c < expected_cost(1e6, 0.0, 0.0, 1e-3, 1.0, 100.0)   # better than never alarming
    assert c < expected_cost(1e6, 1.0, 0.5, 1e-3, 1.0, 100.0)   # better than alarming constantly


def test_cost_optimum_shifts_with_the_costs():
    # If a missed attack becomes very expensive, the more aggressive point pays off.
    fpr = np.array([0.0, 0.001, 0.5])
    tpr = np.array([0.0, 0.90, 1.0])
    thr = np.array([1.0, 0.5, 0.0])
    i_cheap, _, _ = best_operating_point(fpr, tpr, thr, 1e6, 1e-3, 1.0, 100.0)
    i_expensive, _, _ = best_operating_point(fpr, tpr, thr, 1e6, 1e-3, 1.0, 10_000.0)
    assert i_cheap == 1 and i_expensive == 2, (i_cheap, i_expensive)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests ...")
    for name, t in tests:
        t(); print(f"  {name} ... OK")
    print("All tests passed.")
