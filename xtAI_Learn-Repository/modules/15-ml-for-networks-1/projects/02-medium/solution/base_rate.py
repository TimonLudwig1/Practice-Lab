"""The base-rate fallacy, quantitatively (Axelsson 2000) — script section 3.1.

The core question of an analyst is NOT "how often does the detector recognize an attack?" (TPR),
but: **"it raised an alarm — is there really something there?"** That is the positive predictive
value (PPV / precision), and it depends crucially on the base rate:

    P(I|A) = P(A|I)*pi / ( P(A|I)*pi + P(A|~I)*(1-pi) )
           =   TPR*pi  / (   TPR*pi   +    FPR*(1-pi)  )

with  I = intrusion (attack),  A = alarm,  pi = P(I) = base rate.
"""
from __future__ import annotations
import numpy as np


def ppv_at_base_rate(tpr, fpr, pi):
    """Positive predictive value P(attack | alarm) via Bayes.

    Accepts scalars or arrays (for pi). Returns: the same shape as the input.
    """
    tpr = np.asarray(tpr, dtype=float)
    fpr = np.asarray(fpr, dtype=float)
    pi = np.asarray(pi, dtype=float)
    numerator = tpr * pi
    denominator = tpr * pi + fpr * (1.0 - pi)
    # Where no alarm is possible at all (denominator 0), the PPV is undefined -> 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denominator > 0, numerator / denominator, 0.0)
    return out if out.ndim else float(out)


def required_fpr(tpr, pi, target_ppv):
    """Which FPR is needed to reach a target PPV? (rearrangement of the Bayes formula)

        ppv = TPR*pi / (TPR*pi + FPR*(1-pi))
    =>  FPR = TPR*pi*(1-ppv) / (ppv*(1-pi))
    """
    tpr = float(tpr); pi = float(pi); target_ppv = float(target_ppv)
    if not (0.0 < target_ppv < 1.0):
        raise ValueError("target_ppv must lie in (0,1)")
    return tpr * pi * (1.0 - target_ppv) / (target_ppv * (1.0 - pi))


def alarms_per_day(n_flows_per_day, tpr, fpr, pi):
    """Absolute alarm counts per day. Returns: (true_alarms, false_alarms).

    The most important reality check there is: percentages obscure, absolute numbers do not.
    """
    n = float(n_flows_per_day)
    true_alarms = tpr * pi * n
    false_alarms = fpr * (1.0 - pi) * n
    return true_alarms, false_alarms


def expected_cost(n_flows_per_day, tpr, fpr, pi,
                  cost_false_alarm, cost_missed):
    """Expected daily cost of an operating point.

    False alarm    -> analyst time (cost_false_alarm each)
    Missed attack  -> damage      (cost_missed each)
    """
    n = float(n_flows_per_day)
    false_alarms = fpr * (1.0 - pi) * n
    missed = (1.0 - tpr) * pi * n
    return false_alarms * cost_false_alarm + missed * cost_missed


def best_operating_point(fpr_curve, tpr_curve, thresholds, n_flows_per_day, pi,
                         cost_false_alarm, cost_missed):
    """Picks the cost-minimal operating point from a ROC curve.

    Returns: (index, threshold, cost) of the minimum.
    """
    cost = np.array([
        expected_cost(n_flows_per_day, t, f, pi, cost_false_alarm, cost_missed)
        for f, t in zip(fpr_curve, tpr_curve)
    ])
    i = int(np.argmin(cost))
    return i, float(thresholds[i]), float(cost[i])
