"""The base-rate fallacy, quantitatively (Axelsson 2000) — script section 3.1.

>>> YOUR TASK <<<  Fill in the five functions marked with TODO. They are all short —
the difficulty is not the code, it is REALLY understanding the formulas.

The core question of an analyst is NOT "how often does the detector recognize an attack?" (TPR),
but: **"it raised an alarm — is there really something there?"** That is the positive predictive
value (PPV / precision):

    P(I|A) = P(A|I)*pi / ( P(A|I)*pi + P(A|~I)*(1-pi) )
           =   TPR*pi  / (   TPR*pi   +    FPR*(1-pi)  )

with  I = intrusion (attack),  A = alarm,  pi = P(I) = base rate.

Sanity check (which your code has to reproduce):
    TPR=0.99, FPR=0.001, pi=1e-4  ->  PPV ~ 9 %   (so ~91 % false alarms!)

Reference solution: solution/base_rate.py — try it yourself first!
"""
from __future__ import annotations
import numpy as np


def ppv_at_base_rate(tpr, fpr, pi):
    """Positive predictive value P(attack | alarm) via Bayes.

    Must handle scalars AND arrays for `pi` (for the curve in run.py).
    Edge case: if no alarm is possible at all (denominator 0), return 0.0.

    Hint: np.asarray(..., dtype=float) at the start, then np.where(denominator > 0, ...).
    """
    # TODO
    raise NotImplementedError


def required_fpr(tpr, pi, target_ppv):
    """Which FPR is needed to reach a target PPV?

    Rearrange the Bayes formula for FPR (pen and paper!). Raise a ValueError if
    target_ppv does not lie strictly between 0 and 1.

    Self-check: ppv_at_base_rate(tpr, required_fpr(tpr, pi, z), pi) == z
    """
    # TODO
    raise NotImplementedError


def alarms_per_day(n_flows_per_day, tpr, fpr, pi):
    """Absolute alarm counts per day. Returns: (true_alarms, false_alarms).

    The most important reality check there is: percentages obscure, absolute numbers do not.
    Of n flows, pi*n are attacks (of which TPR are detected) and (1-pi)*n are harmless
    (of which FPR trigger a false alarm).
    """
    # TODO
    raise NotImplementedError


def expected_cost(n_flows_per_day, tpr, fpr, pi,
                  cost_false_alarm, cost_missed):
    """Expected daily cost of an operating point.

    Two kinds of error cost money:
      false alarm    -> analyst time  (count: FPR*(1-pi)*n)
      missed attack  -> damage        (count: (1-TPR)*pi*n)
    """
    # TODO
    raise NotImplementedError


def best_operating_point(fpr_curve, tpr_curve, thresholds, n_flows_per_day, pi,
                         cost_false_alarm, cost_missed):
    """Picks the COST-MINIMAL operating point from a ROC curve.

    Compute the expected cost for every point (fpr, tpr) of the curve and take the
    minimum. Returns: (index, threshold, cost) as (int, float, float).

    This is the bridge to module 04 (cost-based threshold in the Adult project): the right
    threshold follows from the COSTS, not from 0.5.
    """
    # TODO
    raise NotImplementedError
