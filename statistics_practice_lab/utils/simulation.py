"""Generic simulation harnesses shared across projects.

Design principle: these helpers are *harnesses*, not statistics. They take the
statistical logic (a confidence-interval function, an estimator) as an argument.
The statistical core itself is always implemented by hand inside each project —
this module only handles the repetitive "repeat 10,000 times and count" plumbing.

This is where the seed of Capstone C3 (`statlab`) lives.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def coverage_simulation(
    sample_generator: Callable[[np.random.Generator], np.ndarray],
    ci_function: Callable[[np.ndarray], tuple[float, float]],
    true_value: float,
    n_simulations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Measure the empirical coverage of a confidence-interval procedure.

    A 95%% confidence interval *claims* that, across many repeated samples, the
    interval contains the true parameter about 95%% of the time. This function
    checks whether a given CI procedure actually delivers on that promise: it
    draws many samples, builds a CI for each, and counts how often the CI
    brackets the known ``true_value``.

    Parameters
    ----------
    sample_generator:
        Callable that takes an rng and returns one fresh sample (1-D array).
    ci_function:
        Callable that takes a sample and returns ``(lower, upper)``.
    true_value:
        The known population parameter (we can only measure coverage when we
        know the truth — hence simulation).
    n_simulations:
        Number of repeated samples. Default 10,000 per repo convention.
    rng:
        Optional Generator; a default-seeded one is created if omitted.

    Returns
    -------
    dict
        ``coverage`` (fraction of CIs containing the truth), ``n_simulations``,
        ``mean_width`` (average interval width — the price of that coverage),
        and ``mc_se`` (Monte-Carlo standard error of the coverage estimate,
        so we know how precise the coverage number itself is).
    """
    if rng is None:
        from .reproducibility import make_rng

        rng = make_rng()

    contained = 0
    total_width = 0.0
    for _ in range(n_simulations):
        sample = sample_generator(rng)
        lower, upper = ci_function(sample)
        total_width += upper - lower
        if lower <= true_value <= upper:
            contained += 1

    coverage = contained / n_simulations
    # Coverage is itself an estimate from a finite simulation; its uncertainty
    # is the SE of a proportion: sqrt(p(1-p)/N).
    mc_se = float(np.sqrt(coverage * (1.0 - coverage) / n_simulations))
    return {
        "coverage": coverage,
        "n_simulations": float(n_simulations),
        "mean_width": total_width / n_simulations,
        "mc_se": mc_se,
    }
