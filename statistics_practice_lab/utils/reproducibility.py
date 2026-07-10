"""Reproducibility helpers.

Every random operation in this repo must be reproducible (see CLAUDE.md).
We use NumPy's modern ``default_rng`` API exclusively — never the legacy global
``np.random.*`` functions, whose shared global state makes results depend on
call order elsewhere in the program.
"""

from __future__ import annotations

import numpy as np

# Default seed used across the repo unless a project has a reason to differ.
# 42 is a convention, nothing magic — the point is that it is *fixed*.
DEFAULT_SEED: int = 42


def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return a fresh, seeded NumPy random Generator.

    Parameters
    ----------
    seed:
        Integer seed. The same seed always yields the same stream of numbers,
        which is what makes a simulation reproducible.

    Returns
    -------
    np.random.Generator
        A ``Generator`` instance (the modern replacement for the legacy
        ``RandomState``). Pass it explicitly into every function that draws
        random numbers, instead of relying on global state.

    Examples
    --------
    >>> rng = make_rng()
    >>> rng.normal(size=3)  # doctest: +SKIP
    array([...])
    """
    return np.random.default_rng(seed)
