"""Generate reproducible random arrays for sorting property tests."""

from __future__ import annotations

import random


# A fixed seed turns a broad random sample into a reproducible regression suite.
DEFAULT_SEED = 70701


def generate_random_cases(
    case_count: int = 200,
    *,
    max_size: int = 80,
    minimum: int = -100,
    maximum: int = 100,
    seed: int = DEFAULT_SEED,
) -> list[list[int]]:
    """Return random integer arrays including empty and duplicate-heavy cases."""
    if isinstance(case_count, bool) or not isinstance(case_count, int):
        raise TypeError("case_count must be an integer")
    if case_count <= 0:
        raise ValueError("case_count must be greater than zero")
    if max_size < 0:
        raise ValueError("max_size must not be negative")
    if minimum > maximum:
        raise ValueError("minimum must not exceed maximum")

    rng = random.Random(seed)
    return [
        [rng.randint(minimum, maximum) for _ in range(rng.randint(0, max_size))]
        for _ in range(case_count)
    ]


def structured_cases() -> list[list[int]]:
    """Return deterministic shapes that random generation may miss."""
    return [
        [],
        [1],
        list(range(30)),
        list(range(30, 0, -1)),
        [7] * 30,
        [0, -1, 1, -1, 0, 1],
        [2, 1] * 20,
    ]
