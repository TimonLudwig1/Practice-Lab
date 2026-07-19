"""Generate reproducible sorting workloads with distinct structural shapes."""

from __future__ import annotations

import random


# A fixed seed ensures every algorithm sees identical values and perturbations.
DEFAULT_SEED = 70702
INPUT_TYPES = ("random", "nearly_sorted", "reversed", "many_duplicates")


def generate_workloads(size: int, *, seed: int = DEFAULT_SEED) -> dict[str, list[int]]:
    """Return four arrays of equal size with different order characteristics."""
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size <= 0:
        raise ValueError("size must be greater than zero")

    rng = random.Random(seed + size)
    random_values = [rng.randint(-10 * size, 10 * size) for _ in range(size)]

    nearly_sorted = list(range(size))
    swap_count = max(1, size // 20)
    for _ in range(swap_count):
        left = rng.randrange(size)
        right = rng.randrange(size)
        nearly_sorted[left], nearly_sorted[right] = (
            nearly_sorted[right],
            nearly_sorted[left],
        )

    return {
        "random": random_values,
        "nearly_sorted": nearly_sorted,
        "reversed": list(range(size, 0, -1)),
        "many_duplicates": [rng.randrange(8) for _ in range(size)],
    }
