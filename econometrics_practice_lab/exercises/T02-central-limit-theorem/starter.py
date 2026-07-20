"""Starter für T02: Central Limit Theorem."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RNG_SEED = 20260714


def make_population(size: int = 200_000) -> np.ndarray:
    """Stark rechtsschiefe Population synthetischer Bestellwerte."""
    rng = np.random.default_rng(RNG_SEED)
    return rng.lognormal(mean=3.55, sigma=1.0, size=size)


def simulate_means(
    population: np.ndarray, n: int, repetitions: int, seed: int
) -> np.ndarray:
    """Simuliere eine Stichprobenverteilung des Mittelwerts."""
    # TODO: Ziehe eine Matrix mit shape=(repetitions, n) und bilde Zeilenmittelwerte.
    raise NotImplementedError


def main() -> None:
    population = make_population()
    mu = population.mean()
    sigma = population.std(ddof=0)
    print(f"Population: Mittel={mu:.3f}, Median={np.median(population):.3f}, "
          f"Schiefe={stats.skew(population):.3f}")

    sample_sizes = [1, 5, 30, 100]
    distributions = {
        n: simulate_means(population, n, 5_000, RNG_SEED + n)
        for n in sample_sizes
    }

    # TODO: Erstelle die Ergebnistabelle einschließlich empirical_sd, theoretical_se,
    #       skewness und coverage_95.
    # TODO: Visualisiere Population und Stichprobenverteilungen.
    # TODO: Lege über jedes Mittelwert-Histogramm eine Normaldichte.


if __name__ == "__main__":
    main()
