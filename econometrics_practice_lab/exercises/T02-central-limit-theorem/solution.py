"""Musterlösung für T02: Central Limit Theorem."""

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
    rng = np.random.default_rng(RNG_SEED)
    return rng.lognormal(mean=3.55, sigma=1.0, size=size)


def simulate_means(
    population: np.ndarray, n: int, repetitions: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = rng.choice(population, size=(repetitions, n), replace=True)
    return samples.mean(axis=1)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    population = make_population()
    mu = population.mean()
    sigma = population.std(ddof=0)
    print(f"Population: Mittel={mu:.3f}, Median={np.median(population):.3f}, "
          f"SD={sigma:.3f}, Schiefe={stats.skew(population):.3f}")

    figure, axis = plt.subplots(figsize=(7, 4))
    upper = np.quantile(population, 0.995)
    axis.hist(population[population <= upper], bins=80, color="#70AD47", edgecolor="white")
    axis.axvline(mu, color="#C00000", linestyle="--", label="Mittelwert")
    axis.axvline(np.median(population), color="#4472C4", linestyle=":", label="Median")
    axis.set(title="Rechtsschiefe Population der Bestellwerte (bis 99,5%-Quantil)",
             xlabel="Bestellwert", ylabel="Häufigkeit")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(RESULTS / "population_distribution.png", dpi=160)
    plt.close(figure)

    sample_sizes = [1, 5, 30, 100]
    distributions = {
        n: simulate_means(population, n, 5_000, RNG_SEED + n)
        for n in sample_sizes
    }

    rows = []
    for n, means in distributions.items():
        theoretical_se = sigma / np.sqrt(n)
        coverage = np.mean(np.abs(means - mu) <= 1.96 * theoretical_se)
        rows.append(
            {"n": n, "mean_of_means": means.mean(),
             "empirical_sd": means.std(ddof=1), "theoretical_se": theoretical_se,
             "skewness": stats.skew(means), "coverage_95": coverage}
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "clt_summary.csv", index=False)
    print("\nStichprobenverteilungen:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (n, means) in zip(axes.flat, distributions.items()):
        axis.hist(means, bins=45, density=True, color="#5B9BD5", alpha=0.75,
                  edgecolor="white")
        x_values = np.linspace(means.min(), means.max(), 300)
        theoretical_se = sigma / np.sqrt(n)
        axis.plot(x_values, stats.norm.pdf(x_values, loc=mu, scale=theoretical_se),
                  color="#C00000", linewidth=2, label="Normalapproximation")
        axis.axvline(mu, color="black", linestyle="--", linewidth=1)
        axis.set(title=f"n = {n}", xlabel="Stichprobenmittel", ylabel="Dichte")
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Central Limit Theorem bei schiefen Bestellwerten")
    figure.tight_layout()
    figure.savefig(RESULTS / "clt_distributions.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
