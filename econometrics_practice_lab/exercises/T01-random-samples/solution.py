"""Musterlösung für T01: Zufallsstichproben und Stichprobenvariation."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RNG_SEED = 20260714


def make_population(size: int = 100_000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    age = rng.integers(18, 81, size=size)
    remote_days = np.clip(rng.poisson(1.4, size=size), 0, 5)
    commute = rng.gamma(shape=3.2, scale=8.0, size=size)
    commute += 0.10 * (age - 40) - 2.2 * remote_days
    commute = np.clip(commute, 0, None)
    return pd.DataFrame(
        {"person_id": np.arange(1, size + 1), "age": age,
         "remote_days": remote_days, "commute_minutes": commute}
    )


def draw_sample(population: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return population.sample(n=n, replace=False, random_state=seed)


def repeated_sample_means(
    population: pd.DataFrame, n: int, repetitions: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = population["commute_minutes"].to_numpy()
    means = np.empty(repetitions)
    for index in range(repetitions):
        selected = rng.choice(len(values), size=n, replace=False)
        means[index] = values[selected].mean()
    return means


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    population = make_population()
    true_mean = population["commute_minutes"].mean()
    population_sd = population["commute_minutes"].std(ddof=1)
    print(f"Population: N={len(population):,}, Mittel={true_mean:.3f}, SD={population_sd:.3f}")

    for seed in (1, 2, 3):
        estimate = draw_sample(population, n=100, seed=seed)["commute_minutes"].mean()
        print(f"Stichprobe seed={seed}: Mittel={estimate:.3f}, Fehler={estimate - true_mean:+.3f}")

    sample_sizes = [50, 200, 1_000]
    repetitions = 2_000
    estimates = {
        n: repeated_sample_means(population, n, repetitions, RNG_SEED + n)
        for n in sample_sizes
    }

    rows = []
    for n, means in estimates.items():
        rows.append(
            {
                "n": n,
                "mean_estimate": means.mean(),
                "bias": means.mean() - true_mean,
                "empirical_se": means.std(ddof=1),
                "theoretical_se": population_sd / np.sqrt(n),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "sampling_summary.csv", index=False)
    print("\nWiederholte Stichproben:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    all_means = np.concatenate(list(estimates.values()))
    x_limits = (all_means.min(), all_means.max())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for axis, (n, means) in zip(axes, estimates.items()):
        axis.hist(means, bins=35, color="#4472C4", alpha=0.85, edgecolor="white")
        axis.axvline(true_mean, color="#C00000", linestyle="--", linewidth=2,
                     label="Populationsmittel")
        axis.set(title=f"n = {n}", xlabel="Stichprobenmittel")
        axis.set_xlim(x_limits)
    axes[0].set_ylabel("Häufigkeit")
    axes[-1].legend(frameon=False)
    fig.suptitle("Stichprobenverteilungen der mittleren Pendelzeit")
    fig.tight_layout()
    fig.savefig(RESULTS / "sample_means.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
