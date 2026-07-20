"""Musterlösung für T03: Sampling Variation vs. Bias."""

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
    engagement = rng.normal(size=size)
    satisfaction = np.clip(70 + 8 * engagement + rng.normal(0, 8, size=size), 0, 100)
    response_probability = 1 / (1 + np.exp(-(-0.8 + 0.075 * (satisfaction - 70))))
    return pd.DataFrame(
        {"customer_id": np.arange(size), "satisfaction": satisfaction,
         "response_probability": response_probability}
    )


def simulate_means(
    values: np.ndarray,
    sample_size: int,
    repetitions: int,
    seed: int,
    probabilities: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = None if probabilities is None else probabilities / probabilities.sum()
    indices = rng.choice(
        len(values), size=(repetitions, sample_size), replace=True, p=probabilities
    )
    return values[indices].mean(axis=1)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    population = make_population()
    values = population["satisfaction"].to_numpy()
    true_mean = values.mean()
    selection_weights = population["response_probability"].to_numpy()
    print(f"Wahrer Populationsmittelwert: {true_mean:.3f}")

    figure, axis = plt.subplots(figsize=(7, 4))
    ordered = population.sort_values("satisfaction")
    axis.scatter(ordered["satisfaction"][::100], ordered["response_probability"][::100],
                 s=12, alpha=0.5, color="#ED7D31")
    axis.set(title="Selektionsmechanismus der freiwilligen Umfrage",
             xlabel="Zufriedenheit", ylabel="Teilnahmewahrscheinlichkeit")
    figure.tight_layout()
    figure.savefig(RESULTS / "selection_mechanism.png", dpi=160)
    plt.close(figure)

    results: dict[tuple[str, int], np.ndarray] = {}
    for n in (100, 1_000):
        results[("Random Sample", n)] = simulate_means(
            values, n, 2_000, RNG_SEED + n
        )
        results[("Freiwillige Umfrage", n)] = simulate_means(
            values, n, 2_000, RNG_SEED + 10_000 + n, selection_weights
        )

    rows = []
    for (design, n), estimates in results.items():
        errors = estimates - true_mean
        rows.append(
            {"design": design, "n": n, "average_estimate": estimates.mean(),
             "bias": errors.mean(), "standard_deviation": estimates.std(ddof=1),
             "rmse": np.sqrt(np.mean(errors ** 2))}
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "bias_variation_summary.csv", index=False)
    print("\nBias und Sampling Variation:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    colors = {"Random Sample": "#4472C4", "Freiwillige Umfrage": "#ED7D31"}
    for axis, ((design, n), estimates) in zip(axes.flat, results.items()):
        axis.hist(estimates, bins=35, color=colors[design], edgecolor="white", alpha=0.85)
        axis.axvline(true_mean, color="#C00000", linestyle="--", linewidth=2,
                     label="Wahrer Mittelwert")
        axis.axvline(estimates.mean(), color="black", linestyle=":", linewidth=2,
                     label="Mittlerer Schätzwert")
        axis.set(title=f"{design}, n={n}", xlabel="Geschätzter Mittelwert",
                 ylabel="Häufigkeit")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Sampling Variation (Breite) und Bias (Verschiebung)")
    figure.tight_layout()
    figure.savefig(RESULTS / "bias_vs_variation.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
