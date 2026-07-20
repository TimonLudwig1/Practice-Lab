"""Starter für T03: Sampling Variation vs. Bias."""

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
    """Ziehe wiederholt mit Zurücklegen; probabilities=None bedeutet gleiche Chancen."""
    # TODO: Normalisiere probabilities, falls sie übergeben wurden.
    # TODO: Ziehe eine Indexmatrix und gib einen Mittelwert pro Zeile zurück.
    raise NotImplementedError


def main() -> None:
    population = make_population()
    values = population["satisfaction"].to_numpy()
    true_mean = values.mean()
    selection_weights = population["response_probability"].to_numpy()
    print(f"Wahrer Populationsmittelwert: {true_mean:.3f}")

    # TODO: Simuliere Random Sample und freiwillige Umfrage für n=100 und n=1000.
    # TODO: Berechne average_estimate, bias, standard_deviation und rmse.
    # TODO: Visualisiere Auswahlmechanismus und alle vier Stichprobenverteilungen.


if __name__ == "__main__":
    main()
