"""Starter für T01: Zufallsstichproben und Stichprobenvariation."""

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
    """Erzeuge eine feste synthetische Stadtpopulation."""
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
    """Ziehe eine einfache Zufallsstichprobe ohne Zurücklegen."""
    # TODO: Verwende population.sample(...).
    raise NotImplementedError


def repeated_sample_means(
    population: pd.DataFrame, n: int, repetitions: int, seed: int
) -> np.ndarray:
    """Gib einen Mittelwert pro wiederholter Zufallsstichprobe zurück."""
    # TODO: Ziehe `repetitions` Stichproben und sammle deren Mittelwerte.
    raise NotImplementedError


def main() -> None:
    population = make_population()
    true_mean = population["commute_minutes"].mean()
    population_sd = population["commute_minutes"].std(ddof=1)
    print(f"Population: N={len(population):,}, Mittel={true_mean:.3f}, SD={population_sd:.3f}")

    # TODO: Ziehe drei Stichproben mit n=100 und den Seeds 1, 2 und 3.

    sample_sizes = [50, 200, 1_000]
    repetitions = 2_000
    estimates = {
        n: repeated_sample_means(population, n, repetitions, RNG_SEED + n)
        for n in sample_sizes
    }

    # TODO: Erzeuge eine Tabelle mit mean_estimate, bias, empirical_se und theoretical_se.
    # TODO: Erzeuge drei Histogramme mit gemeinsamer x-Achse und markiere true_mean.
    # TODO: Speichere Tabelle und Grafik im RESULTS-Ordner.


if __name__ == "__main__":
    main()
