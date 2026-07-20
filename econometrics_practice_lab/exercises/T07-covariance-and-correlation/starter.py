"""Starter für T07: Kovarianz, Korrelation und Maßeinheiten."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 300) -> pd.DataFrame:
    """Erzeuge reproduzierbare Haushaltsdaten."""
    rng = np.random.default_rng(RNG_SEED)
    income = np.clip(rng.normal(loc=4_200, scale=1_300, size=size), 1_200, 9_000)
    leisure = np.clip(90 + 0.11 * income + rng.normal(0, 190, size=size), 0, None)
    commute = np.clip(56 - 0.0045 * income + rng.normal(0, 11, size=size), 5, 100)
    return pd.DataFrame(
        {
            "household_id": np.arange(1, size + 1),
            "income_eur": income,
            "leisure_spending_eur": leisure,
            "commute_minutes": commute,
        }
    )


def sample_covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Berechne die Stichprobenkovarianz ohne np.cov oder pandas.cov."""
    # TODO: Zentriere x und y, multipliziere paarweise und teile durch n - 1.
    raise NotImplementedError


def sample_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Standardisiere die Kovarianz mit den beiden Stichproben-SDs."""
    # TODO: Verwende sample_covariance(...) und Standardabweichungen mit ddof=1.
    raise NotImplementedError


def main() -> None:
    data = make_data()
    income = data["income_eur"].to_numpy()
    leisure = data["leisure_spending_eur"].to_numpy()
    commute = data["commute_minutes"].to_numpy()

    # TODO: Berechne Kovarianz und Korrelation für income/leisure und income/commute.
    # TODO: Füge income_thousand_eur hinzu und wiederhole die erste Rechnung.
    # TODO: Prüfe die Resultate mit pandas und speichere Kovarianz-/Korrelationsmatrix.
    # TODO: Visualisiere Kovarianzbeiträge, Vorzeichen und den Einheiteneffekt.


if __name__ == "__main__":
    main()
