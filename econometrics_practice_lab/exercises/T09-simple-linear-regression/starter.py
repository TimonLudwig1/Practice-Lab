"""Starter für T09: Einfache lineare Regression und Gauss–Markov."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    training = rng.uniform(2, 25, size=size)
    structural_error = rng.normal(0, 8, size=size)
    productivity = 50 + 2.2 * training + structural_error
    return pd.DataFrame(
        {"firm_id": np.arange(1, size + 1), "training_hours": training,
         "productivity_index": productivity, "structural_error": structural_error}
    )


def manual_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Gib Achsenabschnitt und Steigung zurück."""
    # TODO: Verwende die Kovarianz-/Varianz-Formel für die Steigung.
    raise NotImplementedError


def main() -> None:
    data = make_data()
    x = data["training_hours"].to_numpy()
    y = data["productivity_index"].to_numpy()

    # TODO: Schätze OLS manuell und mit statsmodels.
    # TODO: Berechne Fits, Residuen und ihre OLS-Orthogonalitätsbedingungen.
    # TODO: Gruppiere nach Trainingsstunden-Quintilen und untersuche den wahren Fehler.
    # TODO: Speichere Daten, Tabellen und eine vierteilige Diagnosegrafik.


if __name__ == "__main__":
    main()
