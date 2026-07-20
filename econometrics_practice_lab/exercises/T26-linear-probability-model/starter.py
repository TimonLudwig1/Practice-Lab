"""Starter für T26: Linear Probability Model."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    study_hours = rng.uniform(0, 22, size=size)
    true_probability = expit(-4.5 + 0.43 * study_hours)
    completed = rng.binomial(1, true_probability)
    return pd.DataFrame(
        {"learner_id": np.arange(1, size + 1),
         "study_hours_per_week": study_hours, "completed": completed,
         "oracle_completion_probability": true_probability}
    )


def main() -> None:
    data = make_data()

    # TODO: Schätze das LPM und berechne klassische sowie HC1-Inferenz.
    # TODO: Untersuche ungültige Fits und führe den Breusch–Pagan-Test durch.
    # TODO: Vergleiche gebinnte Abschlussquoten mit dem linearen Fit.
    # TODO: Speichere Daten, Tabellen und die Diagnosegrafik.


if __name__ == "__main__":
    main()
