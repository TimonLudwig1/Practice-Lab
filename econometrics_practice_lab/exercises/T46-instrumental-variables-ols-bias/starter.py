"""Starter für T46: Instrumental Variables und OLS-Bias."""

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
RNG_SEED = 20260726
TRUE_EFFECT = 2.0


def make_data(observations: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    motivation = rng.normal(size=observations)
    invitation = rng.binomial(1, 0.5, observations)
    invalid_instrument = (0.9 * motivation + rng.normal(size=observations) > 0).astype(int)
    training = 6 + 2.2 * invitation + 1.7 * motivation + rng.normal(0, 1.7, observations)
    income = 18 + TRUE_EFFECT * training + 4.0 * motivation + rng.normal(0, 2.5, observations)
    return pd.DataFrame({"invitation": invitation, "invalid_instrument": invalid_instrument,
                         "training_hours": training, "income_thousand_eur": income,
                         "oracle_motivation": motivation})


def main() -> None:
    data = make_data()

    # TODO: Vergleiche OLS, Oracle-Regression sowie gültigen und ungültigen IV-Schätzer.
    # TODO: Diagnostiziere Relevanz und Zusammenhang des Instruments mit Motivation.
    # TODO: Speichere Daten, Ergebnistabellen und die Vergleichsgrafik.


if __name__ == "__main__":
    main()
