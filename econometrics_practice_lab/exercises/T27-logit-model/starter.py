"""Starter für T27: Logit-Modell."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720
REGRESSORS = ["price_increase_pct", "loyalty_years", "automatic_payment"]


def make_data(size: int = 1400) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    price_increase = rng.uniform(0, 25, size=size)
    loyalty = rng.uniform(0, 8, size=size)
    automatic_payment = rng.binomial(1, 0.58, size=size)
    linear_index = 2.3 - 0.18 * price_increase + 0.24 * loyalty + 0.72 * automatic_payment
    true_probability = expit(linear_index)
    renewed = rng.binomial(1, true_probability)
    return pd.DataFrame(
        {"customer_id": np.arange(1, size + 1), "price_increase_pct": price_increase,
         "loyalty_years": loyalty, "automatic_payment": automatic_payment,
         "renewed": renewed, "oracle_renewal_probability": true_probability}
    )


def main() -> None:
    data = make_data()

    # TODO: Schätze das Logit-Modell und berechne Odds Ratios samt Intervallen.
    # TODO: Erzeuge Szenarien auf Index-, Odds- und Wahrscheinlichkeitsskala.
    # TODO: Berechne Kalibrierungsdezile und Modellfit-Kennzahlen.
    # TODO: Speichere Daten, Tabellen und die Transformationsgrafik.


if __name__ == "__main__":
    main()
