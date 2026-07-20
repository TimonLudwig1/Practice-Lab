"""Starter für T28: Marginale Effekte im Logit-Modell."""

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
REGRESSORS = ["preparation_hours", "experience_years", "mentor_program"]


def make_data(size: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    preparation = rng.uniform(0, 60, size=size)
    experience = rng.uniform(0, 12, size=size)
    mentor = rng.binomial(1, 0.42, size=size)
    linear_index = -5.1 + 0.095 * preparation + 0.17 * experience + 0.85 * mentor
    probability = expit(linear_index)
    offer = rng.binomial(1, probability)
    return pd.DataFrame(
        {"applicant_id": np.arange(1, size + 1), "preparation_hours": preparation,
         "experience_years": experience, "mentor_program": mentor,
         "received_offer": offer, "oracle_offer_probability": probability}
    )


def main() -> None:
    data = make_data()

    # TODO: Schätze das Logit-Modell.
    # TODO: Berechne individuelle Ableitungen und diskrete Dummy-Effekte.
    # TODO: Vergleiche AME und MEM mit statsmodels und ihren Intervallen.
    # TODO: Speichere Daten, Tabellen und die Grafik marginaler Effekte.


if __name__ == "__main__":
    main()
