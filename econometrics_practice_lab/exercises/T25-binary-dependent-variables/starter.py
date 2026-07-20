"""Starter für T25: Binäre abhängige Variablen."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from statsmodels.stats.proportion import proportion_confint


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    income = np.clip(rng.lognormal(np.log(52), 0.45, size=size), 15, 160)
    debt_to_income = np.clip(rng.beta(2.2, 4.5, size=size) * 0.9, 0.02, 0.88)
    late_payments = np.clip(rng.poisson(0.8, size=size), 0, 4)
    linear_index = (
        -3.5 + 4.8 * debt_to_income + 0.58 * late_payments
        - 0.012 * (income - 50)
    )
    true_probability = expit(linear_index)
    defaulted = rng.binomial(1, true_probability)
    return pd.DataFrame(
        {"loan_id": np.arange(1, size + 1), "income_thousand_eur": income,
         "debt_to_income": debt_to_income, "prior_late_payments": late_payments,
         "defaulted": defaulted, "oracle_default_probability": true_probability}
    )


def main() -> None:
    data = make_data()

    # TODO: Verifiziere das 0/1-Outcome und die Bernoulli-Identitäten.
    # TODO: Berechne Risikodezile und Wilson-Intervalle nach Zahlungsverzügen.
    # TODO: Vergleiche Mehrheitsklassen-Accuracy und Brier Scores.
    # TODO: Speichere Daten, Ergebnistabellen und die Diagnosegrafik.


if __name__ == "__main__":
    main()
