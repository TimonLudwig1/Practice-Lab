"""Starter für T29: Kausalität und Selektionsbias."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    motivation = rng.normal(size=size)
    education = np.clip(13 + 1.1 * motivation + rng.normal(0, 1.4, size=size), 9, 20)
    selection_probability = expit(-0.7 + 1.25 * motivation + 0.16 * (education - 13))
    treatment = rng.binomial(1, selection_probability)
    outcome_zero = 52 + 5.5 * motivation + 1.7 * education + rng.normal(0, 4, size=size)
    treatment_effect = 4 + 1.4 * motivation
    outcome_one = outcome_zero + treatment_effect
    observed = np.where(treatment == 1, outcome_one, outcome_zero)
    return pd.DataFrame(
        {"employee_id": np.arange(1, size + 1), "motivation_score": motivation,
         "education_years": education, "selection_probability": selection_probability,
         "treatment": treatment, "outcome_y0": outcome_zero, "outcome_y1": outcome_one,
         "individual_treatment_effect": treatment_effect,
         "observed_performance": observed}
    )


def main() -> None:
    data = make_data()

    # TODO: Berechne ATE, ATT, naive Differenz und Selektionsbias.
    # TODO: Prüfe die Zerlegung des naiven Vergleichs.
    # TODO: Erstelle Gruppenprofile und Motivations-Bins.
    # TODO: Speichere Daten, Tabellen und die Kausalgrafik.


if __name__ == "__main__":
    main()
