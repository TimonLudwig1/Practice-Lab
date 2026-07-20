"""Starter für T33: Treatment-Effekte in Experimenten."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    baseline = rng.normal(60, 11, size=size)
    high_need = (baseline < 55).astype(int)
    treatment = np.zeros(size, dtype=int)
    treatment[rng.choice(size, size=size // 2, replace=False)] = 1
    outcome_zero = 28 + 0.72 * baseline - 3 * high_need + rng.normal(0, 7, size=size)
    treatment_effect = 4 + 3 * high_need
    outcome_one = outcome_zero + treatment_effect
    observed = np.where(treatment == 1, outcome_one, outcome_zero)
    return pd.DataFrame(
        {"employee_id": np.arange(1, size + 1), "baseline_score": baseline,
         "high_need": high_need, "treatment": treatment, "outcome_y0": outcome_zero,
         "outcome_y1": outcome_one, "individual_treatment_effect": treatment_effect,
         "followup_score": observed}
    )


def main() -> None:
    data = make_data()

    # TODO: Schätze Mittelwertsdifferenz und Welch-Konfidenzintervall.
    # TODO: Vergleiche Treatment-only- und kovariatenadjustierte Regression.
    # TODO: Schätze Subgruppeneffekte über eine Interaktion.
    # TODO: Führe Randomisierungsinferenz durch und speichere alle Ergebnisse.


if __name__ == "__main__":
    main()
