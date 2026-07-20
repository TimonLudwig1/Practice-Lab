"""Starter für T24: Interaktionseffekte und marginale Effekte."""

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


def make_data(size: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    experience = rng.uniform(0, 15, size=size)
    treated = rng.binomial(1, 0.5, size=size)
    score = 50 + 1.6 * experience + 4 * treated + 0.9 * experience * treated
    score += rng.normal(0, 6, size=size)
    return pd.DataFrame(
        {"employee_id": np.arange(1, size + 1), "experience_years": experience,
         "treated": treated, "performance_score": score}
    )


def marginal_effect(model, experience: float) -> dict[str, float]:
    # TODO: Berechne beta_treated + experience*beta_interaction und seinen SE.
    raise NotImplementedError


def main() -> None:
    data = make_data()

    # TODO: Schätze Modell ohne und mit Treatment×Erfahrung.
    # TODO: Berechne marginale Treatment-Effekte und Konfidenzintervalle.
    # TODO: Vergleiche Fits und Residuenmuster beider Modelle.
    # TODO: Speichere Daten, Tabellen und die Interaktionsgrafik.


if __name__ == "__main__":
    main()
