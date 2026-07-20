"""Starter für T36: Individuelle Fixed Effects."""

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
TRUE_BETA = 1.8


def make_data(people: int = 120, periods: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    ability = rng.normal(size=people)
    degree = (ability + rng.normal(0, 0.7, people) > 0).astype(int)
    rows = []
    for person in range(people):
        for period in range(1, periods + 1):
            training = 5 + 1.4 * ability[person] + 0.25 * period + rng.normal(0, 1.4)
            person_intercept = 50 + 7 * ability[person]
            productivity = person_intercept + TRUE_BETA * training + rng.normal(0, 3.5)
            rows.append((person + 1, period, training, productivity, ability[person], degree[person], person_intercept))
    return pd.DataFrame(rows, columns=["person_id", "period", "training_hours",
                                       "productivity_score", "oracle_ability", "university_degree",
                                       "oracle_person_intercept"])


def main() -> None:
    data = make_data()

    # TODO: Schätze Pooled OLS und das Personen-Dummy-Modell.
    # TODO: Rekonstruiere und zentriere individuelle Intercepts.
    # TODO: Analysiere Within-Variation und zeitinvariante Regressoren.
    # TODO: Speichere Daten, Tabellen und die Individual-FE-Grafik.


if __name__ == "__main__":
    main()
