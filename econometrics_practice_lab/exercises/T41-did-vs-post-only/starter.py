"""Starter für T41: DiD versus Post-Treatment-Daten allein."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260721
TRUE_EFFECT = 5.0


def make_data(schools: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    treated = np.repeat([0, 1], schools // 2)
    school_effect = rng.normal(0, 2.2, schools)
    rows = []
    for school in range(schools):
        for post in [0, 1]:
            score = (67 - 10 * treated[school] + school_effect[school] + 2.5 * post
                     + TRUE_EFFECT * treated[school] * post + rng.normal(0, 1.2))
            rows.append((school + 1, post, treated[school], score))
    return pd.DataFrame(rows, columns=["school_id", "post", "treated", "math_score"])


def main() -> None:
    data = make_data()

    # TODO: Vergleiche den Post-only-Schätzer mit dem vollständigen DiD.
    # TODO: Berechne schulweise Veränderungen und rekonstruiere das Gegenfaktum.
    # TODO: Speichere Daten, Ergebnistabellen und eine erklärende Grafik.


if __name__ == "__main__":
    main()
