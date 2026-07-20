"""Starter für T42: DiD mit individuellen Fixed Effects."""

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
RNG_SEED = 20260722
TRUE_EFFECT = -2.5


def make_data(households: int = 120, months: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    efficiency = rng.normal(size=households)
    treated = (efficiency < np.median(efficiency)).astype(int)
    household_noise = rng.normal(0, 1.2, households)
    rows = []
    for household in range(households):
        for month in range(1, months + 1):
            post = int(month >= 5)
            consumption = (21 - 3.5 * efficiency[household] + household_noise[household]
                           - 0.7 * post + TRUE_EFFECT * treated[household] * post
                           + rng.normal(0, 1.0))
            rows.append((household + 1, month, treated[household], post, consumption,
                         efficiency[household]))
    return pd.DataFrame(rows, columns=["household_id", "month", "treated", "post",
                                       "electricity_mwh", "oracle_building_efficiency"])


def main() -> None:
    data = make_data()

    # TODO: Schätze Pooled- und Haushalts-FE-Modell.
    # TODO: Reproduziere den FE-Schätzer durch Demeaning.
    # TODO: Untersuche absorbierte zeitinvariante Variablen und den Matrixrang.
    # TODO: Speichere Daten, Resultate und die Diagnosegrafik.


if __name__ == "__main__":
    main()
