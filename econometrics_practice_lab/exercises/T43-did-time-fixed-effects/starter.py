"""Starter für T43: DiD mit Zeit-Fixed-Effects."""

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
RNG_SEED = 20260723
TRUE_EFFECT = 3.5
TIME_SHOCKS = np.array([0.0, 4.0, -2.0, 5.0, 1.0, -3.0, 6.0, 2.0, -1.0, 4.0, 0.0, -4.0])


def make_data(regions: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    treated = np.repeat([0, 1], regions // 2)
    region_effect = rng.normal(0, 2.0, regions)
    rows = []
    for region in range(regions):
        for month, time_shock in enumerate(TIME_SHOCKS, start=1):
            post = int(month >= 7)
            did = treated[region] * post
            visits = (50 + 5 * treated[region] + region_effect[region] + time_shock
                      + TRUE_EFFECT * did + rng.normal(0, 1.5))
            rows.append((region + 1, month, treated[region], post, did, visits, time_shock))
    return pd.DataFrame(rows, columns=["region_id", "month", "treated", "post", "did",
                                       "visits_per_1000", "oracle_time_shock"])


def main() -> None:
    data = make_data()

    # TODO: Schätze naives, klassisches DiD- und Zeit-FE-Modell.
    # TODO: Rekonstruiere und zentriere die geschätzten Zeit-Fixed-Effects.
    # TODO: Vergleiche Residuenmuster, speichere Resultate und erstelle die Grafik.


if __name__ == "__main__":
    main()
