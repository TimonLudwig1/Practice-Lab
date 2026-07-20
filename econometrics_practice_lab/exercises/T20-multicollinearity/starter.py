"""Starter für T20: Multikollinearität diagnostizieren."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720
REGRESSORS = ["insulation_index", "thermal_score"]


def make_data(size: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    latent_efficiency = rng.normal(size=size)
    insulation = 50 + 10 * latent_efficiency + rng.normal(0, 1.2, size=size)
    thermal = 50 + 10 * latent_efficiency + rng.normal(0, 1.2, size=size)
    energy_cost = 4_000 - 25 * insulation - 25 * thermal + rng.normal(0, 250, size=size)
    return pd.DataFrame(
        {"building_id": np.arange(1, size + 1), "insulation_index": insulation,
         "thermal_score": thermal, "annual_energy_cost_eur": energy_cost}
    )


def main() -> None:
    data = make_data()

    # TODO: Schätze zwei Einzelmodelle und das gemeinsame Modell.
    # TODO: Berechne Korrelation und VIFs.
    # TODO: Bootstrappe die gemeinsamen Koeffizienten und ihre Summe 600-mal.
    # TODO: Speichere Daten, Tabellen und die Multikollinearitätsgrafik.


if __name__ == "__main__":
    main()
