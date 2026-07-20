"""Starter für T12: Nichtlineare Beziehungen und Residuen."""

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


def make_data(size: int = 365) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    temperature = rng.uniform(-5, 35, size=size)
    energy = 82 + 1.15 * (temperature - 18) ** 2 + rng.normal(0, 20, size=size)
    return pd.DataFrame(
        {"day": np.arange(1, size + 1), "temperature_c": temperature, "energy_kwh": energy}
    )


def main() -> None:
    data = make_data()

    # TODO: Schätze ein lineares Modell.
    # TODO: Ergänze temperature_squared und schätze ein quadratisches Modell.
    # TODO: Vergleiche RMSE, R² und Residuen nach Temperaturintervallen.
    # TODO: Berechne das Minimum der quadratischen Kurve.
    # TODO: Speichere Daten, Tabellen und die vierteilige Diagnosegrafik.


if __name__ == "__main__":
    main()
