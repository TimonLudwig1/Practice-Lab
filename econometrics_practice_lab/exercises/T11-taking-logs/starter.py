"""Starter für T11: Log-Transformationen und ihr Definitionsbereich."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    employees = np.maximum(1, np.rint(rng.lognormal(mean=4.0, sigma=1.0, size=size))).astype(int)
    revenue = 40_000 * employees ** 0.82 * np.exp(rng.normal(0, 0.42, size=size))
    profit = 0.07 * revenue + rng.normal(0, 250_000, size=size)
    return pd.DataFrame(
        {"firm_id": np.arange(1, size + 1), "employees": employees,
         "revenue_eur": revenue, "profit_eur": profit}
    )


def describe_scale(values: pd.Series, variable: str, scale: str) -> dict[str, float | str]:
    # TODO: Gib variable, scale, n, mean, median, sd und skewness zurück.
    raise NotImplementedError


def main() -> None:
    data = make_data()

    # TODO: Prüfe Positivität und erzeuge log_employees sowie log_revenue.
    # TODO: Beschreibe Roh- und Log-Skalen in einer gemeinsamen Tabelle.
    # TODO: Untersuche, wie viele Gewinne nicht logarithmierbar sind.
    # TODO: Speichere Daten, Tabellen und vier Verteilungsgrafiken.


if __name__ == "__main__":
    main()
