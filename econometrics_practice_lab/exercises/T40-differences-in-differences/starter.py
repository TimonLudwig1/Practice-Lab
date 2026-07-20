"""Starter für T40: klassisches Differences-in-Differences."""

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
RNG_SEED = 20260720
TRUE_EFFECT = -6.0


def make_data(cities: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    treated = np.repeat([0, 1], cities // 2)
    city_effect = rng.normal(0, 2.4, cities)
    year_shocks = {2017: 0.0, 2018: -0.7, 2019: -1.4, 2020: -2.3,
                   2021: -2.9, 2022: -3.8, 2023: -4.5}
    rows = []
    for city in range(cities):
        for year in range(2017, 2024):
            post = int(year >= 2021)
            no2 = (39 + 7.5 * treated[city] + city_effect[city] + year_shocks[year]
                   + TRUE_EFFECT * treated[city] * post + rng.normal(0, 1.8))
            rows.append((city + 1, year, treated[city], post, no2))
    return pd.DataFrame(rows, columns=["city_id", "year", "treated", "post", "no2_ug_m3"])


def main() -> None:
    data = make_data()

    # TODO: Berechne die vier Gruppen-Zeit-Mittelwerte und den DiD-Schätzer.
    # TODO: Schätze das Interaktionsmodell mit geclusterten Standardfehlern.
    # TODO: Untersuche die Pre-Trends und erstelle die Ergebnisgrafik.
    # TODO: Speichere Daten und Ergebnistabellen in DATA beziehungsweise RESULTS.


if __name__ == "__main__":
    main()
