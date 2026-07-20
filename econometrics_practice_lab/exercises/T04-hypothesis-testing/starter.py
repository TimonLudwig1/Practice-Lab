"""Starter für T04: Teststatistik, kritischer Wert und p-Wert."""

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
RNG_SEED = 20260714


def make_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    times = np.clip(rng.normal(loc=31.6, scale=5.8, size=80), 12, None)
    return pd.DataFrame({"delivery_id": np.arange(1, len(times) + 1),
                         "delivery_minutes": times})


def main() -> None:
    data = make_data()
    values = data["delivery_minutes"].to_numpy()
    null_mean = 30.0
    alpha = 0.05

    # TODO: Berechne sample_mean, sample_sd, standard_error, t_statistic und df.
    # TODO: Berechne den positiven kritischen t-Wert und den zweiseitigen p-Wert.
    # TODO: Berechne die Grenzen des 95%-Konfidenzintervalls.
    # TODO: Verifiziere Teststatistik und p-Wert mit stats.ttest_1samp.
    # TODO: Visualisiere Daten sowie t-Nullverteilung, Ablehnungsbereiche und Teststatistik.
    # TODO: Speichere Daten, Ergebnisse und Grafik.


if __name__ == "__main__":
    main()
