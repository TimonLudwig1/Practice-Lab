"""Starter für T06: Gepaarte Stichproben."""

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
    before = np.clip(rng.normal(loc=22.0, scale=5.0, size=60), 8, None)
    after = np.clip(before - 1.4 + rng.normal(loc=0, scale=2.0, size=60), 5, None)
    return pd.DataFrame(
        {"household_id": np.arange(1, 61), "before_kwh": before, "after_kwh": after}
    )


def main() -> None:
    data = make_data()
    before = data["before_kwh"].to_numpy()
    after = data["after_kwh"].to_numpy()

    # TODO: Prüfe Eindeutigkeit und Vollständigkeit der Paare.
    # TODO: Berechne change = after - before sowie dessen Mittel, SD und SE.
    # TODO: Berechne gepaarten t-Wert, p-Wert und 95%-Konfidenzintervall.
    # TODO: Vergleiche mit stats.ttest_rel und einem fälschlichen Welch-Test.
    # TODO: Mische after zufällig und vergleiche die SD der Differenzen.
    # TODO: Speichere Daten, Ergebnistabelle und Visualisierung.


if __name__ == "__main__":
    main()
