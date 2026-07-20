"""Starter für T05: Test einer Mittelwertsdifferenz unabhängiger Gruppen."""

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
    score_a = np.clip(rng.normal(loc=71.5, scale=10.0, size=110), 0, 100)
    score_b = np.clip(rng.normal(loc=76.0, scale=13.5, size=95), 0, 100)
    return pd.DataFrame(
        {"student_id": np.arange(1, 206),
         "group": np.repeat(["A_alt", "B_neu"], [len(score_a), len(score_b)]),
         "score": np.concatenate([score_a, score_b])}
    )


def main() -> None:
    data = make_data()
    a = data.loc[data["group"] == "A_alt", "score"].to_numpy()
    b = data.loc[data["group"] == "B_neu", "score"].to_numpy()

    # TODO: Prüfe die Eindeutigkeit der student_id.
    # TODO: Berechne Differenz B-A, Welch-Standardfehler und Freiheitsgrade.
    # TODO: Berechne t-Wert, p-Wert und 95%-Konfidenzintervall.
    # TODO: Verifiziere das Resultat mit stats.ttest_ind.
    # TODO: Speichere Daten, Ergebnistabelle und eine aussagekräftige Grafik.


if __name__ == "__main__":
    main()
