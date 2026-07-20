"""Starter für T17: R² und Varianzzerlegung."""

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


def make_data(size: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    study_hours = rng.uniform(0, 10, size=size)
    exam_score = 45 + 4.5 * study_hours + rng.normal(0, 12, size=size)
    irrelevant_noise = rng.normal(size=size)
    return pd.DataFrame(
        {"student_id": np.arange(1, size + 1), "study_hours": study_hours,
         "irrelevant_noise": irrelevant_noise, "exam_score": exam_score}
    )


def decompose(model, y: np.ndarray, name: str) -> dict[str, float | str]:
    # TODO: Berechne TSS, RSS, ESS, beide R²-Formeln und den Identitätsfehler.
    raise NotImplementedError


def main() -> None:
    data = make_data()

    # TODO: Schätze Intercept-only, study_hours und das augmentierte Modell.
    # TODO: Zerlege die Variation für alle drei Modelle.
    # TODO: Vergleiche R² und adjustiertes R².
    # TODO: Speichere Daten, Ergebnistabelle und die vierteilige Visualisierung.


if __name__ == "__main__":
    main()
