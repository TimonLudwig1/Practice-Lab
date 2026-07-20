"""Starter für T13: Level- und Log-Modelle."""

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


def make_data(size_per_scenario: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    specs = []

    x = rng.uniform(1, 20, size_per_scenario)
    specs.append(("level_level", x, 12 + 3.2 * x + rng.normal(0, 5, size_per_scenario)))

    x = rng.uniform(9, 20, size_per_scenario)
    specs.append(("log_level", x, np.exp(2.4 + 0.075 * x + rng.normal(0, 0.12, size_per_scenario))))

    x = rng.lognormal(8.0, 0.8, size_per_scenario)
    specs.append(("level_log", x, 30 + 18 * np.log(x) + rng.normal(0, 12, size_per_scenario)))

    x = rng.uniform(5, 30, size_per_scenario)
    specs.append(("log_log", x, np.exp(7.0 - 1.3 * np.log(x) + rng.normal(0, 0.18, size_per_scenario))))

    frames = []
    for scenario, x, y in specs:
        frames.append(pd.DataFrame({"scenario": scenario, "observation_id": np.arange(1, len(x) + 1),
                                    "x": x, "y": y}))
    return pd.concat(frames, ignore_index=True)


def transform_for_model(group: pd.DataFrame, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    """Gib transformiertes X und Y für das angegebene Modell zurück."""
    # TODO: Implementiere die vier Kombinationen aus Level und natürlichem Log.
    raise NotImplementedError


def main() -> None:
    data = make_data()

    # TODO: Schätze jedes Szenario in seiner korrekten transformierten Form.
    # TODO: Berechne modellgerechte Interpretationsgrößen.
    # TODO: Transformiere Fits für die Grafik zurück in Originaleinheiten.
    # TODO: Speichere Daten, Ergebnistabelle und vier Small-Multiple-Plots.


if __name__ == "__main__":
    main()
