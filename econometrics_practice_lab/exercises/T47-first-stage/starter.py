"""Starter für T47: First Stage und schwache Instrumente."""

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
RNG_SEED = 20260727
TRUE_EFFECT = 1.5


def make_sample(rng: np.random.Generator, strength: float, observations: int = 500) -> pd.DataFrame:
    instrument = rng.binomial(1, 0.5, observations)
    motivation = rng.normal(size=observations)
    coaching = 3 + strength * instrument + 1.2 * motivation + rng.normal(0, 1.5, observations)
    outcome = 10 + TRUE_EFFECT * coaching + 3 * motivation + rng.normal(0, 2, observations)
    return pd.DataFrame({"instrument": instrument, "coaching_hours": coaching,
                         "performance": outcome, "oracle_motivation": motivation})


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    strong = make_sample(rng, strength=2.0)
    weak = make_sample(rng, strength=0.2)

    # TODO: Schätze und diagnostiziere beide First Stages.
    # TODO: Simuliere die Verteilungen der zugehörigen IV-Schätzer.
    # TODO: Speichere Daten, Monte-Carlo-Ergebnisse und die Diagnosegrafik.


if __name__ == "__main__":
    main()
