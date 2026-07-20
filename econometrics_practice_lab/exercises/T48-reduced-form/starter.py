"""Starter für T48: Reduced Form."""

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
RNG_SEED = 20260728
TRUE_EFFECT = 1.8


def make_data(observations: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    invitation = rng.binomial(1, 0.5, observations)
    drive = rng.normal(size=observations)
    sessions = 1.5 + 1.7 * invitation + 1.1 * drive + rng.normal(0, 1.2, observations)
    job_score = 45 + TRUE_EFFECT * sessions + 3.2 * drive + rng.normal(0, 2.2, observations)
    return pd.DataFrame({"invitation": invitation, "counseling_sessions": sessions,
                         "job_search_score": job_score, "oracle_drive": drive})


def main() -> None:
    data = make_data()

    # TODO: Schätze First Stage, Reduced Form und Wald-Ratio.
    # TODO: Prüfe die Produktidentität und untersuche sie im Bootstrap.
    # TODO: Speichere Daten, Tabellen und die Zerlegungsgrafik.


if __name__ == "__main__":
    main()
