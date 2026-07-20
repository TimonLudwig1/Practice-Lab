"""Starter für T49: Local Average Treatment Effect."""

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
RNG_SEED = 20260729


def make_data(observations: int = 20000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    compliance_type = rng.choice(["always", "complier", "never"], observations,
                                 p=[0.30, 0.30, 0.40])
    assignment = rng.binomial(1, 0.5, observations)
    d_if_z0 = (compliance_type == "always").astype(int)
    d_if_z1 = (compliance_type != "never").astype(int)
    treatment = np.where(assignment == 1, d_if_z1, d_if_z0)
    effects = pd.Series(compliance_type).map({"always": 12.0, "complier": 2.0, "never": -3.0}).to_numpy()
    baseline = 50 + pd.Series(compliance_type).map({"always": 7.0, "complier": 0.0, "never": -4.0}).to_numpy()
    outcome = baseline + effects * treatment + rng.normal(0, 2, observations)
    return pd.DataFrame({"assignment_z": assignment, "treatment_d": treatment,
                         "outcome": outcome, "compliance_type": compliance_type,
                         "d_if_z0": d_if_z0, "d_if_z1": d_if_z1,
                         "oracle_treatment_effect": effects})


def main() -> None:
    data = make_data()

    # TODO: Leite Compliance-Anteile aus den Take-up-Raten her.
    # TODO: Berechne First Stage, Reduced Form, Wald-LATE, ATE und naiven Vergleich.
    # TODO: Prüfe Monotonie und speichere Daten, Tabellen und Grafik.


if __name__ == "__main__":
    main()
