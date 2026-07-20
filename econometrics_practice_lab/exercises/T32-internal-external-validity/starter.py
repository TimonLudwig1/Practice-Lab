"""Starter für T32: Interne und externe Validität."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 16000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    age = np.clip(rng.normal(52, 14, size=size), 18, 85)
    severity = rng.normal(size=size)
    rural = rng.binomial(1, 0.30, size=size)
    selection_probability = np.clip(
        expit(-1.15 + 0.85 * severity - 0.032 * (age - 50) - 0.45 * rural), 0.05, 0.85
    )
    participant = rng.binomial(1, selection_probability)
    treatment = np.where(participant == 1, rng.binomial(1, 0.5, size=size), np.nan)
    outcome_zero = 65 - 5 * severity - 0.08 * age - 2 * rural + rng.normal(0, 6, size=size)
    treatment_effect = 4 + 1.6 * severity + 0.035 * (age - 50)
    outcome_one = outcome_zero + treatment_effect
    observed = np.where(participant == 0, np.nan,
                        np.where(treatment == 1, outcome_one, outcome_zero))
    return pd.DataFrame(
        {"person_id": np.arange(1, size + 1), "age": age, "baseline_severity": severity,
         "rural": rural, "selection_probability": selection_probability,
         "trial_participant": participant, "treatment": treatment,
         "outcome_y0": outcome_zero, "outcome_y1": outcome_one,
         "individual_treatment_effect": treatment_effect, "observed_health_score": observed}
    )


def main() -> None:
    data = make_data()

    # TODO: Vergleiche Population und Trial-Stichprobe.
    # TODO: Berechne Sample ATE, Population ATE und Trial-Schätzung.
    # TODO: Generalisiere mit inversen Teilnahmewahrscheinlichkeiten.
    # TODO: Speichere Daten, Tabellen und die Validitätsgrafik.


if __name__ == "__main__":
    main()
