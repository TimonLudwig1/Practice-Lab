"""Starter für T19: Hypothesentests in multipler Regression."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720
REGRESSORS = ["experience_years", "education_years", "certifications", "leadership_training"]


def make_data(size: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    experience = rng.uniform(0, 30, size=size)
    education = rng.uniform(10, 20, size=size)
    certifications = rng.poisson(1.5, size=size)
    training = rng.binomial(1, 0.35, size=size)
    salary = 28_000 + 1_600 * experience + 2_400 * education
    salary += 1_300 * certifications + 4_000 * training + rng.normal(0, 8_000, size=size)
    return pd.DataFrame(
        {"employee_id": np.arange(1, size + 1), "experience_years": experience,
         "education_years": education, "certifications": certifications,
         "leadership_training": training, "annual_salary_eur": salary}
    )


def main() -> None:
    data = make_data()

    # TODO: Schätze unrestringiertes und restringiertes Modell.
    # TODO: Berechne Einzeltest, Gleichheitsrestriktion und gemeinsamen F-Test.
    # TODO: Kontrolliere Restriktionen mit statsmodels.f_test.
    # TODO: Speichere Daten, Tabellen und die Testvisualisierung.


if __name__ == "__main__":
    main()
