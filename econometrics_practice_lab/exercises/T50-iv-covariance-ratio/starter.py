"""Starter für T50: IV als Kovarianzquotient und 2SLS."""

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
RNG_SEED = 20260730
TRUE_EFFECT = 1.5


def make_data(observations: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    experience = rng.normal(size=observations)
    ability = rng.normal(size=observations)
    scholarship = 0.6 * experience + rng.normal(size=observations)
    education = (12 + 1.6 * scholarship + 0.8 * experience + 1.2 * ability
                 + rng.normal(0, 1.5, observations))
    wage = (5 + TRUE_EFFECT * education + 2.0 * experience + 3.0 * ability
            + rng.normal(0, 2.5, observations))
    return pd.DataFrame({"scholarship_index": scholarship, "education_years": education,
                         "wage_index": wage, "experience_index": experience,
                         "oracle_ability": ability})


def main() -> None:
    data = make_data()

    # TODO: Berechne unkonditionale und residualisierte Kovarianzquotienten.
    # TODO: Reproduziere den kontrollierten IV-Schätzer über RF/FS und 2SLS.
    # TODO: Prüfe Skalierungsinvarianz, bootstrappe Unsicherheit und erstelle die Grafik.


if __name__ == "__main__":
    main()
