"""Starter für T34: Paneldaten strukturieren."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720
VARIABLES = ["investment_million_eur", "employees", "productivity_index"]


def make_data(entities: int = 80, periods: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    firm_quality = rng.normal(0, 6, entities)
    firm_size = rng.lognormal(np.log(120), 0.45, entities)
    for firm in range(entities):
        for quarter in range(1, periods + 1):
            investment = 1.5 + 0.15 * quarter + 0.012 * firm_size[firm] + rng.normal(0, 0.7)
            employees = firm_size[firm] + 2.2 * quarter + rng.normal(0, 8)
            productivity = 70 + firm_quality[firm] + 1.8 * investment + 0.5 * quarter + rng.normal(0, 3)
            rows.append((firm + 1, quarter, investment, employees, productivity))
    panel = pd.DataFrame(rows, columns=["firm_id", "quarter", *VARIABLES])
    keep = rng.random(len(panel)) > 0.08
    return panel.loc[keep].reset_index(drop=True)


def main() -> None:
    data = make_data()

    # TODO: Prüfe Indexeindeutigkeit und Panelbalance.
    # TODO: Berechne Beobachtungszahlen sowie Within-/Between-Streuung.
    # TODO: Erstelle die Beobachtungsmatrix und ausgewählte Zeitpfade.
    # TODO: Speichere Daten, Tabellen und die Panelgrafik.


if __name__ == "__main__":
    main()
