"""Musterlösung für T34: Paneldaten strukturieren."""

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
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "unbalanced_firm_panel.csv", index=False)
    duplicate_keys = int(data.duplicated(["firm_id", "quarter"]).sum())
    counts = data.groupby("firm_id")["quarter"].nunique()
    entities = data["firm_id"].nunique()
    periods = data["quarter"].nunique()
    possible_cells = entities * periods
    diagnostics = pd.DataFrame(
        [{"observations": len(data), "entities": entities, "periods": periods,
          "possible_balanced_cells": possible_cells, "missing_cells": possible_cells - len(data),
          "coverage_rate": len(data) / possible_cells, "duplicate_entity_time_keys": duplicate_keys,
          "min_periods_per_entity": counts.min(), "mean_periods_per_entity": counts.mean(),
          "max_periods_per_entity": counts.max(), "balanced_panel": counts.nunique() == 1}]
    )
    assert duplicate_keys == 0
    assert not bool(diagnostics.loc[0, "balanced_panel"])
    diagnostics.to_csv(RESULTS / "panel_diagnostics.csv", index=False)

    entity_counts = counts.rename("observed_periods").reset_index()
    entity_counts.to_csv(RESULTS / "entity_observation_counts.csv", index=False)
    variation_rows = []
    for variable in VARIABLES:
        entity_mean = data.groupby("firm_id")[variable].transform("mean")
        between_values = data.groupby("firm_id")[variable].mean()
        variation_rows.append(
            {"variable": variable, "overall_sd": data[variable].std(ddof=1),
             "between_sd_of_entity_means": between_values.std(ddof=1),
             "within_sd_after_demeaning": (data[variable] - entity_mean).std(ddof=1)}
        )
    variation = pd.DataFrame(variation_rows)
    variation.to_csv(RESULTS / "within_between_variation.csv", index=False)
    print(diagnostics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nVariation:")
    print(variation.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    observation_matrix = (
        data.assign(observed=1).pivot(index="firm_id", columns="quarter", values="observed")
        .reindex(index=range(1, entities + 1), columns=range(1, periods + 1)).fillna(0)
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].imshow(observation_matrix.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[0, 0].set(title="Beobachtete und fehlende Firmenquartale", xlabel="Quartal",
                   ylabel="Firma", xticks=np.arange(periods), xticklabels=np.arange(1, periods + 1))

    selected_firms = [1, 5, 12, 20, 33, 45, 61, 76]
    for firm_id in selected_firms:
        subset = data[data["firm_id"] == firm_id].sort_values("quarter")
        axes[0, 1].plot(subset["quarter"], subset["productivity_index"], marker="o",
                        alpha=0.75, label=f"Firma {firm_id}")
    axes[0, 1].set(title="Ausgewählte Produktivitätspfade", xlabel="Quartal",
                   ylabel="Produktivitätsindex", xticks=np.arange(1, periods + 1))
    axes[0, 1].legend(frameon=False, fontsize=7, ncol=2)

    positions = np.arange(len(variation))
    relative_total = (
        variation["between_sd_of_entity_means"] + variation["within_sd_after_demeaning"]
    )
    between_share = variation["between_sd_of_entity_means"] / relative_total
    within_share = variation["within_sd_after_demeaning"] / relative_total
    axes[1, 0].barh(positions, between_share, color="#4472C4", label="Between-Anteil")
    axes[1, 0].barh(positions, within_share, left=between_share,
                    color="#ED7D31", label="Within-Anteil")
    axes[1, 0].set(title="Relative Within- und Between-Streuung",
                   xlabel="Anteil an der Summe beider Standardabweichungen", xlim=(0, 1),
                   yticks=positions, yticklabels=variation["variable"])
    axes[1, 0].legend(frameon=False)

    frequency = entity_counts["observed_periods"].value_counts().sort_index()
    axes[1, 1].bar(frequency.index.astype(str), frequency.values, color="#4472C4")
    axes[1, 1].set(title="Unbalanciertes Panel", xlabel="Beobachtete Quartale pro Firma",
                   ylabel="Zahl der Firmen")
    figure.suptitle("Paneldaten verbinden Einheiten- und Zeitvariation")
    figure.tight_layout()
    figure.savefig(RESULTS / "panel_structure.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
