"""Musterlösung für T42: DiD mit individuellen Fixed Effects."""

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
RNG_SEED = 20260722
TRUE_EFFECT = -2.5


def make_data(households: int = 120, months: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    efficiency = rng.normal(size=households)
    treated = (efficiency < np.median(efficiency)).astype(int)
    household_noise = rng.normal(0, 1.2, households)
    rows = []
    for household in range(households):
        for month in range(1, months + 1):
            post = int(month >= 5)
            consumption = (21 - 3.5 * efficiency[household] + household_noise[household]
                           - 0.7 * post + TRUE_EFFECT * treated[household] * post
                           + rng.normal(0, 1.0))
            rows.append((household + 1, month, treated[household], post, consumption,
                         efficiency[household]))
    return pd.DataFrame(rows, columns=["household_id", "month", "treated", "post",
                                       "electricity_mwh", "oracle_building_efficiency"])


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data["did"] = data["treated"] * data["post"]
    data.to_csv(DATA / "household_energy_panel.csv", index=False)
    counts = data.groupby("household_id")["month"].nunique()
    assert counts.eq(8).all()

    pooled_design = sm.add_constant(data[["post", "did"]])
    pooled = sm.OLS(data["electricity_mwh"], pooled_design).fit(
        cov_type="cluster", cov_kwds={"groups": data["household_id"]}
    )
    household_dummies = pd.get_dummies(
        data["household_id"], prefix="household", drop_first=True, dtype=float
    )
    fe_design = sm.add_constant(pd.concat([data[["post", "did"]], household_dummies], axis=1))
    fe_model = sm.OLS(data["electricity_mwh"], fe_design).fit(
        cov_type="cluster", cov_kwds={"groups": data["household_id"]}
    )

    demean_columns = ["electricity_mwh", "post", "did", "treated", "oracle_building_efficiency"]
    demeaned = data[demean_columns] - data.groupby("household_id")[demean_columns].transform("mean")
    within = sm.OLS(demeaned["electricity_mwh"], demeaned[["post", "did"]]).fit(
        cov_type="cluster", cov_kwds={"groups": data["household_id"]}
    )
    max_treated_within = demeaned["treated"].abs().max()
    max_efficiency_within = demeaned["oracle_building_efficiency"].abs().max()
    assert max_treated_within < 1e-12
    assert max_efficiency_within < 1e-12
    assert abs(fe_model.params["did"] - within.params["did"]) < 1e-10
    assert abs(fe_model.params["did"] - TRUE_EFFECT) < 0.35

    all_household_dummies = pd.get_dummies(data["household_id"], dtype=float)
    invalid_design = pd.concat(
        [all_household_dummies.reset_index(drop=True), data[["treated"]].reset_index(drop=True)], axis=1
    ).to_numpy()
    design_rank = np.linalg.matrix_rank(invalid_design)
    rank_deficient = design_rank < invalid_design.shape[1]
    assert rank_deficient

    estimates = pd.DataFrame([
        {"method": "Pooled without household controls", "did_estimate": pooled.params["did"],
         "cluster_standard_error": pooled.bse["did"]},
        {"method": "Household dummy FE", "did_estimate": fe_model.params["did"],
         "cluster_standard_error": fe_model.bse["did"]},
        {"method": "Within estimator", "did_estimate": within.params["did"],
         "cluster_standard_error": within.bse["did"]},
    ])
    estimates.to_csv(RESULTS / "individual_fe_estimates.csv", index=False)
    diagnostics = pd.DataFrame([{
        "max_abs_within_treated": max_treated_within,
        "max_abs_within_efficiency": max_efficiency_within,
        "invalid_design_columns": invalid_design.shape[1],
        "invalid_design_rank": design_rank,
        "rank_deficient": rank_deficient,
        "dummy_within_coefficient_difference": abs(fe_model.params["did"] - within.params["did"]),
    }])
    diagnostics.to_csv(RESULTS / "individual_fe_diagnostics.csv", index=False)

    print(estimates.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nDiagnostik:")
    print(diagnostics.to_string(index=False, float_format=lambda value: f"{value:.3e}"))

    means = data.groupby(["month", "treated"], as_index=False)["electricity_mwh"].mean()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for group, label, color in [(0, "Kontrollhaushalte", "#595959"),
                                (1, "Geförderte Haushalte", "#4472C4")]:
        subset = means.loc[means["treated"].eq(group)]
        axes[0].plot(subset["month"], subset["electricity_mwh"], marker="o",
                     color=color, label=label)
    axes[0].axvline(4.5, color="#C00000", linestyle="--")
    axes[0].set(title="Dauerhafte Niveauunterschiede bleiben sichtbar",
                xlabel="Monat", ylabel="Stromverbrauch (MWh)")
    axes[0].legend(frameon=False)

    plot_estimates = estimates.copy()
    y_pos = np.arange(len(plot_estimates))
    axes[1].errorbar(plot_estimates["did_estimate"], y_pos,
                     xerr=1.96 * plot_estimates["cluster_standard_error"], fmt="o",
                     color="#4472C4", capsize=5)
    axes[1].axvline(TRUE_EFFECT, color="#C00000", linestyle="--", label="Wahrer Effekt")
    axes[1].set(title="FE und Within nutzen dieselbe Variation",
                xlabel="DiD-Effekt (MWh)", yticks=y_pos,
                yticklabels=["Pooled ohne\nHaushaltskontrolle", "Haushaltsdummies", "Within"])
    axes[1].legend(frameon=False)
    figure.suptitle("Individuelle Fixed Effects absorbieren konstante Haushaltsunterschiede")
    figure.tight_layout()
    figure.savefig(RESULTS / "individual_fe_did.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
