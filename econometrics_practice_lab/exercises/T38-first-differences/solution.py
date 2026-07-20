"""Musterlösung für T38: First Differences."""

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
TRUE_BETA = 0.6


def make_data(households: int = 150, periods: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    stability = rng.normal(size=households)
    rows = []
    for household in range(households):
        income = 35 + 7 * stability[household] + rng.normal(0, 3)
        for year in range(1, periods + 1):
            income += rng.normal(0, 3.2)
            consumption = 12 + 5 * stability[household] + TRUE_BETA * income + rng.normal(0, 3)
            rows.append((household + 1, year, income, consumption, stability[household]))
    return pd.DataFrame(rows, columns=["household_id", "year", "income_thousand_eur",
                                       "consumption_thousand_eur", "oracle_financial_stability"])


def clustered_ols(y: pd.Series, x: pd.DataFrame, groups: pd.Series):
    return sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data().sort_values(["household_id", "year"]).reset_index(drop=True)
    x_name = "income_thousand_eur"
    y_name = "consumption_thousand_eur"
    data["delta_income"] = data.groupby("household_id")[x_name].diff()
    data["delta_consumption"] = data.groupby("household_id")[y_name].diff()
    data[x_name + "_within"] = data[x_name] - data.groupby("household_id")[x_name].transform("mean")
    data[y_name + "_within"] = data[y_name] - data.groupby("household_id")[y_name].transform("mean")
    data.to_csv(DATA / "household_income_panel.csv", index=False)
    differenced = data.dropna(subset=["delta_income", "delta_consumption"]).copy()
    assert len(differenced) == data["household_id"].nunique() * (data["year"].nunique() - 1)

    pooled = clustered_ols(data[y_name], sm.add_constant(data[[x_name]]), data["household_id"])
    within = clustered_ols(data[y_name + "_within"], data[[x_name + "_within"]], data["household_id"])
    first_difference = clustered_ols(
        differenced["delta_consumption"], differenced[["delta_income"]], differenced["household_id"]
    )
    coefficients = pd.DataFrame(
        [{"method": "Pooled OLS", "coefficient": pooled.params[x_name],
          "standard_error": pooled.bse[x_name]},
         {"method": "Within estimator", "coefficient": within.params[x_name + "_within"],
          "standard_error": within.bse[x_name + "_within"]},
         {"method": "First differences", "coefficient": first_difference.params["delta_income"],
          "standard_error": first_difference.bse["delta_income"]}]
    )
    coefficients["true_coefficient"] = TRUE_BETA
    assert abs(first_difference.params["delta_income"] - TRUE_BETA) < 0.12
    coefficients.to_csv(RESULTS / "estimator_comparison.csv", index=False)

    two_periods = data[data["year"].isin([1, 2])].copy()
    two_periods["x_within_two"] = two_periods[x_name] - two_periods.groupby("household_id")[x_name].transform("mean")
    two_periods["y_within_two"] = two_periods[y_name] - two_periods.groupby("household_id")[y_name].transform("mean")
    within_two = sm.OLS(two_periods["y_within_two"], two_periods[["x_within_two"]]).fit()
    differences_two = two_periods.groupby("household_id")[[x_name, y_name]].diff().dropna()
    fd_two = sm.OLS(differences_two[y_name], differences_two[[x_name]]).fit()
    assert np.isclose(within_two.params["x_within_two"], fd_two.params[x_name])
    correlations = pd.DataFrame(
        [{"quantity": "Income level", "correlation_with_stability": data[x_name].corr(data["oracle_financial_stability"])},
         {"quantity": "Income first difference", "correlation_with_stability": differenced["delta_income"].corr(differenced["oracle_financial_stability"])}]
    )
    diagnostics = pd.DataFrame(
        [{"within_t2_coefficient": within_two.params["x_within_two"],
          "fd_t2_coefficient": fd_two.params[x_name],
          "t2_identity_error": within_two.params["x_within_two"] - fd_two.params[x_name],
          "level_observations": len(data), "differenced_observations": len(differenced)}]
    )
    correlations.to_csv(RESULTS / "time_invariant_correlation.csv", index=False)
    diagnostics.to_csv(RESULTS / "first_difference_diagnostics.csv", index=False)
    print(coefficients.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nT=2-Äquivalenz:")
    print(diagnostics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    selected = [3, 17, 31, 55, 88, 121, 147]
    for household_id in selected:
        subset = data[data["household_id"] == household_id]
        axes[0, 0].plot(subset["year"], subset[y_name], marker="o", alpha=0.75,
                        label=f"HH {household_id}")
    axes[0, 0].set(title="Konsumniveaus enthalten Haushaltseffekte", xlabel="Jahr",
                   ylabel="Konsum (Tsd. €)", xticks=np.arange(1, 7))
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)

    axes[0, 1].scatter(differenced["delta_income"], differenced["delta_consumption"],
                       color="#9DC3E6", alpha=0.45, s=18)
    grid = np.linspace(differenced["delta_income"].min(), differenced["delta_income"].max(), 200)
    axes[0, 1].plot(grid, first_difference.params["delta_income"] * grid,
                    color="#C00000", linewidth=2, label=f"FD β={first_difference.params['delta_income']:.2f}")
    axes[0, 1].axhline(0, color="#595959", linewidth=1)
    axes[0, 1].axvline(0, color="#595959", linewidth=1)
    axes[0, 1].set(title="First Differences verbinden Änderungen", xlabel="Δ Einkommen",
                   ylabel="Δ Konsum")
    axes[0, 1].legend(frameon=False)

    positions = np.arange(len(coefficients))
    axes[1, 0].errorbar(coefficients["coefficient"], positions,
                        xerr=1.96 * coefficients["standard_error"], fmt="o",
                        color="#4472C4", capsize=5)
    axes[1, 0].axvline(TRUE_BETA, color="#C00000", linestyle="--", label="Wahrer Effekt")
    axes[1, 0].set(title="FE und FD entfernen zeitinvariante Stabilität",
                   xlabel="Marginaler Konsumeffekt mit 95%-KI", yticks=positions,
                   yticklabels=coefficients["method"])
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(correlations["quantity"], correlations["correlation_with_stability"],
                   color=["#4472C4", "#ED7D31"])
    axes[1, 1].set_xticks(np.arange(2), ["Einkommensniveau", "Erste Differenz"])
    axes[1, 1].axhline(0, color="#595959", linewidth=1)
    axes[1, 1].set(title="Differenzieren entfernt Niveaukorrelation",
                   ylabel="Korrelation mit zeitinvarianter Stabilität")
    figure.suptitle("First Differences eliminieren konstante Haushaltseffekte")
    figure.tight_layout()
    figure.savefig(RESULTS / "first_differences.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
