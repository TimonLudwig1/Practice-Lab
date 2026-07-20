"""Musterlösung für T37: Within-Estimator."""

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
TRUE_BETA = 1.4


def make_data(firms: int = 90, periods: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    firm_opportunity = rng.normal(0, 4, firms)
    rows = []
    for firm in range(firms):
        for period in range(1, periods + 1):
            cash_flow = 6 + 0.65 * firm_opportunity[firm] + 0.25 * period + rng.normal(0, 1.8)
            investment = 12 + firm_opportunity[firm] + TRUE_BETA * cash_flow + rng.normal(0, 2.5)
            rows.append((firm + 1, period, cash_flow, investment, firm_opportunity[firm]))
    return pd.DataFrame(rows, columns=["firm_id", "period", "cash_flow_million_eur",
                                       "investment_million_eur", "oracle_firm_opportunity"])


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    x_name = "cash_flow_million_eur"
    y_name = "investment_million_eur"
    data["cash_flow_within"] = data[x_name] - data.groupby("firm_id")[x_name].transform("mean")
    data["investment_within"] = data[y_name] - data.groupby("firm_id")[y_name].transform("mean")
    data.to_csv(DATA / "firm_investment_panel.csv", index=False)
    max_abs_group_mean = max(
        data.groupby("firm_id")["cash_flow_within"].mean().abs().max(),
        data.groupby("firm_id")["investment_within"].mean().abs().max()
    )
    assert max_abs_group_mean < 1e-12

    within = sm.OLS(data["investment_within"], data[["cash_flow_within"]]).fit(
        cov_type="cluster", cov_kwds={"groups": data["firm_id"]}
    )
    manual_beta = (
        (data["cash_flow_within"] * data["investment_within"]).sum()
        / (data["cash_flow_within"] ** 2).sum()
    )
    firm_dummies = pd.get_dummies(data["firm_id"], prefix="firm", drop_first=True, dtype=float)
    dummy_design = sm.add_constant(pd.concat([data[[x_name]], firm_dummies], axis=1))
    dummy = sm.OLS(data[y_name], dummy_design).fit(
        cov_type="cluster", cov_kwds={"groups": data["firm_id"]}
    )
    pooled = sm.OLS(data[y_name], sm.add_constant(data[[x_name]])).fit(
        cov_type="cluster", cov_kwds={"groups": data["firm_id"]}
    )
    assert np.isclose(within.params["cash_flow_within"], manual_beta)
    assert np.isclose(within.params["cash_flow_within"], dummy.params[x_name])
    coefficients = pd.DataFrame(
        [{"method": "Pooled OLS", "coefficient": pooled.params[x_name],
          "standard_error": pooled.bse[x_name]},
         {"method": "Dummy fixed effects", "coefficient": dummy.params[x_name],
          "standard_error": dummy.bse[x_name]},
         {"method": "Within estimator", "coefficient": within.params["cash_flow_within"],
          "standard_error": within.bse["cash_flow_within"]},
         {"method": "Manual within ratio", "coefficient": manual_beta,
          "standard_error": np.nan}]
    )
    coefficients["true_coefficient"] = TRUE_BETA
    coefficients.to_csv(RESULTS / "coefficient_equivalence.csv", index=False)
    diagnostics = pd.DataFrame(
        [{"maximum_absolute_entity_mean_after_demeaning": max_abs_group_mean,
          "within_minus_dummy_coefficient": within.params["cash_flow_within"] - dummy.params[x_name],
          "within_minus_manual_coefficient": within.params["cash_flow_within"] - manual_beta}]
    )
    diagnostics.to_csv(RESULTS / "within_diagnostics.csv", index=False)
    print(coefficients.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nDiagnostik:")
    print(diagnostics.to_string(index=False, float_format=lambda value: f"{value:.3e}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(data[x_name], data[y_name], color="#9DC3E6", alpha=0.4, s=18)
    raw_grid = np.linspace(data[x_name].min(), data[x_name].max(), 200)
    axes[0, 0].plot(raw_grid, pooled.params["const"] + pooled.params[x_name] * raw_grid,
                    color="#C00000", linewidth=2, label=f"Pooled β={pooled.params[x_name]:.2f}")
    axes[0, 0].set(title="Rohdaten enthalten Between-Unterschiede", xlabel="Cashflow",
                   ylabel="Investition")
    axes[0, 0].legend(frameon=False)

    within_grid = np.linspace(data["cash_flow_within"].min(), data["cash_flow_within"].max(), 200)
    axes[0, 1].scatter(data["cash_flow_within"], data["investment_within"],
                       color="#9DC3E6", alpha=0.4, s=18)
    axes[0, 1].plot(within_grid, manual_beta * within_grid, color="#C00000", linewidth=2,
                    label=f"Within β={manual_beta:.2f}")
    axes[0, 1].axhline(0, color="#595959", linewidth=1)
    axes[0, 1].axvline(0, color="#595959", linewidth=1)
    axes[0, 1].set(title="Demeaning isoliert Firmenabweichungen", xlabel="Cashflow − Firmenmittel",
                   ylabel="Investition − Firmenmittel")
    axes[0, 1].legend(frameon=False)

    selected_firms = [2, 11, 24, 39, 55, 71]
    for firm_id in selected_firms:
        subset = data[data["firm_id"] == firm_id]
        axes[1, 0].plot(subset["period"], subset[x_name], marker="o", alpha=0.75,
                        label=f"Firma {firm_id}")
        axes[1, 0].axhline(subset[x_name].mean(), color="#A5A5A5", linewidth=0.7, alpha=0.5)
    axes[1, 0].set(title="Jede Firma wird um ihr eigenes Mittel zentriert", xlabel="Periode",
                   ylabel="Cashflow", xticks=np.arange(1, 9))
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)

    plot_coefficients = coefficients.dropna(subset=["standard_error"])
    positions = np.arange(len(plot_coefficients))
    axes[1, 1].errorbar(plot_coefficients["coefficient"], positions,
                        xerr=1.96 * plot_coefficients["standard_error"], fmt="o",
                        color="#4472C4", capsize=5)
    axes[1, 1].scatter([manual_beta], [len(plot_coefficients)], marker="x", color="#70AD47",
                       s=70, label="Handrechnung")
    axes[1, 1].axvline(TRUE_BETA, color="#C00000", linestyle="--", label="Wahrer Effekt")
    axes[1, 1].set(title="Within, Dummies und Handrechnung stimmen überein",
                   xlabel="Cashflow-Koeffizient", yticks=np.arange(len(coefficients)),
                   yticklabels=coefficients["method"])
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Der Within-Estimator entfernt Firmenmittel vor der Regression")
    figure.tight_layout()
    figure.savefig(RESULTS / "within_estimator.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
