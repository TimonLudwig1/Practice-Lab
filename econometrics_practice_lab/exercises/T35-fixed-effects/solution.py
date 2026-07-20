"""Musterlösung für T35: Fixed Effects."""

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
TRUE_BETA = 2.5


def make_data(stores: int = 100, periods: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    quality = rng.normal(0, 8, stores)
    downtown = (quality + rng.normal(0, 5, stores) > 1).astype(int)
    rows = []
    for store in range(stores):
        for month in range(1, periods + 1):
            advertising = 12 + 0.55 * quality[store] + 0.18 * month + rng.normal(0, 2.2)
            sales = 100 + quality[store] + TRUE_BETA * advertising + rng.normal(0, 5)
            rows.append((store + 1, month, advertising, sales, quality[store], downtown[store]))
    return pd.DataFrame(rows, columns=["store_id", "month", "advertising_thousand_eur",
                                       "sales_index", "oracle_store_quality", "downtown_location"])


def fit_clustered(y: pd.Series, design: pd.DataFrame, groups: pd.Series):
    return sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": groups})


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "store_panel.csv", index=False)
    x_name = "advertising_thousand_eur"
    pooled = fit_clustered(data["sales_index"], sm.add_constant(data[[x_name]]), data["store_id"])
    store_means = data.groupby("store_id")[[x_name, "sales_index", "oracle_store_quality"]].mean()
    between = sm.OLS(store_means["sales_index"], sm.add_constant(store_means[[x_name]])).fit()
    store_dummies = pd.get_dummies(data["store_id"], prefix="store", drop_first=True, dtype=float)
    fe_design = sm.add_constant(pd.concat([data[[x_name]], store_dummies], axis=1))
    fixed_effects = fit_clustered(data["sales_index"], fe_design, data["store_id"])
    oracle = fit_clustered(
        data["sales_index"], sm.add_constant(data[[x_name, "oracle_store_quality"]]), data["store_id"]
    )
    rows = []
    for name, model in (("Pooled OLS", pooled), ("Between", between),
                        ("Store fixed effects", fixed_effects), ("Oracle quality control", oracle)):
        rows.append({"model": name, "advertising_coefficient": model.params[x_name],
                     "standard_error": model.bse[x_name], "ci_95_low": model.conf_int().loc[x_name, 0],
                     "ci_95_high": model.conf_int().loc[x_name, 1], "true_coefficient": TRUE_BETA})
    comparison = pd.DataFrame(rows)
    assert abs(fixed_effects.params[x_name] - TRUE_BETA) < 0.25
    assert abs(fixed_effects.params[x_name] - TRUE_BETA) < abs(pooled.params[x_name] - TRUE_BETA)
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

    data["advertising_within"] = data[x_name] - data.groupby("store_id")[x_name].transform("mean")
    data["sales_within"] = data["sales_index"] - data.groupby("store_id")["sales_index"].transform("mean")
    within_model = sm.OLS(data["sales_within"], data[["advertising_within"]]).fit()
    assert np.isclose(within_model.params["advertising_within"], fixed_effects.params[x_name])
    time_invariant_design = pd.concat(
        [pd.Series(1.0, index=data.index, name="const"), store_dummies,
         data[["downtown_location"]]], axis=1
    )
    rank_table = pd.DataFrame(
        [{"columns": time_invariant_design.shape[1],
          "rank": np.linalg.matrix_rank(time_invariant_design.to_numpy()),
          "rank_deficient": np.linalg.matrix_rank(time_invariant_design.to_numpy()) < time_invariant_design.shape[1],
          "correlation_quality_mean_advertising": store_means[x_name].corr(store_means["oracle_store_quality"])}]
    )
    assert bool(rank_table.loc[0, "rank_deficient"])
    rank_table.to_csv(RESULTS / "identification_diagnostics.csv", index=False)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nIdentifikation:")
    print(rank_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(data[x_name], data["sales_index"], color="#9DC3E6", alpha=0.35, s=18)
    raw_grid = np.linspace(data[x_name].min(), data[x_name].max(), 200)
    axes[0, 0].plot(raw_grid, pooled.params["const"] + pooled.params[x_name] * raw_grid,
                    color="#C00000", linewidth=2, label=f"Pooled β={pooled.params[x_name]:.2f}")
    axes[0, 0].set(title="Pooled vermischt Within und Between", xlabel="Werbung (Tsd. €)",
                   ylabel="Umsatzindex")
    axes[0, 0].legend(frameon=False)

    within_grid = np.linspace(data["advertising_within"].min(), data["advertising_within"].max(), 200)
    axes[0, 1].scatter(data["advertising_within"], data["sales_within"], color="#9DC3E6",
                       alpha=0.35, s=18)
    axes[0, 1].plot(within_grid, fixed_effects.params[x_name] * within_grid,
                    color="#C00000", linewidth=2, label=f"FE β={fixed_effects.params[x_name]:.2f}")
    axes[0, 1].axhline(0, color="#595959", linewidth=1)
    axes[0, 1].axvline(0, color="#595959", linewidth=1)
    axes[0, 1].set(title="FE nutzt Abweichungen vom Filialmittel", xlabel="Werbung innerhalb Filiale",
                   ylabel="Umsatz innerhalb Filiale")
    axes[0, 1].legend(frameon=False)

    positions = np.arange(len(comparison))
    estimates = comparison["advertising_coefficient"].to_numpy()
    left = estimates - comparison["ci_95_low"].to_numpy()
    right = comparison["ci_95_high"].to_numpy() - estimates
    axes[1, 0].errorbar(estimates, positions, xerr=np.vstack([left, right]), fmt="o",
                        color="#4472C4", capsize=5)
    axes[1, 0].axvline(TRUE_BETA, color="#C00000", linestyle="--", label="Wahrer Effekt")
    axes[1, 0].set(title="Zeitinvariante Qualität verzerrt Pooled OLS",
                   xlabel="Werbekoeffizient mit 95%-KI", yticks=positions,
                   yticklabels=comparison["model"])
    axes[1, 0].legend(frameon=False)

    axes[1, 1].scatter(store_means[x_name], store_means["oracle_store_quality"],
                       color="#4472C4", alpha=0.65, s=28)
    axes[1, 1].set(title=f"Qualität korreliert mit mittlerer Werbung (r={rank_table.loc[0, 'correlation_quality_mean_advertising']:.2f})",
                   xlabel="Mittlere Werbung je Filiale", ylabel="Zeitinvariante Filialqualität")
    figure.suptitle("Fixed Effects entfernen zeitinvariante Unterschiede zwischen Filialen")
    figure.tight_layout()
    figure.savefig(RESULTS / "fixed_effects.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
