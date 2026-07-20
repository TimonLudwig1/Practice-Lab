"""Musterlösung für T20: Multikollinearität diagnostizieren."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720
REGRESSORS = ["insulation_index", "thermal_score"]


def make_data(size: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    latent_efficiency = rng.normal(size=size)
    insulation = 50 + 10 * latent_efficiency + rng.normal(0, 1.2, size=size)
    thermal = 50 + 10 * latent_efficiency + rng.normal(0, 1.2, size=size)
    energy_cost = 4_000 - 25 * insulation - 25 * thermal + rng.normal(0, 250, size=size)
    return pd.DataFrame(
        {"building_id": np.arange(1, size + 1), "insulation_index": insulation,
         "thermal_score": thermal, "annual_energy_cost_eur": energy_cost}
    )


def fit(data: pd.DataFrame, terms: list[str]):
    return sm.OLS(data["annual_energy_cost_eur"], sm.add_constant(data[terms])).fit()


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "building_efficiency.csv", index=False)
    models = {
        "Insulation only": fit(data, ["insulation_index"]),
        "Thermal only": fit(data, ["thermal_score"]),
        "Both": fit(data, REGRESSORS),
    }

    rows = []
    for model_name, model in models.items():
        for term in REGRESSORS:
            if term in model.params:
                rows.append(
                    {"model": model_name, "term": term, "estimate": model.params[term],
                     "standard_error": model.bse[term], "p_value": model.pvalues[term],
                     "r_squared": model.rsquared, "rmse": np.sqrt(np.mean(model.resid ** 2))}
                )
    coefficients = pd.DataFrame(rows)
    coefficients.to_csv(RESULTS / "model_coefficients.csv", index=False)

    design = sm.add_constant(data[REGRESSORS])
    vif = pd.DataFrame(
        {"term": design.columns,
         "vif": [variance_inflation_factor(design.to_numpy(), index)
                 for index in range(design.shape[1])]}
    )
    vif.to_csv(RESULTS / "vif.csv", index=False)

    rng = np.random.default_rng(RNG_SEED + 1)
    bootstrap_rows = []
    for repetition in range(600):
        indices = rng.integers(0, len(data), size=len(data))
        sample = data.iloc[indices]
        model = fit(sample, REGRESSORS)
        beta_insulation = model.params["insulation_index"]
        beta_thermal = model.params["thermal_score"]
        bootstrap_rows.append(
            {"repetition": repetition + 1, "beta_insulation": beta_insulation,
             "beta_thermal": beta_thermal, "beta_sum": beta_insulation + beta_thermal}
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(RESULTS / "bootstrap_draws.csv", index=False)
    bootstrap_summary = (
        bootstrap[["beta_insulation", "beta_thermal", "beta_sum"]]
        .agg(["mean", "std", lambda values: values.quantile(0.025),
              lambda values: values.quantile(0.975)])
        .T.reset_index()
    )
    bootstrap_summary.columns = ["quantity", "mean", "sd", "q025", "q975"]
    regressor_vifs = vif.loc[vif["term"].isin(REGRESSORS), "vif"]
    bootstrap_sd = bootstrap_summary.set_index("quantity")["sd"]
    assert (regressor_vifs > 10).all()
    assert bootstrap_sd["beta_sum"] < bootstrap_sd[["beta_insulation", "beta_thermal"]].min()
    bootstrap_summary.to_csv(RESULTS / "bootstrap_summary.csv", index=False)
    print(f"Korrelation der Regressoren: {data[REGRESSORS].corr().iloc[0, 1]:.4f}")
    print(vif.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nBootstrap:")
    print(bootstrap_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    correlation = data[REGRESSORS].corr().iloc[0, 1]
    axes[0, 0].scatter(data["insulation_index"], data["thermal_score"], color="#5B9BD5",
                       alpha=0.6, s=23)
    axes[0, 0].set(title=f"Fast redundante Regressoren (r={correlation:.3f})",
                   xlabel="Isolationsindex", ylabel="Thermischer Score")

    for index, (model_name, group) in enumerate(coefficients.groupby("model", sort=False)):
        axes[0, 1].scatter(group["estimate"], [index] * len(group),
                           label=model_name, s=58)
    axes[0, 1].axvline(-25, color="#C00000", linestyle="--", label="Wahre Einzeleffekte")
    axes[0, 1].set(title="Koeffizienten hängen von der Spezifikation ab",
                   xlabel="Geschätzter Koeffizient", yticks=[])
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].hist(bootstrap["beta_insulation"], bins=35, alpha=0.62, color="#4472C4",
                    label="β Isolierung")
    axes[1, 0].hist(bootstrap["beta_thermal"], bins=35, alpha=0.62, color="#ED7D31",
                    label="β Thermik")
    axes[1, 0].axvline(-25, color="#595959", linestyle="--")
    axes[1, 0].set(title="Instabile Einzelkoeffizienten", xlabel="Bootstrap-Koeffizient",
                   ylabel="Häufigkeit")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].hist(bootstrap["beta_sum"], bins=35, color="#70AD47", edgecolor="white")
    axes[1, 1].axvline(-50, color="#C00000", linestyle="--", label="Wahre Summe")
    axes[1, 1].set(title="Stabilerer gemeinsamer Zusammenhang", xlabel="β Isolierung + β Thermik",
                   ylabel="Häufigkeit")
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Multikollinearität erschwert die Trennung einzelner Effekte")
    figure.tight_layout()
    figure.savefig(RESULTS / "multicollinearity.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
