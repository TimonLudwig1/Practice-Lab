"""Musterlösung für T45: nicht-binäre Behandlungsintensität."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260725


def true_dose_effect(dose: np.ndarray | float) -> np.ndarray | float:
    return 7.0 * (1 - np.exp(-np.asarray(dose) / 35.0))


def make_data(cities: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    capacity = rng.normal(size=cities)
    dose = np.zeros(cities)
    funded = np.argsort(capacity)[30:]
    dose[funded] = np.clip(55 + 18 * capacity[funded] + rng.normal(0, 10, len(funded)), 10, 100)
    city_intercept = 48 + 5 * capacity + rng.normal(0, 1.3, cities)
    time_shocks = np.array([0.0, 1.2, -0.8, 2.0, 0.5, 1.6])
    rows = []
    for city in range(cities):
        for year_index, year in enumerate(range(2019, 2025)):
            post = int(year >= 2022)
            effect = float(true_dose_effect(dose[city])) * post
            outcome = city_intercept[city] + time_shocks[year_index] + effect + rng.normal(0, 1.1)
            rows.append((city + 1, year, post, dose[city], outcome, capacity[city], effect))
    return pd.DataFrame(rows, columns=["city_id", "year", "post", "funding_eur_per_resident",
                                       "investment_index", "oracle_capacity", "oracle_treatment_effect"])


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    dose = data["funding_eur_per_resident"]
    data["dose_10"] = dose / 10
    data["dose_10_post"] = data["dose_10"] * data["post"]
    data["dose_10_sq_post"] = data["dose_10"] ** 2 * data["post"]
    city_dose = data.groupby("city_id")["funding_eur_per_resident"].first()
    bins = [-0.01, 0.01, 40, 70, np.inf]
    labels = ["control", "low", "medium", "high"]
    intensity = pd.cut(city_dose, bins=bins, labels=labels)
    data["intensity_group"] = data["city_id"].map(intensity.astype(str))
    for group in ["low", "medium", "high"]:
        data[f"{group}_post"] = (data["intensity_group"].eq(group) & data["post"].eq(1)).astype(int)
    data.to_csv(DATA / "municipal_funding_panel.csv", index=False)

    cluster = {"groups": data["city_id"]}
    linear = smf.ols(
        "investment_index ~ dose_10_post + C(city_id) + C(year)", data=data
    ).fit(cov_type="cluster", cov_kwds=cluster)
    quadratic = smf.ols(
        "investment_index ~ dose_10_post + dose_10_sq_post + C(city_id) + C(year)", data=data
    ).fit(cov_type="cluster", cov_kwds=cluster)
    categorical = smf.ols(
        "investment_index ~ low_post + medium_post + high_post + C(city_id) + C(year)", data=data
    ).fit(cov_type="cluster", cov_kwds=cluster)
    assert quadratic.params["dose_10_sq_post"] < 0
    assert quadratic.pvalues["dose_10_sq_post"] < 0.05
    assert np.sqrt(np.mean(quadratic.resid ** 2)) < np.sqrt(np.mean(linear.resid ** 2))

    model_comparison = pd.DataFrame([
        {"model": "Linear dose", "rmse": np.sqrt(np.mean(linear.resid ** 2)),
         "aic_nonclustered_definition": linear.aic,
         "linear_effect_per_10_eur": linear.params["dose_10_post"],
         "quadratic_term": 0.0, "quadratic_p_value": np.nan},
        {"model": "Quadratic dose", "rmse": np.sqrt(np.mean(quadratic.resid ** 2)),
         "aic_nonclustered_definition": quadratic.aic,
         "linear_effect_per_10_eur": quadratic.params["dose_10_post"],
         "quadratic_term": quadratic.params["dose_10_sq_post"],
         "quadratic_p_value": quadratic.pvalues["dose_10_sq_post"]},
        {"model": "Categorical dose", "rmse": np.sqrt(np.mean(categorical.resid ** 2)),
         "aic_nonclustered_definition": categorical.aic,
         "linear_effect_per_10_eur": np.nan, "quadratic_term": np.nan,
         "quadratic_p_value": np.nan},
    ])
    model_comparison.to_csv(RESULTS / "dose_model_comparison.csv", index=False)

    requested_doses = np.array([20.0, 50.0, 80.0])
    x = requested_doses / 10
    response = pd.DataFrame({
        "dose_eur_per_resident": requested_doses,
        "linear_predicted_total_effect": linear.params["dose_10_post"] * x,
        "quadratic_predicted_total_effect": (
            quadratic.params["dose_10_post"] * x + quadratic.params["dose_10_sq_post"] * x ** 2
        ),
        "quadratic_marginal_effect_per_10_eur": (
            quadratic.params["dose_10_post"] + 2 * quadratic.params["dose_10_sq_post"] * x
        ),
        "oracle_total_effect": true_dose_effect(requested_doses),
    })
    response.to_csv(RESULTS / "dose_effects_at_selected_values.csv", index=False)

    category_rows = []
    city_information = data.groupby("city_id", as_index=False).first()
    for group in ["low", "medium", "high"]:
        subset = city_information.loc[city_information["intensity_group"].eq(group)]
        category_rows.append({
            "intensity_group": group,
            "mean_dose": subset["funding_eur_per_resident"].mean(),
            "estimated_post_effect": categorical.params[f"{group}_post"],
            "cluster_standard_error": categorical.bse[f"{group}_post"],
            "oracle_mean_effect": subset["funding_eur_per_resident"].pipe(true_dose_effect).mean(),
        })
    category_effects = pd.DataFrame(category_rows)
    category_effects.to_csv(RESULTS / "categorical_intensity_effects.csv", index=False)

    pre_post = data.groupby(["city_id", "post"], as_index=False)["investment_index"].mean()
    changes = pre_post.pivot(index="city_id", columns="post", values="investment_index")
    city_change = pd.DataFrame({"city_id": changes.index, "change": changes[1] - changes[0]}).reset_index(drop=True)
    city_change = city_change.merge(city_information[["city_id", "funding_eur_per_resident"]], on="city_id")
    control_change = city_change.loc[city_change["funding_eur_per_resident"].eq(0), "change"].mean()
    city_change["control_adjusted_change"] = city_change["change"] - control_change
    city_change.to_csv(RESULTS / "city_level_changes.csv", index=False)

    grid = np.linspace(0, 100, 101)
    grid_x = grid / 10
    response_grid = pd.DataFrame({
        "dose_eur_per_resident": grid,
        "linear_effect": linear.params["dose_10_post"] * grid_x,
        "quadratic_effect": (quadratic.params["dose_10_post"] * grid_x
                             + quadratic.params["dose_10_sq_post"] * grid_x ** 2),
        "oracle_effect": true_dose_effect(grid),
    })
    response_grid.to_csv(RESULTS / "dose_response_grid.csv", index=False)

    print(model_comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nEffekte an ausgewählten Dosiswerten:")
    print(response.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nKategoriale Intensitätseffekte:")
    print(category_effects.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(city_change["funding_eur_per_resident"], city_change["control_adjusted_change"],
                    color="#9DC3E6", alpha=0.65, s=28, label="Kommunen")
    axes[0].plot(grid, response_grid["oracle_effect"], color="#595959", linewidth=2,
                 label="Wahre Dosiswirkung")
    axes[0].plot(grid, response_grid["linear_effect"], color="#C00000", linestyle="--",
                 label="Lineares Modell")
    axes[0].plot(grid, response_grid["quadratic_effect"], color="#4472C4", linewidth=2,
                 label="Quadratisches Modell")
    axes[0].set(title="Konstante Grenzwirkung ist zu restriktiv",
                xlabel="Förderung (€ je Einwohner)", ylabel="Kontrollbereinigte Indexänderung")
    axes[0].legend(frameon=False)

    y_pos = np.arange(len(category_effects))
    axes[1].errorbar(category_effects["estimated_post_effect"], y_pos,
                     xerr=1.96 * category_effects["cluster_standard_error"], fmt="o",
                     color="#4472C4", capsize=5, label="Geschätzt")
    axes[1].scatter(category_effects["oracle_mean_effect"], y_pos, marker="x",
                    color="#C00000", s=70, label="Wahrer Gruppenmittelwert")
    axes[1].set(title="Kategorien lockern die lineare Annahme",
                xlabel="Post-Effekt auf den Investitionsindex", yticks=y_pos,
                yticklabels=["Niedrige Dosis", "Mittlere Dosis", "Hohe Dosis"])
    axes[1].legend(frameon=False)
    figure.suptitle("Nicht-binäre Behandlung: Die Wirkung hängt von der Dosis ab")
    figure.tight_layout()
    figure.savefig(RESULTS / "non_binary_treatment.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
