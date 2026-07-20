"""Musterlösung für T18: Multiple Regression und ceteris paribus."""

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
REGRESSORS = ["area_sqm", "quality_score", "distance_center_km", "building_age_years"]


def make_data(size: int = 350) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    area = rng.uniform(30, 150, size=size)
    distance = rng.uniform(0.5, 20, size=size)
    quality = np.clip(7.2 - 0.16 * distance + rng.normal(0, 1.2, size=size), 1, 10)
    age = rng.uniform(0, 80, size=size)
    rent = 350 + 10.5 * area + 95 * quality - 18 * distance - 2.5 * age
    rent += rng.normal(0, 160, size=size)
    return pd.DataFrame(
        {"apartment_id": np.arange(1, size + 1), "area_sqm": area,
         "quality_score": quality, "distance_center_km": distance,
         "building_age_years": age, "rent_eur": rent}
    )


def comparison_row(name: str, model) -> dict[str, float | str]:
    return {"model": name, "r_squared": model.rsquared,
            "adjusted_r_squared": model.rsquared_adj,
            "rmse": np.sqrt(np.mean(model.resid ** 2))}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "apartment_characteristics.csv", index=False)

    simple = sm.OLS(data["rent_eur"], sm.add_constant(data[["area_sqm"]])).fit()
    multiple = sm.OLS(data["rent_eur"], sm.add_constant(data[REGRESSORS])).fit()
    ci = multiple.conf_int()
    coefficients = pd.DataFrame(
        {"term": multiple.params.index, "estimate": multiple.params.values,
         "standard_error": multiple.bse.values, "p_value": multiple.pvalues.values,
         "ci_95_low": ci[0].values, "ci_95_high": ci[1].values}
    )

    controls = ["quality_score", "distance_center_km", "building_age_years"]
    area_residual = sm.OLS(data["area_sqm"], sm.add_constant(data[controls])).fit().resid
    rent_residual = sm.OLS(data["rent_eur"], sm.add_constant(data[controls])).fit().resid
    fwl = sm.OLS(rent_residual, area_residual).fit()
    assert np.isclose(fwl.params.iloc[0], multiple.params["area_sqm"])
    coefficients["fwl_area_estimate"] = np.where(
        coefficients["term"] == "area_sqm", fwl.params.iloc[0], np.nan
    )

    standardized = (
        data[REGRESSORS + ["rent_eur"]] - data[REGRESSORS + ["rent_eur"]].mean()
    ) / data[REGRESSORS + ["rent_eur"]].std(ddof=1)
    standardized_model = sm.OLS(standardized["rent_eur"],
                                sm.add_constant(standardized[REGRESSORS])).fit()
    coefficients["standardized_estimate"] = coefficients["term"].map(
        standardized_model.params.to_dict()
    )
    coefficients.to_csv(RESULTS / "coefficients.csv", index=False)

    comparison = pd.DataFrame(
        [comparison_row("Simple: area", simple), comparison_row("Multiple", multiple)]
    )
    comparison["area_coefficient"] = [simple.params["area_sqm"], multiple.params["area_sqm"]]
    assert multiple.rsquared > simple.rsquared
    assert np.sqrt(np.mean(multiple.resid ** 2)) < np.sqrt(np.mean(simple.resid ** 2))
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMultiple Koeffizienten:")
    print(coefficients.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(multiple.fittedvalues, data["rent_eur"], color="#5B9BD5",
                       alpha=0.6, s=23)
    limits = [min(multiple.fittedvalues.min(), data["rent_eur"].min()),
              max(multiple.fittedvalues.max(), data["rent_eur"].max())]
    axes[0, 0].plot(limits, limits, color="#C00000", linestyle="--")
    axes[0, 0].set(title=f"Multipler Fit (R²={multiple.rsquared:.3f})",
                   xlabel="Vorhergesagte Miete (€)", ylabel="Beobachtete Miete (€)")

    order = np.argsort(area_residual.to_numpy())
    axes[0, 1].scatter(area_residual, rent_residual, color="#5B9BD5", alpha=0.6, s=23)
    axes[0, 1].plot(area_residual.to_numpy()[order], fwl.fittedvalues.to_numpy()[order],
                    color="#C00000", linewidth=2,
                    label=f"FWL-Steigung = {fwl.params.iloc[0]:.2f}")
    axes[0, 1].set(title="Partieller Zusammenhang der Wohnfläche",
                   xlabel="Wohnfläche bereinigt um Kontrollen",
                   ylabel="Miete bereinigt um Kontrollen")
    axes[0, 1].legend(frameon=False)

    standardized_terms = coefficients[coefficients["term"] != "const"]
    colors = np.where(standardized_terms["standardized_estimate"] >= 0, "#4472C4", "#ED7D31")
    axes[1, 0].barh(standardized_terms["term"], standardized_terms["standardized_estimate"],
                    color=colors)
    axes[1, 0].axvline(0, color="#595959", linewidth=1)
    axes[1, 0].set(title="Standardisierte ceteris-paribus-Koeffizienten",
                   xlabel="Standardisierte Steigung")

    axes[1, 1].scatter(multiple.fittedvalues, multiple.resid, color="#4472C4",
                       alpha=0.6, s=23)
    axes[1, 1].axhline(0, color="#C00000", linestyle="--")
    axes[1, 1].set(title="Residuen gegen Fits", xlabel="Vorhergesagte Miete (€)",
                   ylabel="Residuum (€)")
    figure.suptitle("Multiple Regression isoliert partielle lineare Zusammenhänge")
    figure.tight_layout()
    figure.savefig(RESULTS / "multiple_regression.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
