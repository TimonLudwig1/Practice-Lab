"""Musterlösung für T22: Kategoriale Variablen und Referenzkategorien."""

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
LOCATIONS = ["Center", "Suburb", "University", "Industrial"]


def make_data(size: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    location = rng.choice(LOCATIONS, size=size, p=[0.25, 0.35, 0.25, 0.15])
    marketing = rng.uniform(5, 40, size=size)
    effects = {"Center": 120, "Suburb": 0, "University": 70, "Industrial": -90}
    revenue = 500 + 18 * marketing + np.array([effects[value] for value in location])
    revenue += rng.normal(0, 90, size=size)
    return pd.DataFrame(
        {"store_id": np.arange(1, size + 1), "location": location,
         "marketing_thousand_eur": marketing, "weekly_revenue_thousand_eur": revenue}
    )


def design_matrix(data: pd.DataFrame, reference: str) -> pd.DataFrame:
    categories = [reference] + [value for value in LOCATIONS if value != reference]
    categorical = pd.Categorical(data["location"], categories=categories)
    dummies = pd.get_dummies(categorical, prefix="location", drop_first=True, dtype=float)
    design = pd.DataFrame(
        {"marketing_centered": data["marketing_thousand_eur"].to_numpy() - 20},
        index=data.index,
    )
    return sm.add_constant(pd.concat([design, dummies.set_axis(data.index)], axis=1))


def coefficient_rows(reference: str, model) -> list[dict[str, float | str]]:
    ci = model.conf_int()
    return [
        {"reference": reference, "term": term, "estimate": model.params[term],
         "standard_error": model.bse[term], "ci_95_low": ci.loc[term, 0],
         "ci_95_high": ci.loc[term, 1]}
        for term in model.params.index
    ]


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "store_locations.csv", index=False)
    models = {}
    designs = {}
    rows = []
    for reference in ("Suburb", "Center"):
        designs[reference] = design_matrix(data, reference)
        models[reference] = sm.OLS(
            data["weekly_revenue_thousand_eur"], designs[reference]
        ).fit()
        rows.extend(coefficient_rows(reference, models[reference]))
    coefficients = pd.DataFrame(rows)
    coefficients.to_csv(RESULTS / "coefficients_by_reference.csv", index=False)
    assert np.allclose(models["Suburb"].fittedvalues, models["Center"].fittedvalues)

    prediction_rows = []
    for location in LOCATIONS:
        new_data = pd.DataFrame(
            {"location": [location], "marketing_thousand_eur": [20.0]}
        )
        design = design_matrix(new_data, "Suburb")
        design = design.reindex(columns=designs["Suburb"].columns, fill_value=0.0)
        design["const"] = 1.0
        prediction = models["Suburb"].get_prediction(design).summary_frame().iloc[0]
        prediction_rows.append(
            {"location": location, "marketing_thousand_eur": 20.0,
             "adjusted_mean": prediction["mean"], "ci_95_low": prediction["mean_ci_lower"],
             "ci_95_high": prediction["mean_ci_upper"]}
        )
    adjusted = pd.DataFrame(prediction_rows)
    adjusted.to_csv(RESULTS / "adjusted_location_means.csv", index=False)
    print(coefficients.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nAdjustierte Standortmittel:")
    print(adjusted.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    raw_groups = [data.loc[data["location"] == location, "weekly_revenue_thousand_eur"]
                  for location in LOCATIONS]
    axes[0, 0].boxplot(raw_groups, tick_labels=LOCATIONS, patch_artist=True,
                       boxprops={"facecolor": "#9DC3E6"}, medianprops={"color": "#C00000"})
    axes[0, 0].set(title="Rohverteilungen nach Standort", ylabel="Wochenumsatz (Tsd. €)")

    positions = np.arange(len(adjusted))
    values = adjusted["adjusted_mean"].to_numpy()
    left = values - adjusted["ci_95_low"].to_numpy()
    right = adjusted["ci_95_high"].to_numpy() - values
    axes[0, 1].errorbar(values, positions, xerr=np.vstack([left, right]), fmt="o",
                        color="#4472C4", capsize=5)
    axes[0, 1].set(title="Adjustiert auf 20 Tsd. € Marketing", xlabel="Erwarteter Wochenumsatz",
                   yticks=positions, yticklabels=adjusted["location"])

    for axis, reference in zip(axes[1], ("Suburb", "Center")):
        subset = coefficients[(coefficients["reference"] == reference) &
                              coefficients["term"].str.startswith("location_")]
        estimates = subset["estimate"].to_numpy()
        positions = np.arange(len(subset))
        left = estimates - subset["ci_95_low"].to_numpy()
        right = subset["ci_95_high"].to_numpy() - estimates
        axis.errorbar(estimates, positions, xerr=np.vstack([left, right]), fmt="o",
                      color="#4472C4", capsize=5)
        axis.axvline(0, color="#C00000", linestyle="--")
        axis.set(title=f"Referenz: {reference}", xlabel="Differenz zur Referenz",
                 yticks=positions, yticklabels=subset["term"].str.replace("location_", ""))
    figure.suptitle("Referenzkategorien ändern Koeffizienten, nicht Vorhersagen")
    figure.tight_layout()
    figure.savefig(RESULTS / "categorical_variables.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
