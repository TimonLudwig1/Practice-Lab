"""Musterlösung für T28: Marginale Effekte im Logit-Modell."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720
REGRESSORS = ["preparation_hours", "experience_years", "mentor_program"]


def make_data(size: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    preparation = rng.uniform(0, 60, size=size)
    experience = rng.uniform(0, 12, size=size)
    mentor = rng.binomial(1, 0.42, size=size)
    linear_index = -5.1 + 0.095 * preparation + 0.17 * experience + 0.85 * mentor
    probability = expit(linear_index)
    offer = rng.binomial(1, probability)
    return pd.DataFrame(
        {"applicant_id": np.arange(1, size + 1), "preparation_hours": preparation,
         "experience_years": experience, "mentor_program": mentor,
         "received_offer": offer, "oracle_offer_probability": probability}
    )


def marginal_rows(result, method: str, terms: list[str]) -> list[dict[str, float | str]]:
    intervals = result.conf_int()
    return [
        {"method": method, "term": term, "marginal_effect": result.margeff[index],
         "standard_error": result.margeff_se[index], "ci_95_low": intervals[index, 0],
         "ci_95_high": intervals[index, 1]}
        for index, term in enumerate(terms)
    ]


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "job_offers.csv", index=False)
    design = sm.add_constant(data[REGRESSORS])
    model = sm.Logit(data["received_offer"], design).fit(disp=False)
    ci = model.conf_int()
    coefficients = pd.DataFrame(
        {"term": model.params.index, "estimate": model.params.values,
         "standard_error": model.bse.values, "p_value": model.pvalues.values,
         "ci_95_low": ci[0].values, "ci_95_high": ci[1].values}
    )
    coefficients.to_csv(RESULTS / "coefficients.csv", index=False)

    predicted = model.predict(design)
    continuous_effect = model.params["preparation_hours"] * predicted * (1 - predicted)
    design_mentor_one = design.copy()
    design_mentor_zero = design.copy()
    design_mentor_one["mentor_program"] = 1.0
    design_mentor_zero["mentor_program"] = 0.0
    mentor_effect = model.predict(design_mentor_one) - model.predict(design_mentor_zero)
    individual = pd.DataFrame(
        {"applicant_id": data["applicant_id"], "predicted_probability": predicted,
         "preparation_marginal_effect": continuous_effect,
         "mentor_discrete_effect": mentor_effect}
    )
    individual.to_csv(RESULTS / "individual_marginal_effects.csv", index=False)

    ame_result = model.get_margeff(at="overall", method="dydx", dummy=True)
    mem_result = model.get_margeff(at="mean", method="dydx", dummy=True)
    marginal_summary = pd.DataFrame(
        marginal_rows(ame_result, "AME", REGRESSORS)
        + marginal_rows(mem_result, "MEM", REGRESSORS)
    )
    manual_effects = {
        ("AME", "preparation_hours"): continuous_effect.mean(),
        ("AME", "mentor_program"): mentor_effect.mean(),
    }
    mean_preparation = data["preparation_hours"].mean()
    mean_experience = data["experience_years"].mean()
    mean_mentor = data["mentor_program"].mean()
    mean_index = (
        model.params["const"] + model.params["preparation_hours"] * mean_preparation
        + model.params["experience_years"] * mean_experience
        + model.params["mentor_program"] * mean_mentor
    )
    mean_probability = expit(mean_index)
    manual_effects[("MEM", "preparation_hours")] = (
        model.params["preparation_hours"] * mean_probability * (1 - mean_probability)
    )
    mean_base_index = (
        model.params["const"] + model.params["preparation_hours"] * mean_preparation
        + model.params["experience_years"] * mean_experience
    )
    manual_effects[("MEM", "mentor_program")] = (
        expit(mean_base_index + model.params["mentor_program"]) - expit(mean_base_index)
    )
    marginal_summary["manual_check"] = [
        manual_effects.get((row.method, row.term), np.nan)
        for row in marginal_summary.itertuples(index=False)
    ]
    checked = marginal_summary.dropna(subset=["manual_check"])
    assert np.allclose(checked["marginal_effect"], checked["manual_check"])
    assert (individual["preparation_marginal_effect"] > 0).all()
    assert (individual["mentor_discrete_effect"] > 0).all()
    marginal_summary.to_csv(RESULTS / "marginal_effect_summary.csv", index=False)

    print(coefficients.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nMarginale Effekte:")
    print(marginal_summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    preparation_grid = np.linspace(0, 60, 240)
    colors = {0: "#4472C4", 1: "#ED7D31"}
    labels = {0: "Ohne Mentoring", 1: "Mit Mentoring"}
    for mentor_value in (0, 1):
        grid_index = (
            model.params["const"] + model.params["preparation_hours"] * preparation_grid
            + model.params["experience_years"] * mean_experience
            + model.params["mentor_program"] * mentor_value
        )
        grid_probability = expit(grid_index)
        grid_effect = (
            model.params["preparation_hours"] * grid_probability * (1 - grid_probability)
        )
        axes[0, 0].plot(preparation_grid, grid_probability, color=colors[mentor_value],
                        linewidth=2, label=labels[mentor_value])
        axes[0, 1].plot(preparation_grid, grid_effect, color=colors[mentor_value],
                        linewidth=2, label=labels[mentor_value])
    axes[0, 0].set(title="Vorhergesagte Angebotswahrscheinlichkeit",
                   xlabel="Vorbereitungsstunden", ylabel="Wahrscheinlichkeit", ylim=(0, 1))
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(title="Effekt einer zusätzlichen Vorbereitungsstunde",
                   xlabel="Vorbereitungsstunden", ylabel="Wahrscheinlichkeitsänderung")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].scatter(individual["predicted_probability"],
                       individual["preparation_marginal_effect"], color="#4472C4",
                       alpha=0.35, s=18, label="Individuelle Effekte")
    probability_grid = np.linspace(0, 1, 240)
    axes[1, 0].plot(probability_grid,
                    model.params["preparation_hours"] * probability_grid * (1 - probability_grid),
                    color="#C00000", linewidth=2, label="β·p(1−p)")
    axes[1, 0].set(title="Der Effekt ist nahe p=0,5 am größten",
                   xlabel="Vorhergesagte Wahrscheinlichkeit",
                   ylabel="Marginaler Effekt Vorbereitung")
    axes[1, 0].legend(frameon=False)

    plot_summary = marginal_summary[
        marginal_summary["term"].isin(["preparation_hours", "mentor_program"])
    ].copy()
    plot_summary["label"] = plot_summary["term"].map(
        {"preparation_hours": "Vorbereitung", "mentor_program": "Mentoring"}
    ) + ": " + plot_summary["method"]
    positions = np.arange(len(plot_summary))
    estimates = plot_summary["marginal_effect"].to_numpy()
    left = estimates - plot_summary["ci_95_low"].to_numpy()
    right = plot_summary["ci_95_high"].to_numpy() - estimates
    axes[1, 1].errorbar(estimates, positions, xerr=np.vstack([left, right]), fmt="o",
                        color="#4472C4", capsize=5)
    axes[1, 1].axvline(0, color="#C00000", linestyle="--")
    axes[1, 1].set(title="AME und MEM beantworten verschiedene Fragen",
                   xlabel="Wahrscheinlichkeitsänderung mit 95%-KI",
                   yticks=positions, yticklabels=plot_summary["label"])
    figure.suptitle("Marginale Logit-Effekte hängen vom Ausgangsniveau ab")
    figure.tight_layout()
    figure.savefig(RESULTS / "marginal_effects.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
