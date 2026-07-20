"""Musterlösung für T24: Interaktionseffekte und marginale Effekte."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    experience = rng.uniform(0, 15, size=size)
    treated = rng.binomial(1, 0.5, size=size)
    score = 50 + 1.6 * experience + 4 * treated + 0.9 * experience * treated
    score += rng.normal(0, 6, size=size)
    return pd.DataFrame(
        {"employee_id": np.arange(1, size + 1), "experience_years": experience,
         "treated": treated, "performance_score": score}
    )


def marginal_effect(model, experience: float) -> dict[str, float]:
    beta_treatment = model.params["treated"]
    beta_interaction = model.params["treated_x_experience"]
    covariance = model.cov_params()
    variance = (
        covariance.loc["treated", "treated"]
        + experience ** 2 * covariance.loc["treated_x_experience", "treated_x_experience"]
        + 2 * experience * covariance.loc["treated", "treated_x_experience"]
    )
    estimate = beta_treatment + experience * beta_interaction
    standard_error = np.sqrt(variance)
    critical = stats.t.ppf(0.975, df=model.df_resid)
    return {"experience_years": experience, "treatment_effect": estimate,
            "standard_error": standard_error, "ci_95_low": estimate - critical * standard_error,
            "ci_95_high": estimate + critical * standard_error}


def comparison_row(name: str, model) -> dict[str, float | str]:
    return {"model": name, "r_squared": model.rsquared,
            "adjusted_r_squared": model.rsquared_adj,
            "rmse": np.sqrt(np.mean(model.resid ** 2))}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data["treated_x_experience"] = data["treated"] * data["experience_years"]
    data.to_csv(DATA / "training_program.csv", index=False)
    main_effects = sm.OLS(
        data["performance_score"],
        sm.add_constant(data[["experience_years", "treated"]]),
    ).fit()
    interaction = sm.OLS(
        data["performance_score"],
        sm.add_constant(data[["experience_years", "treated", "treated_x_experience"]]),
    ).fit()

    comparison = pd.DataFrame(
        [comparison_row("No interaction", main_effects),
         comparison_row("Treatment × experience", interaction)]
    )
    assert interaction.rsquared > main_effects.rsquared
    assert np.sqrt(np.mean(interaction.resid ** 2)) < np.sqrt(np.mean(main_effects.resid ** 2))
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
    ci = interaction.conf_int()
    coefficients = pd.DataFrame(
        {"term": interaction.params.index, "estimate": interaction.params.values,
         "standard_error": interaction.bse.values, "p_value": interaction.pvalues.values,
         "ci_95_low": ci[0].values, "ci_95_high": ci[1].values}
    )
    coefficients.to_csv(RESULTS / "coefficients.csv", index=False)

    requested_effects = pd.DataFrame([marginal_effect(interaction, value)
                                      for value in (0, 5, 10, 15)])
    assert requested_effects["treatment_effect"].is_monotonic_increasing
    assert (requested_effects["ci_95_low"] > 0).all()
    requested_effects.to_csv(RESULTS / "marginal_treatment_effects.csv", index=False)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMarginale Treatment-Effekte:")
    print(requested_effects.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    experience_grid = np.linspace(0, 15, 160)
    marginal_grid = pd.DataFrame([marginal_effect(interaction, value)
                                  for value in experience_grid])
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = {0: "#4472C4", 1: "#ED7D31"}
    labels = {0: "Kontrolle", 1: "Treatment"}
    for treated in (0, 1):
        subset = data[data["treated"] == treated]
        axes[0, 0].scatter(subset["experience_years"], subset["performance_score"],
                           color=colors[treated], alpha=0.45, s=22, label=labels[treated])
        design = pd.DataFrame(
            {"const": 1.0, "experience_years": experience_grid, "treated": treated,
             "treated_x_experience": treated * experience_grid}
        )
        axes[0, 0].plot(experience_grid, interaction.predict(design), color=colors[treated],
                        linewidth=2)
    axes[0, 0].set(title="Interaktionsmodell erlaubt verschiedene Steigungen",
                   xlabel="Berufserfahrung (Jahre)", ylabel="Leistungsscore")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(marginal_grid["experience_years"], marginal_grid["treatment_effect"],
                    color="#4472C4", linewidth=2, label="Treatment-Effekt")
    axes[0, 1].fill_between(marginal_grid["experience_years"], marginal_grid["ci_95_low"],
                            marginal_grid["ci_95_high"], color="#4472C4", alpha=0.2,
                            label="95%-KI")
    axes[0, 1].axhline(0, color="#C00000", linestyle="--")
    axes[0, 1].set(title="Bedingter Treatment-Effekt", xlabel="Berufserfahrung (Jahre)",
                   ylabel="Treatment − Kontrolle")
    axes[0, 1].legend(frameon=False)

    data["main_residual"] = main_effects.resid
    data["experience_bin"] = pd.cut(data["experience_years"], bins=np.linspace(0, 15, 7),
                                     include_lowest=True)
    grouped = (
        data.groupby(["treated", "experience_bin"], observed=True)
        .agg(mean_experience=("experience_years", "mean"), mean_residual=("main_residual", "mean"))
        .reset_index()
    )
    for treated in (0, 1):
        subset = grouped[grouped["treated"] == treated]
        axes[1, 0].plot(subset["mean_experience"], subset["mean_residual"], marker="o",
                        color=colors[treated], label=labels[treated])
    axes[1, 0].axhline(0, color="#C00000", linestyle="--")
    axes[1, 0].set(title="Ohne Interaktion bleiben Gruppenmuster", xlabel="Berufserfahrung (Jahre)",
                   ylabel="Mittleres Residuum")
    axes[1, 0].legend(frameon=False)

    terms = coefficients[coefficients["term"] != "const"]
    estimates = terms["estimate"].to_numpy()
    positions = np.arange(len(terms))
    left = estimates - terms["ci_95_low"].to_numpy()
    right = terms["ci_95_high"].to_numpy() - estimates
    axes[1, 1].errorbar(estimates, positions, xerr=np.vstack([left, right]), fmt="o",
                        color="#4472C4", capsize=5)
    axes[1, 1].axvline(0, color="#C00000", linestyle="--")
    axes[1, 1].set(title="Interaktionskoeffizienten", xlabel="Koeffizient mit 95%-KI",
                   yticks=positions, yticklabels=terms["term"])
    figure.suptitle("Interaktionen machen Effekte von einem zweiten Merkmal abhängig")
    figure.tight_layout()
    figure.savefig(RESULTS / "interaction_effects.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
