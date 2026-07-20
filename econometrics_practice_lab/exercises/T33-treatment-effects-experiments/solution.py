"""Musterlösung für T33: Treatment-Effekte in Experimenten."""

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


def make_data(size: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    baseline = rng.normal(60, 11, size=size)
    high_need = (baseline < 55).astype(int)
    treatment = np.zeros(size, dtype=int)
    treatment[rng.choice(size, size=size // 2, replace=False)] = 1
    outcome_zero = 28 + 0.72 * baseline - 3 * high_need + rng.normal(0, 7, size=size)
    treatment_effect = 4 + 3 * high_need
    outcome_one = outcome_zero + treatment_effect
    observed = np.where(treatment == 1, outcome_one, outcome_zero)
    return pd.DataFrame(
        {"employee_id": np.arange(1, size + 1), "baseline_score": baseline,
         "high_need": high_need, "treatment": treatment, "outcome_y0": outcome_zero,
         "outcome_y1": outcome_one, "individual_treatment_effect": treatment_effect,
         "followup_score": observed}
    )


def robust_term(model, term: str) -> tuple[float, float, float, float]:
    robust = model.get_robustcov_results(cov_type="HC1")
    index = model.model.exog_names.index(term)
    estimate = robust.params[index]
    standard_error = robust.bse[index]
    critical = stats.t.ppf(0.975, robust.df_resid)
    return estimate, standard_error, estimate - critical * standard_error, estimate + critical * standard_error


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "mentoring_rct.csv", index=False)
    treated = data.loc[data["treatment"] == 1, "followup_score"]
    control = data.loc[data["treatment"] == 0, "followup_score"]
    difference = treated.mean() - control.mean()
    variance_t = treated.var(ddof=1) / len(treated)
    variance_c = control.var(ddof=1) / len(control)
    welch_se = np.sqrt(variance_t + variance_c)
    welch_df = (variance_t + variance_c) ** 2 / (
        variance_t ** 2 / (len(treated) - 1) + variance_c ** 2 / (len(control) - 1)
    )
    welch_critical = stats.t.ppf(0.975, welch_df)

    treatment_only = sm.OLS(
        data["followup_score"], sm.add_constant(data[["treatment"]])
    ).fit()
    adjusted = sm.OLS(
        data["followup_score"], sm.add_constant(data[["treatment", "baseline_score", "high_need"]])
    ).fit()
    data["treatment_x_high_need"] = data["treatment"] * data["high_need"]
    interaction = sm.OLS(
        data["followup_score"],
        sm.add_constant(data[["treatment", "baseline_score", "high_need", "treatment_x_high_need"]])
    ).fit()
    assert np.isclose(difference, treatment_only.params["treatment"])

    unadjusted_robust = robust_term(treatment_only, "treatment")
    adjusted_robust = robust_term(adjusted, "treatment")
    estimates = pd.DataFrame(
        [{"method": "Difference in means (Welch)", "estimate": difference,
          "standard_error": welch_se, "ci_95_low": difference - welch_critical * welch_se,
          "ci_95_high": difference + welch_critical * welch_se},
         {"method": "Treatment-only OLS (HC1)", "estimate": unadjusted_robust[0],
          "standard_error": unadjusted_robust[1], "ci_95_low": unadjusted_robust[2],
          "ci_95_high": unadjusted_robust[3]},
         {"method": "Covariate-adjusted OLS (HC1)", "estimate": adjusted_robust[0],
          "standard_error": adjusted_robust[1], "ci_95_low": adjusted_robust[2],
          "ci_95_high": adjusted_robust[3]}]
    )
    assert adjusted_robust[1] < unadjusted_robust[1]
    estimates.to_csv(RESULTS / "effect_estimates.csv", index=False)

    robust_interaction = interaction.get_robustcov_results(cov_type="HC1")
    names = interaction.model.exog_names
    index_treatment = names.index("treatment")
    index_interaction = names.index("treatment_x_high_need")
    covariance = robust_interaction.cov_params()
    low_effect = robust_interaction.params[index_treatment]
    low_variance = covariance[index_treatment, index_treatment]
    high_effect = low_effect + robust_interaction.params[index_interaction]
    high_variance = (
        covariance[index_treatment, index_treatment]
        + covariance[index_interaction, index_interaction]
        + 2 * covariance[index_treatment, index_interaction]
    )
    critical = stats.t.ppf(0.975, robust_interaction.df_resid)
    subgroup = pd.DataFrame(
        [{"subgroup": "Low need", "estimate": low_effect, "standard_error": np.sqrt(low_variance),
          "ci_95_low": low_effect - critical * np.sqrt(low_variance),
          "ci_95_high": low_effect + critical * np.sqrt(low_variance), "true_effect": 4.0},
         {"subgroup": "High need", "estimate": high_effect, "standard_error": np.sqrt(high_variance),
          "ci_95_low": high_effect - critical * np.sqrt(high_variance),
          "ci_95_high": high_effect + critical * np.sqrt(high_variance), "true_effect": 7.0}]
    )
    subgroup.to_csv(RESULTS / "subgroup_effects.csv", index=False)

    rng = np.random.default_rng(RNG_SEED + 1)
    outcomes = data["followup_score"].to_numpy()
    assignments = data["treatment"].to_numpy()
    permutation_estimates = []
    for _ in range(2000):
        permuted = rng.permutation(assignments)
        permutation_estimates.append(outcomes[permuted == 1].mean() - outcomes[permuted == 0].mean())
    permutation = pd.DataFrame({"permutation": np.arange(1, 2001),
                                "difference_under_sharp_null": permutation_estimates})
    randomization_p = (1 + np.sum(np.abs(permutation_estimates) >= abs(difference))) / 2001
    inference = pd.DataFrame(
        [{"observed_difference": difference, "randomization_p_value_two_sided": randomization_p,
          "welch_t_statistic": difference / welch_se,
          "welch_p_value_two_sided": 2 * stats.t.sf(abs(difference / welch_se), welch_df),
          "permutations": 2000}]
    )
    assert randomization_p < 0.05
    permutation.to_csv(RESULTS / "randomization_distribution.csv", index=False)
    inference.to_csv(RESULTS / "inference_comparison.csv", index=False)
    print(estimates.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSubgruppeneffekte:")
    print(subgroup.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nInferenz:")
    print(inference.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    group_means = [control.mean(), treated.mean()]
    group_ses = [control.sem(), treated.sem()]
    axes[0, 0].errorbar(group_means, [0, 1], xerr=1.96 * np.array(group_ses), fmt="o",
                        color="#4472C4", capsize=5)
    axes[0, 0].set(title=f"Unadjustierter Effekt = {difference:.2f}",
                   xlabel="Follow-up-Score mit 95%-KI", yticks=[0, 1],
                   yticklabels=["Kontrolle", "Treatment"])

    positions = np.arange(len(estimates))
    values = estimates["estimate"].to_numpy()
    left = values - estimates["ci_95_low"].to_numpy()
    right = estimates["ci_95_high"].to_numpy() - values
    axes[0, 1].errorbar(values, positions, xerr=np.vstack([left, right]), fmt="o",
                        color="#4472C4", capsize=5)
    axes[0, 1].axvline(0, color="#C00000", linestyle="--")
    axes[0, 1].set(title="Adjustierung erhöht die Präzision", xlabel="Geschätzter ATE mit 95%-KI",
                   yticks=positions, yticklabels=estimates["method"])

    axes[1, 0].hist(permutation["difference_under_sharp_null"], bins=40,
                    color="#9DC3E6", edgecolor="white")
    axes[1, 0].axvline(difference, color="#C00000", linewidth=2,
                       label=f"Beobachtet = {difference:.2f}")
    axes[1, 0].axvline(-difference, color="#C00000", linewidth=2)
    axes[1, 0].set(title=f"Randomisierungsinferenz: p={randomization_p:.4f}",
                   xlabel="Differenz unter scharfer Null", ylabel="Häufigkeit")
    axes[1, 0].legend(frameon=False)

    positions = np.arange(len(subgroup))
    values = subgroup["estimate"].to_numpy()
    left = values - subgroup["ci_95_low"].to_numpy()
    right = subgroup["ci_95_high"].to_numpy() - values
    axes[1, 1].errorbar(values, positions, xerr=np.vstack([left, right]), fmt="o",
                        color="#4472C4", capsize=5, label="Geschätzt")
    axes[1, 1].scatter(subgroup["true_effect"], positions, marker="x", color="#C00000",
                       s=70, label="Wahr in Simulation")
    axes[1, 1].set(title="Vorab definierte Subgruppeneffekte", xlabel="Treatment-Effekt",
                   yticks=positions, yticklabels=subgroup["subgroup"])
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Treatment-Effekte im RCT: Schätzung, Präzision und Heterogenität")
    figure.tight_layout()
    figure.savefig(RESULTS / "treatment_effects.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
