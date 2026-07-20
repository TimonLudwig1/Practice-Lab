"""Musterlösung für T26: Linear Probability Model."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    study_hours = rng.uniform(0, 22, size=size)
    true_probability = expit(-4.5 + 0.43 * study_hours)
    completed = rng.binomial(1, true_probability)
    return pd.DataFrame(
        {"learner_id": np.arange(1, size + 1),
         "study_hours_per_week": study_hours, "completed": completed,
         "oracle_completion_probability": true_probability}
    )


def coefficient_table(model, robust_model) -> pd.DataFrame:
    critical = 1.96
    rows = []
    for covariance, standard_errors in (
        ("Classical", model.bse.to_numpy()), ("HC1 robust", robust_model.bse)
    ):
        for index, term in enumerate(model.params.index):
            estimate = model.params.iloc[index]
            standard_error = standard_errors[index]
            rows.append(
                {"covariance": covariance, "term": term, "estimate": estimate,
                 "standard_error": standard_error,
                 "ci_95_low": estimate - critical * standard_error,
                 "ci_95_high": estimate + critical * standard_error}
            )
    return pd.DataFrame(rows)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "course_completion.csv", index=False)
    design = sm.add_constant(data[["study_hours_per_week"]])
    model = sm.OLS(data["completed"], design).fit()
    robust_model = model.get_robustcov_results(cov_type="HC1")
    coefficients = coefficient_table(model, robust_model)
    coefficients.to_csv(RESULTS / "coefficients.csv", index=False)

    fitted = model.fittedvalues
    bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(model.resid, design)
    extended_hours = np.linspace(-3, 27, 301)
    extended_design = pd.DataFrame({"const": 1.0, "study_hours_per_week": extended_hours})
    extended_fit = model.predict(extended_design)
    diagnostics = pd.DataFrame(
        [{"sample_min_fitted": fitted.min(), "sample_max_fitted": fitted.max(),
          "sample_share_outside_0_1": ((fitted < 0) | (fitted > 1)).mean(),
          "extended_min_prediction": extended_fit.min(),
          "extended_max_prediction": extended_fit.max(),
          "extended_share_outside_0_1": ((extended_fit < 0) | (extended_fit > 1)).mean(),
          "breusch_pagan_lm": bp_lm, "breusch_pagan_lm_pvalue": bp_lm_pvalue,
          "breusch_pagan_f": bp_f, "breusch_pagan_f_pvalue": bp_f_pvalue}]
    )
    assert set(data["completed"].unique()) == {0, 1}
    assert ((extended_fit < 0) | (extended_fit > 1)).any()
    assert bp_lm_pvalue < 0.05
    diagnostics.to_csv(RESULTS / "diagnostics.csv", index=False)

    data["hours_bin"] = pd.cut(data["study_hours_per_week"], bins=np.linspace(0, 22, 12),
                                include_lowest=True)
    binned = (
        data.groupby("hours_bin", observed=True)
        .agg(mean_hours=("study_hours_per_week", "mean"),
             completion_rate=("completed", "mean"), observations=("completed", "size"),
             mean_squared_residual=("completed", lambda values: np.nan))
        .reset_index()
    )
    squared_residual = pd.Series(model.resid.to_numpy() ** 2, index=data.index)
    binned_variance = squared_residual.groupby(data["hours_bin"], observed=True).mean()
    binned["mean_squared_residual"] = binned["hours_bin"].map(binned_variance).astype(float)
    binned["lpm_prediction"] = model.predict(
        pd.DataFrame({"const": 1.0, "study_hours_per_week": binned["mean_hours"]})
    )
    binned.to_csv(RESULTS / "binned_completion_rates.csv", index=False)

    print(coefficients.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nDiagnostik:")
    print(diagnostics.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    rng = np.random.default_rng(RNG_SEED + 1)
    jittered = data["completed"] + rng.normal(0, 0.025, size=len(data))
    axes[0, 0].scatter(data["study_hours_per_week"], jittered, color="#9DC3E6",
                       alpha=0.3, s=15, label="Einzeloutcomes")
    axes[0, 0].scatter(binned["mean_hours"], binned["completion_rate"], color="#C00000",
                       s=48, label="Abschlussquote je Bin")
    observed_grid = np.linspace(0, 22, 220)
    observed_fit = model.predict(
        pd.DataFrame({"const": 1.0, "study_hours_per_week": observed_grid})
    )
    axes[0, 0].plot(observed_grid, observed_fit, color="#4472C4", linewidth=2,
                    label=f"LPM: +{100 * model.params['study_hours_per_week']:.1f} pp/Stunde")
    axes[0, 0].set(title="LPM als lineare Approximation", xlabel="Lernstunden pro Woche",
                   ylabel="Abschluss (0/1) beziehungsweise Anteil", ylim=(-0.1, 1.1))
    axes[0, 0].legend(frameon=False, fontsize=8)

    valid = (extended_fit >= 0) & (extended_fit <= 1)
    valid_fit = np.where(valid, extended_fit, np.nan)
    invalid_fit = np.where(valid, np.nan, extended_fit)
    axes[0, 1].plot(extended_hours, valid_fit, color="#4472C4", linewidth=3,
                    label="Innerhalb [0,1]")
    axes[0, 1].plot(extended_hours, invalid_fit, color="#C00000", linewidth=3,
                    label="Ungültige Wahrscheinlichkeit")
    axes[0, 1].axhline(0, color="#595959", linewidth=1)
    axes[0, 1].axhline(1, color="#595959", linewidth=1)
    axes[0, 1].axvspan(0, 22, color="#A5A5A5", alpha=0.12, label="Beobachteter Bereich")
    axes[0, 1].set(title="Die Gerade kennt keine Wahrscheinlichkeitsgrenzen",
                   xlabel="Lernstunden pro Woche", ylabel="LPM-Vorhersage")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].scatter(binned["lpm_prediction"], binned["mean_squared_residual"],
                       color="#4472C4", s=52, label="Gebinnte quadrierte Residuen")
    probability_grid = np.linspace(0, 1, 250)
    axes[1, 0].plot(probability_grid, probability_grid * (1 - probability_grid),
                    color="#C00000", linewidth=2, label="p(1−p)")
    axes[1, 0].set(title="Binäre Residuen sind heteroskedastisch",
                   xlabel="Vorhergesagte Wahrscheinlichkeit", ylabel="Mittleres Residuum²",
                   xlim=(0, 1), ylim=(0, 0.28))
    axes[1, 0].legend(frameon=False, fontsize=8)

    slope_rows = coefficients[coefficients["term"] == "study_hours_per_week"]
    positions = np.arange(len(slope_rows))
    estimates = slope_rows["estimate"].to_numpy()
    left = estimates - slope_rows["ci_95_low"].to_numpy()
    right = slope_rows["ci_95_high"].to_numpy() - estimates
    axes[1, 1].errorbar(estimates, positions, xerr=np.vstack([left, right]), fmt="o",
                        color="#4472C4", capsize=5)
    axes[1, 1].axvline(0, color="#C00000", linestyle="--")
    axes[1, 1].set(title="Gleiche Steigung, andere Standardfehler",
                   xlabel="Wahrscheinlichkeitsänderung pro Stunde",
                   yticks=positions, yticklabels=slope_rows["covariance"])
    figure.suptitle("Das Linear Probability Model ist intuitiv, aber nicht begrenzt")
    figure.tight_layout()
    figure.savefig(RESULTS / "linear_probability_model.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
