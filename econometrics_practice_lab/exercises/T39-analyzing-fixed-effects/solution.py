"""Musterlösung für T39: Fixed Effects direkt schätzen und analysieren."""

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
TRUE_BETA = 1.2


def make_data(hospitals: int = 60, max_periods: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    quality = rng.normal(0, 5, hospitals)
    urban = (quality + rng.normal(0, 5, hospitals) > 0).astype(int)
    rows = []
    for hospital in range(hospitals):
        observed_periods = np.sort(rng.choice(np.arange(1, max_periods + 1),
                                              size=rng.integers(4, max_periods + 1), replace=False))
        for period in observed_periods:
            staffing = 18 + 0.45 * quality[hospital] + rng.normal(0, 2)
            true_intercept = 70 + quality[hospital]
            patient_score = true_intercept + TRUE_BETA * staffing + rng.normal(0, 4)
            rows.append((hospital + 1, period, staffing, patient_score, true_intercept, urban[hospital]))
    return pd.DataFrame(rows, columns=["hospital_id", "period", "nurses_per_100_beds",
                                       "patient_score", "oracle_hospital_intercept", "urban"])


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "hospital_panel.csv", index=False)
    x_name = "nurses_per_100_beds"
    pooled = sm.OLS(data["patient_score"], sm.add_constant(data[[x_name]])).fit()
    hospital_dummies = pd.get_dummies(data["hospital_id"], prefix="hospital", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([data[[x_name]], hospital_dummies], axis=1))
    model = sm.OLS(data["patient_score"], design).fit()
    coefficient_table = pd.DataFrame(
        [{"model": "Pooled OLS", "staffing_coefficient": pooled.params[x_name],
          "standard_error": pooled.bse[x_name], "true_coefficient": TRUE_BETA},
         {"model": "Hospital fixed effects", "staffing_coefficient": model.params[x_name],
          "standard_error": model.bse[x_name], "true_coefficient": TRUE_BETA}]
    )
    assert abs(model.params[x_name] - TRUE_BETA) < 0.25
    coefficient_table.to_csv(RESULTS / "staffing_coefficient.csv", index=False)

    covariance = model.cov_params()
    summary = (
        data.groupby("hospital_id")
        .agg(observed_periods=("period", "size"), urban=("urban", "first"),
             oracle_hospital_intercept=("oracle_hospital_intercept", "first"))
        .reset_index()
    )
    estimate_rows = []
    for row in summary.itertuples(index=False):
        if row.hospital_id == 1:
            estimate = model.params["const"]
            variance = covariance.loc["const", "const"]
        else:
            dummy = f"hospital_{row.hospital_id}"
            estimate = model.params["const"] + model.params[dummy]
            variance = (
                covariance.loc["const", "const"] + covariance.loc[dummy, dummy]
                + 2 * covariance.loc["const", dummy]
            )
        standard_error = np.sqrt(variance)
        estimate_rows.append(
            {"hospital_id": row.hospital_id, "observed_periods": row.observed_periods,
             "urban": row.urban, "estimated_intercept": estimate,
             "standard_error": standard_error, "ci_95_low": estimate - 1.96 * standard_error,
             "ci_95_high": estimate + 1.96 * standard_error,
             "oracle_hospital_intercept": row.oracle_hospital_intercept}
        )
    effects = pd.DataFrame(estimate_rows)
    effects["estimated_centered_fe"] = effects["estimated_intercept"] - effects["estimated_intercept"].mean()
    effects["oracle_centered_fe"] = (
        effects["oracle_hospital_intercept"] - effects["oracle_hospital_intercept"].mean()
    )
    effects["absolute_centered_error"] = (
        effects["estimated_centered_fe"] - effects["oracle_centered_fe"]
    ).abs()
    effects["interval_width"] = effects["ci_95_high"] - effects["ci_95_low"]
    effects["estimated_rank"] = effects["estimated_centered_fe"].rank(ascending=False, method="min")
    effects["oracle_rank"] = effects["oracle_centered_fe"].rank(ascending=False, method="min")
    effects.to_csv(RESULTS / "hospital_fixed_effects.csv", index=False)

    cutoff = int(np.ceil(len(effects) * 0.25))
    estimated_top = set(effects.nsmallest(cutoff, "estimated_rank")["hospital_id"])
    oracle_top = set(effects.nsmallest(cutoff, "oracle_rank")["hospital_id"])
    fe_correlation = effects["estimated_centered_fe"].corr(effects["oracle_centered_fe"])
    top_overlap = len(estimated_top & oracle_top) / cutoff
    diagnostics = pd.DataFrame(
        [{"centered_fe_correlation": fe_correlation, "top_quartile_overlap": top_overlap,
          "correlation_periods_interval_width": effects["observed_periods"].corr(effects["interval_width"]),
          "mean_absolute_centered_error": effects["absolute_centered_error"].mean()}]
    )
    assert fe_correlation > 0.75
    assert top_overlap >= 0.60
    diagnostics.to_csv(RESULTS / "fixed_effect_diagnostics.csv", index=False)
    print(coefficient_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFixed-Effect-Diagnostik:")
    print(diagnostics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    sizes = 20 + 5 * effects["observed_periods"]
    axes[0, 0].scatter(effects["oracle_centered_fe"], effects["estimated_centered_fe"],
                       s=sizes, color="#4472C4", alpha=0.65)
    limits = [min(effects["oracle_centered_fe"].min(), effects["estimated_centered_fe"].min()),
              max(effects["oracle_centered_fe"].max(), effects["estimated_centered_fe"].max())]
    axes[0, 0].plot(limits, limits, color="#C00000", linestyle="--")
    axes[0, 0].set(title=f"Geschätzte versus wahre zentrierte FE (r={fe_correlation:.2f})",
                   xlabel="Wahre Krankenhausqualität", ylabel="Geschätzter Fixed Effect")

    selected = pd.concat([effects.nsmallest(4, "estimated_rank"),
                          effects.nlargest(4, "estimated_rank")]).sort_values("estimated_intercept")
    positions = np.arange(len(selected))
    estimates = selected["estimated_intercept"].to_numpy()
    left = estimates - selected["ci_95_low"].to_numpy()
    right = selected["ci_95_high"].to_numpy() - estimates
    axes[0, 1].errorbar(estimates, positions, xerr=np.vstack([left, right]), fmt="o",
                        color="#4472C4", capsize=5, label="Geschätzt")
    axes[0, 1].scatter(selected["oracle_hospital_intercept"], positions, marker="x",
                       color="#C00000", s=65, label="Wahr")
    axes[0, 1].set(title="Extreme Rankings bleiben unsicher", xlabel="Krankenhausintercept",
                   yticks=positions,
                   yticklabels=[f"KH {value}" for value in selected["hospital_id"]])
    axes[0, 1].legend(frameon=False)

    axes[1, 0].scatter(effects["observed_periods"], effects["interval_width"],
                       color="#4472C4", alpha=0.7, s=32)
    axes[1, 0].set(title="Mehr Perioden liefern engere Intervalle",
                   xlabel="Beobachtete Perioden", ylabel="Breite des 95%-KI")

    axes[1, 1].boxplot(
        [effects.loc[effects["urban"] == 0, "estimated_centered_fe"],
         effects.loc[effects["urban"] == 1, "estimated_centered_fe"]],
        tick_labels=["Nicht urban", "Urban"], patch_artist=True,
        boxprops={"facecolor": "#9DC3E6"}, medianprops={"color": "#C00000"}
    )
    axes[1, 1].axhline(0, color="#595959", linestyle="--")
    axes[1, 1].set(title="Post-hoc-Gruppenvergleich ist deskriptiv",
                   ylabel="Geschätzter zentrierter Fixed Effect")
    figure.suptitle("Direkt geschätzte Fixed Effects sind informative, aber unsichere Größen")
    figure.tight_layout()
    figure.savefig(RESULTS / "analyzing_fixed_effects.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
