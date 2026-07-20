"""Musterlösung für T50: IV als Kovarianzquotient und 2SLS."""

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
RNG_SEED = 20260730
TRUE_EFFECT = 1.5


def make_data(observations: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    experience = rng.normal(size=observations)
    ability = rng.normal(size=observations)
    scholarship = 0.6 * experience + rng.normal(size=observations)
    education = (12 + 1.6 * scholarship + 0.8 * experience + 1.2 * ability
                 + rng.normal(0, 1.5, observations))
    wage = (5 + TRUE_EFFECT * education + 2.0 * experience + 3.0 * ability
            + rng.normal(0, 2.5, observations))
    return pd.DataFrame({"scholarship_index": scholarship, "education_years": education,
                         "wage_index": wage, "experience_index": experience,
                         "oracle_ability": ability})


def residualize(data: pd.DataFrame, variable: str) -> np.ndarray:
    controls = sm.add_constant(data[["experience_index"]])
    return sm.OLS(data[variable], controls).fit().resid.to_numpy()


def covariance_ratio(z: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return np.cov(z, y, ddof=1)[0, 1] / np.cov(z, x, ddof=1)[0, 1]


def controlled_ratio(data: pd.DataFrame) -> float:
    z_resid = residualize(data, "scholarship_index")
    x_resid = residualize(data, "education_years")
    y_resid = residualize(data, "wage_index")
    return covariance_ratio(z_resid, x_resid, y_resid)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "education_iv_data.csv", index=False)

    ols = sm.OLS(
        data["wage_index"], sm.add_constant(data[["education_years", "experience_index"]])
    ).fit()
    oracle = sm.OLS(
        data["wage_index"],
        sm.add_constant(data[["education_years", "experience_index", "oracle_ability"]]),
    ).fit()
    unconditional_ratio = covariance_ratio(
        data["scholarship_index"].to_numpy(), data["education_years"].to_numpy(),
        data["wage_index"].to_numpy()
    )
    residual_iv = controlled_ratio(data)

    first_stage = sm.OLS(
        data["education_years"], sm.add_constant(data[["scholarship_index", "experience_index"]])
    ).fit()
    reduced_form = sm.OLS(
        data["wage_index"], sm.add_constant(data[["scholarship_index", "experience_index"]])
    ).fit()
    coefficient_ratio = (reduced_form.params["scholarship_index"]
                         / first_stage.params["scholarship_index"])
    second_stage_data = data.copy()
    second_stage_data["predicted_education"] = first_stage.fittedvalues
    second_stage = sm.OLS(
        second_stage_data["wage_index"],
        sm.add_constant(second_stage_data[["predicted_education", "experience_index"]]),
    ).fit()
    two_stage_estimate = second_stage.params["predicted_education"]

    scaled = data.copy()
    scaled["scholarship_index"] = 50 + 10 * scaled["scholarship_index"]
    scaled_ratio = controlled_ratio(scaled)
    equivalence_error = max(abs(residual_iv - coefficient_ratio),
                            abs(residual_iv - two_stage_estimate),
                            abs(residual_iv - scaled_ratio))
    assert equivalence_error < 1e-10
    assert abs(residual_iv - TRUE_EFFECT) < 0.2
    assert ols.params["education_years"] - TRUE_EFFECT > 0.5
    assert abs(unconditional_ratio - residual_iv) > 0.2

    estimates = pd.DataFrame([
        {"method": "OLS with experience", "estimate": ols.params["education_years"]},
        {"method": "Unconditional covariance ratio", "estimate": unconditional_ratio},
        {"method": "Residualized covariance ratio", "estimate": residual_iv},
        {"method": "Controlled reduced form / first stage", "estimate": coefficient_ratio},
        {"method": "Manual two-stage least squares", "estimate": two_stage_estimate},
        {"method": "Oracle OLS with ability", "estimate": oracle.params["education_years"]},
    ])
    estimates.to_csv(RESULTS / "iv_estimator_equivalence.csv", index=False)
    components = pd.DataFrame([{
        "controlled_first_stage": first_stage.params["scholarship_index"],
        "controlled_reduced_form": reduced_form.params["scholarship_index"],
        "ratio": coefficient_ratio,
        "first_stage_f": first_stage.tvalues["scholarship_index"] ** 2,
        "scaled_instrument_ratio": scaled_ratio,
        "max_equivalence_error": equivalence_error,
    }])
    components.to_csv(RESULTS / "iv_covariance_components.csv", index=False)

    rng = np.random.default_rng(RNG_SEED + 1)
    bootstrap_estimates = []
    for _ in range(400):
        indices = rng.integers(0, len(data), len(data))
        bootstrap_estimates.append(controlled_ratio(data.iloc[indices]))
    bootstrap = pd.DataFrame({"iv_estimate": bootstrap_estimates})
    bootstrap.to_csv(RESULTS / "iv_bootstrap.csv", index=False)
    ci_low, ci_high = np.quantile(bootstrap["iv_estimate"], [0.025, 0.975])
    bootstrap_summary = pd.DataFrame([{
        "point_estimate": residual_iv, "bootstrap_ci_95_low": ci_low,
        "bootstrap_ci_95_high": ci_high, "bootstrap_standard_error": bootstrap["iv_estimate"].std(ddof=1),
    }])
    bootstrap_summary.to_csv(RESULTS / "iv_bootstrap_summary.csv", index=False)

    print(estimates.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nKomponenten und Äquivalenz:")
    print(components.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nBootstrap-Inferenz:")
    print(bootstrap_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    z_resid = residualize(data, "scholarship_index")
    x_resid = residualize(data, "education_years")
    y_resid = residualize(data, "wage_index")
    sample_indices = np.random.default_rng(RNG_SEED).choice(len(data), 450, replace=False)
    z_sample = z_resid[sample_indices]
    grid = np.linspace(z_resid.min(), z_resid.max(), 100)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(z_sample, x_resid[sample_indices], color="#9DC3E6", alpha=0.55, s=22)
    axes[0].plot(grid, first_stage.params["scholarship_index"] * grid,
                 color="#4472C4", linewidth=2)
    axes[0].set(title=f"First Stage: π = {first_stage.params['scholarship_index']:.2f}",
                xlabel="Residualisiertes Instrument Z̃", ylabel="Residualisierte Bildung X̃")

    axes[1].scatter(z_sample, y_resid[sample_indices], color="#F4B183", alpha=0.55, s=22)
    axes[1].plot(grid, reduced_form.params["scholarship_index"] * grid,
                 color="#C00000", linewidth=2)
    axes[1].set(title=f"Reduced Form: ρ = {reduced_form.params['scholarship_index']:.2f}",
                xlabel="Residualisiertes Instrument Z̃", ylabel="Residualisierter Lohn Ỹ")

    plot_estimates = estimates.copy()
    y_pos = np.arange(len(plot_estimates))
    axes[2].scatter(plot_estimates["estimate"], y_pos, color="#4472C4", s=55)
    axes[2].axvline(TRUE_EFFECT, color="#C00000", linestyle="--", label="Wahrer Effekt")
    axes[2].set(title=f"ρ / π = Cov(Z̃,Ỹ) / Cov(Z̃,X̃) = {residual_iv:.2f}",
                xlabel="Bildungseffekt", yticks=y_pos,
                yticklabels=["OLS", "Ratio ohne Kontrolle", "Residualisierte Ratio",
                             "RF / FS", "2SLS", "Oracle OLS"])
    axes[2].legend(frameon=False)
    figure.suptitle("Alle korrekt kontrollierten IV-Darstellungen liefern denselben Schätzer")
    figure.tight_layout()
    figure.savefig(RESULTS / "iv_covariance_ratio.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
