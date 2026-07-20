"""Musterlösung für T21: Omitted Variable Bias."""

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


def make_data(size: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    ability = rng.normal(size=size)
    education = np.clip(12 + 1.2 * ability + rng.normal(0, 1.5, size=size), 8, 20)
    log_wage = 2.2 + 0.08 * education + 0.22 * ability + rng.normal(0, 0.20, size=size)
    return pd.DataFrame(
        {"person_id": np.arange(1, size + 1), "education_years": education,
         "ability_score": ability, "log_hourly_wage": log_wage,
         "hourly_wage_eur": np.exp(log_wage)}
    )


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "education_wages.csv", index=False)
    short = sm.OLS(data["log_hourly_wage"],
                   sm.add_constant(data[["education_years"]])).fit()
    long = sm.OLS(data["log_hourly_wage"],
                  sm.add_constant(data[["education_years", "ability_score"]])).fit()
    auxiliary = sm.OLS(data["ability_score"],
                       sm.add_constant(data[["education_years"]])).fit()

    model_comparison = pd.DataFrame(
        [
            {"model": "Short: omit ability", "education_coefficient": short.params["education_years"],
             "education_se": short.bse["education_years"], "ability_coefficient": np.nan,
             "r_squared": short.rsquared},
            {"model": "Long: control ability", "education_coefficient": long.params["education_years"],
             "education_se": long.bse["education_years"],
             "ability_coefficient": long.params["ability_score"], "r_squared": long.rsquared},
        ]
    )
    model_comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

    observed_difference = short.params["education_years"] - long.params["education_years"]
    predicted_difference = long.params["ability_score"] * auxiliary.params["education_years"]
    decomposition = pd.DataFrame(
        [{"short_minus_long": observed_difference,
          "ability_effect_gamma": long.params["ability_score"],
          "education_ability_slope_delta": auxiliary.params["education_years"],
          "gamma_times_delta": predicted_difference,
          "identity_error": observed_difference - predicted_difference,
          "true_education_effect": 0.08}]
    )
    decomposition.to_csv(RESULTS / "ovb_decomposition.csv", index=False)
    assert np.isclose(observed_difference, predicted_difference)
    print(model_comparison.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nOVB-Zerlegung:")
    print(decomposition.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(data["education_years"], data["ability_score"], color="#5B9BD5",
                       alpha=0.55, s=22)
    order = np.argsort(data["education_years"].to_numpy())
    axes[0, 0].plot(data["education_years"].to_numpy()[order],
                    auxiliary.fittedvalues.to_numpy()[order], color="#C00000", linewidth=2,
                    label=f"δ̂ = {auxiliary.params['education_years']:.3f}")
    axes[0, 0].set(title="Ausgelassene Fähigkeit korreliert mit Bildung",
                   xlabel="Bildungsjahre", ylabel="Fähigkeitsskala")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].scatter(data["ability_score"], short.resid, color="#4472C4", alpha=0.55, s=22)
    residual_aux = sm.OLS(short.resid, sm.add_constant(data[["ability_score"]])).fit()
    ability_order = np.argsort(data["ability_score"].to_numpy())
    axes[0, 1].plot(data["ability_score"].to_numpy()[ability_order],
                    residual_aux.fittedvalues.to_numpy()[ability_order], color="#C00000", linewidth=2)
    axes[0, 1].axhline(0, color="#595959", linestyle="--")
    axes[0, 1].set(title="Short-Model-Residuen enthalten Fähigkeit",
                   xlabel="Fähigkeitsskala", ylabel="Residuum log(Lohn)")

    estimates = model_comparison["education_coefficient"].to_numpy()
    ses = model_comparison["education_se"].to_numpy()
    positions = np.arange(len(estimates))
    axes[1, 0].errorbar(estimates, positions, xerr=1.96 * ses, fmt="o", color="#4472C4",
                        capsize=5)
    axes[1, 0].axvline(0.08, color="#C00000", linestyle="--", label="Wahrer Bildungseffekt")
    axes[1, 0].set(title="Auslassung verschiebt den Bildungskoeffizienten",
                   xlabel="Koeffizient in log(Lohn)", yticks=positions,
                   yticklabels=["Kurz", "Lang"])
    axes[1, 0].legend(frameon=False)

    long_component = long.params["education_years"]
    axes[1, 1].bar(["Short-Koeffizient"], [long_component], color="#70AD47",
                   label=f"Long = {long_component:.3f}")
    axes[1, 1].bar(["Short-Koeffizient"], [predicted_difference],
                   bottom=[long_component], color="#ED7D31",
                   label=f"Ability Bias = {predicted_difference:.3f}")
    axes[1, 1].text(0, short.params["education_years"] - 0.004,
                    f"Summe = Short = {short.params['education_years']:.3f}",
                    ha="center", va="top")
    axes[1, 1].set(title="Short = Long + ausgelassener Anteil",
                   ylabel="Beitrag zum Bildungskoeffizienten")
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Omitted Variable Bias entsteht durch Wirkung und Korrelation")
    figure.tight_layout()
    figure.savefig(RESULTS / "omitted_variable_bias.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
