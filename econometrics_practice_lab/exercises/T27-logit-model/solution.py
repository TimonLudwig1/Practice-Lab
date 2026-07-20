"""Musterlösung für T27: Logit-Modell."""

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
REGRESSORS = ["price_increase_pct", "loyalty_years", "automatic_payment"]


def make_data(size: int = 1400) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    price_increase = rng.uniform(0, 25, size=size)
    loyalty = rng.uniform(0, 8, size=size)
    automatic_payment = rng.binomial(1, 0.58, size=size)
    linear_index = 2.3 - 0.18 * price_increase + 0.24 * loyalty + 0.72 * automatic_payment
    true_probability = expit(linear_index)
    renewed = rng.binomial(1, true_probability)
    return pd.DataFrame(
        {"customer_id": np.arange(1, size + 1), "price_increase_pct": price_increase,
         "loyalty_years": loyalty, "automatic_payment": automatic_payment,
         "renewed": renewed, "oracle_renewal_probability": true_probability}
    )


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "subscription_renewals.csv", index=False)
    design = sm.add_constant(data[REGRESSORS])
    model = sm.Logit(data["renewed"], design).fit(disp=False)
    ci = model.conf_int()
    coefficients = pd.DataFrame(
        {"term": model.params.index, "log_odds_coefficient": model.params.values,
         "standard_error": model.bse.values, "p_value": model.pvalues.values,
         "ci_95_low_log_odds": ci[0].values, "ci_95_high_log_odds": ci[1].values,
         "odds_ratio": np.exp(model.params.values),
         "ci_95_low_odds_ratio": np.exp(ci[0].values),
         "ci_95_high_odds_ratio": np.exp(ci[1].values)}
    )
    coefficients.to_csv(RESULTS / "coefficients_and_odds_ratios.csv", index=False)

    scenario_prices = np.array([0, 5, 10, 15, 20, 25], dtype=float)
    scenario_design = pd.DataFrame(
        {"const": 1.0, "price_increase_pct": scenario_prices,
         "loyalty_years": 3.0, "automatic_payment": 0.0}
    )
    scenario_index = scenario_design @ model.params
    scenarios = pd.DataFrame(
        {"price_increase_pct": scenario_prices, "loyalty_years": 3.0,
         "automatic_payment": 0, "linear_index_log_odds": scenario_index,
         "odds": np.exp(scenario_index), "renewal_probability": expit(scenario_index)}
    )
    assert np.allclose(scenarios["renewal_probability"], model.predict(scenario_design))
    assert np.allclose(
        np.log(scenarios["renewal_probability"] / (1 - scenarios["renewal_probability"])),
        scenarios["linear_index_log_odds"]
    )
    scenarios.to_csv(RESULTS / "scenario_predictions.csv", index=False)

    predicted = model.predict(design)
    data["predicted_probability"] = predicted
    data["prediction_decile"] = pd.qcut(predicted, 10, labels=False, duplicates="drop") + 1
    calibration = (
        data.groupby("prediction_decile", observed=True)
        .agg(observations=("renewed", "size"),
             mean_predicted_probability=("predicted_probability", "mean"),
             observed_renewal_rate=("renewed", "mean"))
        .reset_index()
    )
    calibration.to_csv(RESULTS / "calibration_deciles.csv", index=False)

    base_probability = data["renewed"].mean()
    model_brier = np.mean((data["renewed"] - predicted) ** 2)
    base_brier = np.mean((data["renewed"] - base_probability) ** 2)
    null_log_likelihood = np.sum(
        data["renewed"] * np.log(base_probability)
        + (1 - data["renewed"]) * np.log(1 - base_probability)
    )
    diagnostics = pd.DataFrame(
        [{"converged": model.mle_retvals["converged"], "log_likelihood": model.llf,
          "null_log_likelihood": null_log_likelihood, "mcfadden_pseudo_r_squared": model.prsquared,
          "aic": model.aic, "base_rate": base_probability, "model_brier": model_brier,
          "constant_brier": base_brier}]
    )
    price_odds_ratio = coefficients.loc[
        coefficients["term"] == "price_increase_pct", "odds_ratio"
    ].iloc[0]
    assert model.mle_retvals["converged"]
    assert price_odds_ratio < 1
    assert ((predicted > 0) & (predicted < 1)).all()
    assert model_brier < base_brier
    diagnostics.to_csv(RESULTS / "model_diagnostics.csv", index=False)

    print(coefficients.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSzenariovorhersagen:")
    print(scenarios.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    price_grid = np.linspace(0, 25, 240)
    plot_loyalty = data["loyalty_years"].mean()
    plot_automatic_payment = data["automatic_payment"].mean()
    plot_design = pd.DataFrame(
        {"const": 1.0, "price_increase_pct": price_grid,
         "loyalty_years": plot_loyalty,
         "automatic_payment": plot_automatic_payment}
    )
    plot_index = plot_design @ model.params
    plot_probability = expit(plot_index)
    data["price_bin"] = pd.cut(data["price_increase_pct"], np.linspace(0, 25, 11),
                                include_lowest=True)
    raw_bins = (
        data.groupby("price_bin", observed=True)
        .agg(mean_price=("price_increase_pct", "mean"), renewal_rate=("renewed", "mean"))
        .reset_index()
    )
    rng = np.random.default_rng(RNG_SEED + 1)
    jittered = data["renewed"] + rng.normal(0, 0.025, size=len(data))
    axes[0, 0].scatter(data["price_increase_pct"], jittered, color="#9DC3E6",
                       alpha=0.25, s=14, label="Einzeloutcomes")
    axes[0, 0].scatter(raw_bins["mean_price"], raw_bins["renewal_rate"], color="#C00000",
                       s=45, label="Roher Anteil je Preis-Bin")
    axes[0, 0].plot(price_grid, plot_probability, color="#4472C4", linewidth=2,
                    label="Logit bei mittleren Kovariaten")
    axes[0, 0].set(title="Logit bildet eine S-förmige Wahrscheinlichkeit",
                   xlabel="Preiserhöhung (Prozentpunkte)", ylabel="Verlängerung/Anteil",
                   ylim=(-0.08, 1.08))
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(price_grid, plot_index, color="#4472C4", linewidth=2)
    axes[0, 1].axhline(0, color="#595959", linestyle="--")
    axes[0, 1].set(title="Linearität gilt für Log-Odds", xlabel="Preiserhöhung (Prozentpunkte)",
                   ylabel="Linearer Index η = log(Odds)")

    plot_odds = np.exp(plot_index)
    axes[1, 0].plot(price_grid, plot_odds, color="#ED7D31", linewidth=2)
    axes[1, 0].axhline(1, color="#595959", linestyle="--", label="Odds = 1 ↔ p = 0,5")
    axes[1, 0].set(title=f"Pro Punkt werden Odds mit {price_odds_ratio:.3f} multipliziert",
                   xlabel="Preiserhöhung (Prozentpunkte)", ylabel="Odds der Verlängerung")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot([0, 1], [0, 1], color="#595959", linestyle="--",
                    label="Perfekte Kalibrierung")
    axes[1, 1].scatter(calibration["mean_predicted_probability"],
                       calibration["observed_renewal_rate"], color="#4472C4", s=58)
    label_offsets = {8: (-10, 7), 9: (-8, -13), 10: (6, 0)}
    for _, row in calibration.iterrows():
        decile = int(row["prediction_decile"])
        offset = label_offsets.get(decile, (4, 4))
        axes[1, 1].annotate(str(int(row["prediction_decile"])),
                            (row["mean_predicted_probability"], row["observed_renewal_rate"]),
                            xytext=offset, textcoords="offset points", fontsize=8)
    axes[1, 1].set(title="Kalibrierung nach Vorhersagedezil",
                   xlabel="Mittlere vorhergesagte Wahrscheinlichkeit",
                   ylabel="Beobachteter Anteil", xlim=(0, 1.03), ylim=(0, 1.03))
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Logit verbindet lineare Log-Odds mit begrenzten Wahrscheinlichkeiten")
    figure.tight_layout()
    figure.savefig(RESULTS / "logit_model.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
