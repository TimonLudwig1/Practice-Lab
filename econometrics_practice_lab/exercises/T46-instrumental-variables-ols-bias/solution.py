"""Musterlösung für T46: Instrumental Variables und OLS-Bias."""

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
RNG_SEED = 20260726
TRUE_EFFECT = 2.0


def make_data(observations: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    motivation = rng.normal(size=observations)
    invitation = rng.binomial(1, 0.5, observations)
    invalid_instrument = (0.9 * motivation + rng.normal(size=observations) > 0).astype(int)
    training = 6 + 2.2 * invitation + 1.7 * motivation + rng.normal(0, 1.7, observations)
    income = 18 + TRUE_EFFECT * training + 4.0 * motivation + rng.normal(0, 2.5, observations)
    return pd.DataFrame({"invitation": invitation, "invalid_instrument": invalid_instrument,
                         "training_hours": training, "income_thousand_eur": income,
                         "oracle_motivation": motivation})


def iv_components(data: pd.DataFrame, instrument: str) -> tuple[float, float, float]:
    first_stage = sm.OLS(data["training_hours"], sm.add_constant(data[[instrument]])).fit()
    reduced_form = sm.OLS(data["income_thousand_eur"], sm.add_constant(data[[instrument]])).fit()
    return (first_stage.params[instrument], reduced_form.params[instrument],
            reduced_form.params[instrument] / first_stage.params[instrument])


def standardized_mean_difference(data: pd.DataFrame, variable: str, group: str) -> float:
    zero = data.loc[data[group].eq(0), variable]
    one = data.loc[data[group].eq(1), variable]
    pooled_sd = np.sqrt((zero.var(ddof=1) + one.var(ddof=1)) / 2)
    return (one.mean() - zero.mean()) / pooled_sd


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "training_iv_data.csv", index=False)

    ols = sm.OLS(data["income_thousand_eur"], sm.add_constant(data[["training_hours"]])).fit()
    oracle = sm.OLS(
        data["income_thousand_eur"], sm.add_constant(data[["training_hours", "oracle_motivation"]])
    ).fit()
    valid_fs, valid_rf, valid_iv = iv_components(data, "invitation")
    invalid_fs, invalid_rf, invalid_iv = iv_components(data, "invalid_instrument")
    estimates = pd.DataFrame([
        {"method": "OLS", "effect_estimate": ols.params["training_hours"]},
        {"method": "Valid IV", "effect_estimate": valid_iv},
        {"method": "Invalid IV", "effect_estimate": invalid_iv},
        {"method": "Oracle OLS with motivation", "effect_estimate": oracle.params["training_hours"]},
        {"method": "True causal effect", "effect_estimate": TRUE_EFFECT},
    ])
    estimates.to_csv(RESULTS / "ols_iv_comparison.csv", index=False)

    balance_rows = []
    for instrument, label in [("invitation", "Valid invitation"),
                              ("invalid_instrument", "Invalid instrument")]:
        balance_rows.append({
            "instrument": label,
            "training_smd": standardized_mean_difference(data, "training_hours", instrument),
            "motivation_smd": standardized_mean_difference(data, "oracle_motivation", instrument),
        })
    balance = pd.DataFrame(balance_rows)
    balance.to_csv(RESULTS / "instrument_diagnostics.csv", index=False)
    components = pd.DataFrame([
        {"instrument": "Valid invitation", "first_stage": valid_fs,
         "reduced_form": valid_rf, "iv_ratio": valid_iv},
        {"instrument": "Invalid instrument", "first_stage": invalid_fs,
         "reduced_form": invalid_rf, "iv_ratio": invalid_iv},
    ])
    components.to_csv(RESULTS / "iv_components.csv", index=False)

    assert ols.params["training_hours"] - TRUE_EFFECT > 0.8
    assert abs(valid_iv - TRUE_EFFECT) < 0.35
    assert abs(oracle.params["training_hours"] - TRUE_EFFECT) < 0.15
    assert abs(balance.loc[0, "motivation_smd"]) < 0.15
    assert abs(balance.loc[1, "motivation_smd"]) > 0.7

    print(estimates.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFirst Stage, Reduced Form und Ratio:")
    print(components.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nStandardisierte Mittelwertsdifferenzen:")
    print(balance.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    plotted = estimates.iloc[:4]
    y_pos = np.arange(len(plotted))
    axes[0].scatter(plotted["effect_estimate"], y_pos, color="#4472C4", s=65)
    axes[0].axvline(TRUE_EFFECT, color="#C00000", linestyle="--", label="Wahrer Effekt")
    axes[0].set(title="Endogenität verzerrt OLS und ungültiges IV",
                xlabel="Effekt je Weiterbildungsstunde", yticks=y_pos,
                yticklabels=["OLS", "Gültiges IV", "Ungültiges IV", "Oracle OLS"])
    axes[0].legend(frameon=False)

    width = 0.34
    positions = np.arange(len(balance))
    axes[1].bar(positions - width / 2, balance["training_smd"], width,
                color="#4472C4", label="Weiterbildung (Relevanz)")
    axes[1].bar(positions + width / 2, balance["motivation_smd"], width,
                color="#C00000", label="Motivation (Exogenität verletzt)")
    axes[1].axhline(0, color="#595959", linewidth=1)
    axes[1].set(title="Ein gültiges Instrument verschiebt X, nicht den Confounder",
                ylabel="Standardisierte Mittelwertsdifferenz", xticks=positions,
                xticklabels=["Zufällige\nEinladung", "Ungültiges\nInstrument"])
    axes[1].legend(frameon=False)
    figure.suptitle("Instrumental Variables isolieren exogene Variation im Treatment")
    figure.tight_layout()
    figure.savefig(RESULTS / "iv_vs_ols_bias.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
