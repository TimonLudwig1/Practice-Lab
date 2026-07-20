"""Musterlösung für T25: Binäre abhängige Variablen."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from statsmodels.stats.proportion import proportion_confint


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    income = np.clip(rng.lognormal(np.log(52), 0.45, size=size), 15, 160)
    debt_to_income = np.clip(rng.beta(2.2, 4.5, size=size) * 0.9, 0.02, 0.88)
    late_payments = np.clip(rng.poisson(0.8, size=size), 0, 4)
    linear_index = (
        -3.5 + 4.8 * debt_to_income + 0.58 * late_payments
        - 0.012 * (income - 50)
    )
    true_probability = expit(linear_index)
    defaulted = rng.binomial(1, true_probability)
    return pd.DataFrame(
        {"loan_id": np.arange(1, size + 1), "income_thousand_eur": income,
         "debt_to_income": debt_to_income, "prior_late_payments": late_payments,
         "defaulted": defaulted, "oracle_default_probability": true_probability}
    )


def grouped_rate(data: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    for value, subset in data.groupby(group, observed=True):
        successes = int(subset["defaulted"].sum())
        low, high = proportion_confint(successes, len(subset), alpha=0.05, method="wilson")
        rows.append(
            {group: value, "observations": len(subset), "defaults": successes,
             "default_rate": subset["defaulted"].mean(), "ci_95_low": low,
             "ci_95_high": high}
        )
    return pd.DataFrame(rows)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "loan_defaults.csv", index=False)

    observed_values = set(data["defaulted"].unique())
    event_rate = data["defaulted"].mean()
    empirical_variance = data["defaulted"].var(ddof=0)
    bernoulli_variance = event_rate * (1 - event_rate)
    majority_accuracy = max(event_rate, 1 - event_rate)
    constant_brier = np.mean((data["defaulted"] - event_rate) ** 2)
    oracle_brier = np.mean(
        (data["defaulted"] - data["oracle_default_probability"]) ** 2
    )
    summary = pd.DataFrame(
        [{"observations": len(data), "event_rate": event_rate,
          "mean_of_binary_outcome": data["defaulted"].mean(),
          "empirical_variance_ddof0": empirical_variance,
          "p_times_one_minus_p": bernoulli_variance,
          "majority_class_accuracy": majority_accuracy,
          "constant_probability_brier": constant_brier,
          "oracle_probability_brier": oracle_brier}]
    )
    assert observed_values == {0, 1}
    assert np.isclose(empirical_variance, bernoulli_variance)
    assert oracle_brier < constant_brier
    summary.to_csv(RESULTS / "event_summary.csv", index=False)

    data["risk_decile"] = pd.qcut(
        data["oracle_default_probability"], q=10, labels=False, duplicates="drop"
    ) + 1
    risk_deciles = (
        data.groupby("risk_decile", observed=True)
        .agg(observations=("defaulted", "size"), defaults=("defaulted", "sum"),
             mean_oracle_probability=("oracle_default_probability", "mean"),
             observed_default_rate=("defaulted", "mean"))
        .reset_index()
    )
    risk_deciles.to_csv(RESULTS / "risk_deciles.csv", index=False)
    late_payment_rates = grouped_rate(data, "prior_late_payments")
    late_payment_rates.to_csv(RESULTS / "late_payment_rates.csv", index=False)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nAusfallquoten nach früheren Zahlungsverzügen:")
    print(late_payment_rates.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    rng = np.random.default_rng(RNG_SEED + 1)
    jittered = data["defaulted"] + rng.normal(0, 0.025, size=len(data))
    axes[0, 0].scatter(data["oracle_default_probability"], jittered, color="#9DC3E6",
                       alpha=0.35, s=16, label="Einzeloutcome (mit Jitter)")
    axes[0, 0].plot([0, 1], [0, 1], color="#595959", linestyle="--",
                    label="Wahrscheinlichkeit = Anteil")
    axes[0, 0].scatter(risk_deciles["mean_oracle_probability"],
                       risk_deciles["observed_default_rate"], color="#C00000", s=50,
                       label="Beobachteter Anteil je Dezil")
    axes[0, 0].set(title="Ein Ergebnis ist binär, der bedingte Mittelwert nicht",
                   xlabel="Datengenerierende Ausfallwahrscheinlichkeit",
                   ylabel="Ausfall (0/1) beziehungsweise Anteil", xlim=(-0.02, 1.02),
                   ylim=(-0.08, 1.08))
    axes[0, 0].legend(frameon=False, fontsize=8)

    positions = np.arange(len(late_payment_rates))
    rates = late_payment_rates["default_rate"].to_numpy()
    left = rates - late_payment_rates["ci_95_low"].to_numpy()
    right = late_payment_rates["ci_95_high"].to_numpy() - rates
    axes[0, 1].errorbar(rates, positions, xerr=np.vstack([left, right]), fmt="o",
                        color="#4472C4", capsize=5)
    axes[0, 1].set(title="Ausfallanteile mit 95%-Wilson-Intervall",
                   xlabel="Ausfallanteil", ylabel="Frühere Zahlungsverzüge",
                   yticks=positions,
                   yticklabels=late_payment_rates["prior_late_payments"].astype(int),
                   xlim=(0, 1))

    probability_grid = np.linspace(0, 1, 300)
    axes[1, 0].plot(probability_grid, probability_grid * (1 - probability_grid),
                    color="#4472C4", linewidth=2)
    axes[1, 0].scatter([event_rate], [empirical_variance], color="#C00000", s=60,
                       label=f"Stichprobe: p̄={event_rate:.3f}")
    axes[1, 0].set(title="Bernoulli-Varianz ist p(1−p)", xlabel="Wahrscheinlichkeit p",
                   ylabel="Varianz", xlim=(0, 1), ylim=(0, 0.27))
    axes[1, 0].legend(frameon=False)

    axes[1, 1].hist(
        [data.loc[data["defaulted"] == 0, "oracle_default_probability"],
         data.loc[data["defaulted"] == 1, "oracle_default_probability"]],
        bins=np.linspace(0, 1, 25), density=True, alpha=0.68, stacked=False,
        color=["#4472C4", "#ED7D31"], label=["Kein Ausfall", "Ausfall"]
    )
    axes[1, 1].set(title="Risikoverteilungen überlappen",
                   xlabel="Datengenerierende Ausfallwahrscheinlichkeit", ylabel="Dichte",
                   xlim=(0, 1))
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Binäre Outcomes verbinden Einzelereignisse mit Wahrscheinlichkeiten")
    figure.tight_layout()
    figure.savefig(RESULTS / "binary_outcomes.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
