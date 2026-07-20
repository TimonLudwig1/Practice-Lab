"""Musterlösung für T32: Interne und externe Validität."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 16000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    age = np.clip(rng.normal(52, 14, size=size), 18, 85)
    severity = rng.normal(size=size)
    rural = rng.binomial(1, 0.30, size=size)
    selection_probability = np.clip(
        expit(-1.15 + 0.85 * severity - 0.032 * (age - 50) - 0.45 * rural), 0.05, 0.85
    )
    participant = rng.binomial(1, selection_probability)
    treatment = np.where(participant == 1, rng.binomial(1, 0.5, size=size), np.nan)
    outcome_zero = 65 - 5 * severity - 0.08 * age - 2 * rural + rng.normal(0, 6, size=size)
    treatment_effect = 4 + 1.6 * severity + 0.035 * (age - 50)
    outcome_one = outcome_zero + treatment_effect
    observed = np.where(participant == 0, np.nan,
                        np.where(treatment == 1, outcome_one, outcome_zero))
    return pd.DataFrame(
        {"person_id": np.arange(1, size + 1), "age": age, "baseline_severity": severity,
         "rural": rural, "selection_probability": selection_probability,
         "trial_participant": participant, "treatment": treatment,
         "outcome_y0": outcome_zero, "outcome_y1": outcome_one,
         "individual_treatment_effect": treatment_effect, "observed_health_score": observed}
    )


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values, weights=weights))


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "health_trial_population.csv", index=False)
    sample = data[data["trial_participant"] == 1].copy()
    treated = sample["treatment"] == 1
    population_ate = data["individual_treatment_effect"].mean()
    sample_ate = sample["individual_treatment_effect"].mean()
    trial_estimate = (
        sample.loc[treated, "observed_health_score"].mean()
        - sample.loc[~treated, "observed_health_score"].mean()
    )
    sample["generalization_weight"] = 1 / sample["selection_probability"]
    generalized = (
        weighted_mean(sample.loc[treated, "observed_health_score"],
                      sample.loc[treated, "generalization_weight"])
        - weighted_mean(sample.loc[~treated, "observed_health_score"],
                        sample.loc[~treated, "generalization_weight"])
    )
    estimands = pd.DataFrame(
        [{"population_ate": population_ate, "sample_ate": sample_ate,
          "unweighted_trial_estimate": trial_estimate,
          "internal_estimation_error": trial_estimate - sample_ate,
          "external_validity_gap": sample_ate - population_ate,
          "total_error_for_population": trial_estimate - population_ate,
          "generalized_ipw_estimate": generalized,
          "generalized_error_for_population": generalized - population_ate}]
    )
    assert abs(trial_estimate - sample_ate) < 0.6
    assert abs(generalized - population_ate) < abs(trial_estimate - population_ate)
    estimands.to_csv(RESULTS / "validity_estimands.csv", index=False)

    characteristics = pd.DataFrame(
        [{"group": "Target population", "observations": len(data),
          "mean_age": data["age"].mean(), "mean_severity": data["baseline_severity"].mean(),
          "rural_share": data["rural"].mean()},
         {"group": "Trial participants", "observations": len(sample),
          "mean_age": sample["age"].mean(), "mean_severity": sample["baseline_severity"].mean(),
          "rural_share": sample["rural"].mean()}]
    )
    characteristics.to_csv(RESULTS / "population_sample_comparison.csv", index=False)
    data["severity_bin"] = pd.qcut(data["baseline_severity"], 10, labels=False) + 1
    bins = (
        data.groupby("severity_bin")
        .agg(mean_severity=("baseline_severity", "mean"),
             participation_rate=("trial_participant", "mean"),
             mean_treatment_effect=("individual_treatment_effect", "mean"))
        .reset_index()
    )
    bins.to_csv(RESULTS / "selection_effect_bins.csv", index=False)
    print(estimands.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPopulation und Stichprobe:")
    print(characteristics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(data["age"], bins=30, density=True, alpha=0.55, color="#4472C4",
                    label="Zielpopulation")
    axes[0, 0].hist(sample["age"], bins=30, density=True, alpha=0.55, color="#ED7D31",
                    label="Studienteilnehmende")
    axes[0, 0].set(title="Die Studie rekrutiert selektiv nach Alter", xlabel="Alter", ylabel="Dichte")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(bins["mean_severity"], bins["participation_rate"], marker="o",
                    color="#4472C4")
    axes[0, 1].set(title="Stärkere Belastung erhöht die Teilnahme",
                   xlabel="Mittlere Ausgangsbelastung", ylabel="Teilnahmeanteil", ylim=(0, 1))

    axes[1, 0].plot(bins["mean_severity"], bins["mean_treatment_effect"], marker="o",
                    color="#ED7D31")
    axes[1, 0].axhline(population_ate, color="#595959", linestyle="--",
                       label=f"Population ATE = {population_ate:.2f}")
    axes[1, 0].set(title="Wirkung hängt von Ausgangsbelastung ab",
                   xlabel="Mittlere Ausgangsbelastung", ylabel="Treatment-Effekt")
    axes[1, 0].legend(frameon=False)

    labels = ["Population ATE (wahr)", "Sample ATE (wahr)", "Trial ungewichtet", "Trial generalisiert"]
    values = [population_ate, sample_ate, trial_estimate, generalized]
    positions = np.arange(len(labels))
    axes[1, 1].scatter(values, positions, color=["#595959", "#70AD47", "#4472C4", "#ED7D31"], s=65)
    axes[1, 1].axvline(population_ate, color="#595959", linestyle="--")
    axes[1, 1].set(title="Interne Schätzung und externe Zielgröße",
                   xlabel="Mittlerer Treatment-Effekt", yticks=positions, yticklabels=labels)
    figure.suptitle("Randomisierung schützt intern, nicht automatisch bei der Übertragung")
    figure.tight_layout()
    figure.savefig(RESULTS / "internal_external_validity.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
