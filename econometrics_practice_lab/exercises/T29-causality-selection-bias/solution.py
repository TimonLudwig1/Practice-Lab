"""Musterlösung für T29: Kausalität und Selektionsbias."""

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


def make_data(size: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    motivation = rng.normal(size=size)
    education = np.clip(13 + 1.1 * motivation + rng.normal(0, 1.4, size=size), 9, 20)
    selection_probability = expit(-0.7 + 1.25 * motivation + 0.16 * (education - 13))
    treatment = rng.binomial(1, selection_probability)
    outcome_zero = 52 + 5.5 * motivation + 1.7 * education + rng.normal(0, 4, size=size)
    treatment_effect = 4 + 1.4 * motivation
    outcome_one = outcome_zero + treatment_effect
    observed = np.where(treatment == 1, outcome_one, outcome_zero)
    return pd.DataFrame(
        {"employee_id": np.arange(1, size + 1), "motivation_score": motivation,
         "education_years": education, "selection_probability": selection_probability,
         "treatment": treatment, "outcome_y0": outcome_zero, "outcome_y1": outcome_one,
         "individual_treatment_effect": treatment_effect,
         "observed_performance": observed}
    )


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "voluntary_training.csv", index=False)
    treated = data["treatment"] == 1
    control = ~treated
    ate = data["individual_treatment_effect"].mean()
    att = data.loc[treated, "individual_treatment_effect"].mean()
    naive = (
        data.loc[treated, "observed_performance"].mean()
        - data.loc[control, "observed_performance"].mean()
    )
    selection_bias = data.loc[treated, "outcome_y0"].mean() - data.loc[control, "outcome_y0"].mean()
    estimands = pd.DataFrame(
        [{"ate": ate, "att": att, "naive_observed_difference": naive,
          "selection_bias_in_y0": selection_bias, "att_plus_selection_bias": att + selection_bias,
          "decomposition_error": naive - att - selection_bias}]
    )
    observed_from_potential = np.where(
        data["treatment"] == 1, data["outcome_y1"], data["outcome_y0"]
    )
    assert np.allclose(observed_from_potential, data["observed_performance"])
    assert np.isclose(naive, att + selection_bias)
    assert naive > att > ate
    estimands.to_csv(RESULTS / "causal_estimands.csv", index=False)

    group_profiles = (
        data.groupby("treatment")
        .agg(observations=("employee_id", "size"), motivation_mean=("motivation_score", "mean"),
             education_mean=("education_years", "mean"), y0_mean=("outcome_y0", "mean"),
             observed_mean=("observed_performance", "mean"),
             treatment_effect_mean=("individual_treatment_effect", "mean"))
        .reset_index()
    )
    group_profiles.to_csv(RESULTS / "group_profiles.csv", index=False)
    data["motivation_bin"] = pd.qcut(data["motivation_score"], 10, labels=False) + 1
    motivation_bins = (
        data.groupby("motivation_bin")
        .agg(mean_motivation=("motivation_score", "mean"), participation_rate=("treatment", "mean"),
             mean_treatment_effect=("individual_treatment_effect", "mean"))
        .reset_index()
    )
    motivation_bins.to_csv(RESULTS / "motivation_bins.csv", index=False)
    print(estimands.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nGruppenprofile:")
    print(group_profiles.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(motivation_bins["mean_motivation"], motivation_bins["participation_rate"],
                    marker="o", color="#4472C4")
    axes[0, 0].set(title="Motivation beeinflusst die Selbstselektion",
                   xlabel="Mittlere Motivation im Dezil", ylabel="Teilnahmeanteil", ylim=(0, 1))

    axes[0, 1].boxplot(
        [data.loc[control, "outcome_y0"], data.loc[treated, "outcome_y0"]],
        tick_labels=["Kontrolle", "Teilnehmende"], patch_artist=True,
        boxprops={"facecolor": "#9DC3E6"}, medianprops={"color": "#C00000"}
    )
    axes[0, 1].set(title="Gruppen unterschieden sich schon ohne Treatment",
                   ylabel="Potenzielles Outcome Y(0)")

    axes[1, 0].bar(["Naive Differenz"], [att], color="#70AD47")
    axes[1, 0].bar(["Naive Differenz"], [selection_bias], bottom=[att], color="#ED7D31")
    axes[1, 0].text(0, att / 2, f"ATT\n{att:.2f}", ha="center", va="center")
    axes[1, 0].text(0, att + selection_bias / 2, f"Selektionsbias\n{selection_bias:.2f}",
                    ha="center", va="center")
    axes[1, 0].text(0, naive - 0.20, f"Naiv = {naive:.2f}", ha="center", va="top")
    axes[1, 0].set(title="Beobachteter Unterschied ist nicht nur Wirkung",
                   ylabel="Leistungspunkte")

    axes[1, 1].scatter(data["motivation_score"], data["individual_treatment_effect"],
                       color="#4472C4", alpha=0.3, s=15)
    axes[1, 1].plot(motivation_bins["mean_motivation"],
                    motivation_bins["mean_treatment_effect"], color="#C00000", linewidth=2,
                    marker="o", label="Mittlerer Effekt je Dezil")
    axes[1, 1].axhline(ate, color="#595959", linestyle="--", label=f"ATE = {ate:.2f}")
    axes[1, 1].set(title="Treatment-Effekte sind heterogen", xlabel="Motivation",
                   ylabel="Y(1) − Y(0)")
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Selbstselektion vermischt Ausgangsunterschiede und kausale Wirkung")
    figure.tight_layout()
    figure.savefig(RESULTS / "selection_bias.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
