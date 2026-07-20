"""Musterlösung für T30: Randomisierte Experimente und RCTs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720
BASELINES = ["baseline_score", "motivation_score", "digital_access"]


def make_data(size: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    baseline = rng.normal(60, 10, size=size)
    motivation = rng.normal(size=size)
    digital_access = rng.binomial(1, 0.72, size=size)
    outcome_zero = 20 + 0.72 * baseline + 3 * motivation + 2 * digital_access + rng.normal(0, 7, size=size)
    effect = 5 + 0.8 * motivation
    outcome_one = outcome_zero + effect
    treatment = np.zeros(size, dtype=int)
    treatment[rng.choice(size, size=size // 2, replace=False)] = 1
    observed = np.where(treatment == 1, outcome_one, outcome_zero)
    return pd.DataFrame(
        {"student_id": np.arange(1, size + 1), "baseline_score": baseline,
         "motivation_score": motivation, "digital_access": digital_access,
         "treatment": treatment, "outcome_y0": outcome_zero, "outcome_y1": outcome_one,
         "observed_exam_score": observed}
    )


def standardized_difference(data: pd.DataFrame, variable: str) -> float:
    treated = data.loc[data["treatment"] == 1, variable]
    control = data.loc[data["treatment"] == 0, variable]
    pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    return (treated.mean() - control.mean()) / pooled_sd


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "randomized_coaching.csv", index=False)
    treated = data["treatment"] == 1
    sample_ate = (data["outcome_y1"] - data["outcome_y0"]).mean()
    estimate = data.loc[treated, "observed_exam_score"].mean() - data.loc[~treated, "observed_exam_score"].mean()
    assert data["treatment"].sum() == len(data) // 2
    assert np.allclose(
        data["observed_exam_score"],
        np.where(data["treatment"] == 1, data["outcome_y1"], data["outcome_y0"])
    )

    balance = pd.DataFrame(
        [{"variable": variable,
          "treatment_mean": data.loc[treated, variable].mean(),
          "control_mean": data.loc[~treated, variable].mean(),
          "standardized_difference": standardized_difference(data, variable)}
         for variable in BASELINES]
    )
    balance.to_csv(RESULTS / "assignment_balance.csv", index=False)

    rng = np.random.default_rng(RNG_SEED + 1)
    y0 = data["outcome_y0"].to_numpy()
    y1 = data["outcome_y1"].to_numpy()
    repeated_rows = []
    for repetition in range(1500):
        assignment = np.zeros(len(data), dtype=bool)
        assignment[rng.choice(len(data), len(data) // 2, replace=False)] = True
        repeated_estimate = y1[assignment].mean() - y0[~assignment].mean()
        repeated_rows.append({"repetition": repetition + 1, "difference_in_means": repeated_estimate})
    repeated = pd.DataFrame(repeated_rows)
    repeated["running_mean"] = repeated["difference_in_means"].expanding().mean()
    repeated.to_csv(RESULTS / "repeated_randomization.csv", index=False)
    summary = pd.DataFrame(
        [{"sample_ate": sample_ate, "observed_difference": estimate,
          "observed_randomization_error": estimate - sample_ate,
          "mean_over_randomizations": repeated["difference_in_means"].mean(),
          "randomization_sd": repeated["difference_in_means"].std(ddof=1),
          "q025": repeated["difference_in_means"].quantile(0.025),
          "q975": repeated["difference_in_means"].quantile(0.975)}]
    )
    assert abs(repeated["difference_in_means"].mean() - sample_ate) < 0.1
    summary.to_csv(RESULTS / "randomization_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBaseline-Balance der realisierten Zuweisung:")
    print(balance.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    positions = np.arange(len(balance))
    axes[0, 0].scatter(balance["standardized_difference"], positions, color="#4472C4", s=55)
    axes[0, 0].axvline(0, color="#595959", linewidth=1)
    axes[0, 0].axvline(-0.1, color="#C00000", linestyle="--")
    axes[0, 0].axvline(0.1, color="#C00000", linestyle="--", label="±0,10 Orientierung")
    axes[0, 0].set(title="Eine Zuweisung kann zufällig ungleich sein",
                   xlabel="Standardisierte Mittelwertsdifferenz", yticks=positions,
                   yticklabels=balance["variable"])
    axes[0, 0].legend(frameon=False)

    means = [data.loc[~treated, "observed_exam_score"].mean(), data.loc[treated, "observed_exam_score"].mean()]
    ses = [data.loc[~treated, "observed_exam_score"].sem(), data.loc[treated, "observed_exam_score"].sem()]
    axes[0, 1].errorbar(means, [0, 1], xerr=1.96 * np.array(ses), fmt="o",
                        color="#4472C4", capsize=5)
    axes[0, 1].set(title=f"Realisierter Unterschied = {estimate:.2f}",
                   xlabel="Mittlerer Examensscore mit 95%-KI", yticks=[0, 1],
                   yticklabels=["Kontrolle", "Treatment"])

    axes[1, 0].hist(repeated["difference_in_means"], bins=40, color="#9DC3E6",
                    edgecolor="white")
    axes[1, 0].axvline(sample_ate, color="#C00000", linestyle="--",
                       label=f"Sample ATE = {sample_ate:.2f}")
    axes[1, 0].axvline(estimate, color="#595959", linewidth=2,
                       label=f"Realisierte Schätzung = {estimate:.2f}")
    axes[1, 0].set(title="Schätzwerte über wiederholte Randomisierung",
                   xlabel="Treatment minus Kontrolle", ylabel="Häufigkeit")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(repeated["repetition"], repeated["running_mean"], color="#4472C4")
    axes[1, 1].axhline(sample_ate, color="#C00000", linestyle="--", label="Sample ATE")
    axes[1, 1].set(title="Im Mittel trifft Randomisierung den Sample ATE",
                   xlabel="Anzahl Randomisierungen", ylabel="Laufender Mittelwert")
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Randomisierung erzeugt Vergleichbarkeit im Erwartungswert")
    figure.tight_layout()
    figure.savefig(RESULTS / "randomized_experiment.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
