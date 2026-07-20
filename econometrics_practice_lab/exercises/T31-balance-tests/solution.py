"""Musterlösung für T31: Balance Tests."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720
BASELINES = ["age", "prior_spend_eur", "prior_visits", "baseline_conversion",
             "mobile_user", "newsletter_subscriber"]


def make_data(size: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    treatment = np.zeros(size, dtype=int)
    treatment[rng.choice(size, size=size // 2, replace=False)] = 1
    return pd.DataFrame(
        {"user_id": np.arange(1, size + 1), "age": np.clip(rng.normal(39, 11, size), 18, 75),
         "prior_spend_eur": rng.lognormal(np.log(80), 0.65, size),
         "prior_visits": rng.poisson(6, size),
         "baseline_conversion": rng.binomial(1, 0.30, size),
         "mobile_user": rng.binomial(1, 0.66, size),
         "newsletter_subscriber": rng.binomial(1, 0.46, size), "treatment": treatment}
    )


def balance_row(data: pd.DataFrame, variable: str, assignment: np.ndarray) -> dict[str, float | str]:
    treated = data.loc[assignment == 1, variable]
    control = data.loc[assignment == 0, variable]
    pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    test = stats.ttest_ind(treated, control, equal_var=False)
    return {"variable": variable, "treatment_mean": treated.mean(),
            "control_mean": control.mean(), "raw_difference": treated.mean() - control.mean(),
            "standardized_mean_difference": (treated.mean() - control.mean()) / pooled_sd,
            "t_statistic": test.statistic, "p_value": test.pvalue}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "online_experiment_baseline.csv", index=False)
    assignment = data["treatment"].to_numpy()
    balance = pd.DataFrame([balance_row(data, variable, assignment) for variable in BASELINES])
    balance["abs_smd_above_0_10"] = balance["standardized_mean_difference"].abs() > 0.10
    balance["p_below_0_05"] = balance["p_value"] < 0.05
    balance.to_csv(RESULTS / "balance_table.csv", index=False)

    standardized = (data[BASELINES] - data[BASELINES].mean()) / data[BASELINES].std(ddof=1)
    joint_model = sm.OLS(data["treatment"], sm.add_constant(standardized)).fit()
    joint = pd.DataFrame(
        [{"f_statistic": joint_model.fvalue, "df_numerator": int(joint_model.df_model),
          "df_denominator": int(joint_model.df_resid), "p_value": joint_model.f_pvalue,
          "r_squared": joint_model.rsquared}]
    )
    joint.to_csv(RESULTS / "joint_balance_test.csv", index=False)

    rng = np.random.default_rng(RNG_SEED + 1)
    simulation_rows = []
    for repetition in range(800):
        simulated_assignment = np.zeros(len(data), dtype=int)
        simulated_assignment[rng.choice(len(data), len(data) // 2, replace=False)] = 1
        p_values = [
            stats.ttest_ind(data.loc[simulated_assignment == 1, variable],
                            data.loc[simulated_assignment == 0, variable],
                            equal_var=False).pvalue
            for variable in BASELINES
        ]
        simulation_rows.append(
            {"repetition": repetition + 1, "minimum_p_value": min(p_values),
             "number_p_below_0_05": sum(value < 0.05 for value in p_values),
             "any_p_below_0_05": any(value < 0.05 for value in p_values)}
        )
    simulation = pd.DataFrame(simulation_rows)
    simulation.to_csv(RESULTS / "randomization_multiplicity.csv", index=False)
    share_any = simulation["any_p_below_0_05"].mean()
    assert data["treatment"].sum() == len(data) // 2
    assert 0.15 < share_any < 0.40
    print(balance.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nGemeinsamer Test:")
    print(joint.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nAnteil korrekter Randomisierungen mit mindestens einem p<0,05: {share_any:.3f}")

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    positions = np.arange(len(balance))
    smd = balance["standardized_mean_difference"].to_numpy()
    axes[0, 0].scatter(smd, positions, color="#4472C4", s=55)
    axes[0, 0].axvline(0, color="#595959", linewidth=1)
    axes[0, 0].axvline(-0.1, color="#C00000", linestyle="--")
    axes[0, 0].axvline(0.1, color="#C00000", linestyle="--")
    axes[0, 0].set(title="Standardisierte Baseline-Differenzen", xlabel="Treatment − Kontrolle (SMD)",
                   yticks=positions, yticklabels=balance["variable"])

    axes[0, 1].scatter(balance["p_value"], positions, color="#4472C4", s=55)
    axes[0, 1].axvline(0.05, color="#C00000", linestyle="--", label="0,05")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set(title="Einzelne Balance-p-Werte", xlabel="p-Wert (log-Skala)",
                   yticks=positions, yticklabels=balance["variable"], xlim=(0.005, 1))
    axes[0, 1].legend(frameon=False)

    counts = simulation["number_p_below_0_05"].value_counts().sort_index()
    axes[1, 0].bar(counts.index.astype(str), counts.values, color="#4472C4")
    axes[1, 0].set(title="Falsche Alarme trotz korrekter Randomisierung",
                   xlabel="Zahl der sechs Tests mit p<0,05", ylabel="Randomisierungen")

    axes[1, 1].hist(simulation["minimum_p_value"], bins=35, color="#9DC3E6",
                    edgecolor="white")
    axes[1, 1].axvline(0.05, color="#C00000", linestyle="--", label="0,05")
    axes[1, 1].set(title=f"Mindestens ein p<0,05 in {100 * share_any:.1f}% der Zuweisungen",
                   xlabel="Kleinster p-Wert aus sechs Tests", ylabel="Häufigkeit")
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Balance Tests diagnostizieren die Zuweisung, beweisen aber keine Perfektion")
    figure.tight_layout()
    figure.savefig(RESULTS / "balance_tests.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
