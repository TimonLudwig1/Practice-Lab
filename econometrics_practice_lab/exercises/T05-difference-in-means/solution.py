"""Musterlösung für T05: Test einer Mittelwertsdifferenz unabhängiger Gruppen."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260714


def make_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    score_a = np.clip(rng.normal(loc=71.5, scale=10.0, size=110), 0, 100)
    score_b = np.clip(rng.normal(loc=76.0, scale=13.5, size=95), 0, 100)
    return pd.DataFrame(
        {"student_id": np.arange(1, 206),
         "group": np.repeat(["A_alt", "B_neu"], [len(score_a), len(score_b)]),
         "score": np.concatenate([score_a, score_b])}
    )


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    assert data["student_id"].is_unique
    data.to_csv(DATA / "quiz_scores.csv", index=False)
    a = data.loc[data["group"] == "A_alt", "score"].to_numpy()
    b = data.loc[data["group"] == "B_neu", "score"].to_numpy()

    n_a, n_b = len(a), len(b)
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    difference = mean_b - mean_a
    se_squared = var_a / n_a + var_b / n_b
    standard_error = np.sqrt(se_squared)
    df = se_squared ** 2 / (
        (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    )
    t_statistic = difference / standard_error
    p_value = 2 * stats.t.sf(abs(t_statistic), df=df)
    critical_value = stats.t.ppf(0.975, df=df)
    ci_low = difference - critical_value * standard_error
    ci_high = difference + critical_value * standard_error
    scipy_result = stats.ttest_ind(b, a, equal_var=False)

    result = pd.DataFrame(
        [{"n_a": n_a, "mean_a": mean_a, "sd_a": np.sqrt(var_a),
          "n_b": n_b, "mean_b": mean_b, "sd_b": np.sqrt(var_b),
          "difference_b_minus_a": difference, "standard_error": standard_error,
          "welch_df": df, "t_statistic": t_statistic, "p_value_two_sided": p_value,
          "ci_95_low": ci_low, "ci_95_high": ci_high}]
    )
    result.to_csv(RESULTS / "welch_test_results.csv", index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSciPy-Kontrolle: t={scipy_result.statistic:.4f}, p={scipy_result.pvalue:.4f}")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(min(data["score"]), max(data["score"]), 20)
    axes[0].hist(a, bins=bins, alpha=0.65, color="#4472C4", label="A: alt")
    axes[0].hist(b, bins=bins, alpha=0.65, color="#ED7D31", label="B: neu")
    axes[0].set(title="Score-Verteilungen", xlabel="Testscore", ylabel="Häufigkeit")
    axes[0].legend(frameon=False)

    axes[1].errorbar(difference, 0, xerr=[[difference - ci_low], [ci_high - difference]],
                     fmt="o", color="#4472C4", capsize=6, markersize=7)
    axes[1].axvline(0, color="#C00000", linestyle="--", label="Kein Unterschied")
    axes[1].set(title="Mittelwertsdifferenz mit 95%-KI", xlabel="Scorepunkte (B − A)",
                yticks=[], ylim=(-1, 1))
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(RESULTS / "group_comparison.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
