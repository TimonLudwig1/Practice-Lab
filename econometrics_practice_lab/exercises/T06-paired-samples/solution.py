"""Musterlösung für T06: Gepaarte Stichproben."""

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
    before = np.clip(rng.normal(loc=22.0, scale=5.0, size=60), 8, None)
    after = np.clip(before - 1.4 + rng.normal(loc=0, scale=2.0, size=60), 5, None)
    return pd.DataFrame(
        {"household_id": np.arange(1, 61), "before_kwh": before, "after_kwh": after}
    )


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    assert data["household_id"].is_unique
    assert not data[["before_kwh", "after_kwh"]].isna().any().any()
    data.to_csv(DATA / "household_energy.csv", index=False)
    before = data["before_kwh"].to_numpy()
    after = data["after_kwh"].to_numpy()
    change = after - before

    n = len(change)
    mean_change = change.mean()
    sd_change = change.std(ddof=1)
    paired_se = sd_change / np.sqrt(n)
    paired_t = mean_change / paired_se
    paired_df = n - 1
    paired_p = 2 * stats.t.sf(abs(paired_t), df=paired_df)
    critical_value = stats.t.ppf(0.975, df=paired_df)
    ci_low = mean_change - critical_value * paired_se
    ci_high = mean_change + critical_value * paired_se
    paired_scipy = stats.ttest_rel(after, before)

    independent = stats.ttest_ind(after, before, equal_var=False)
    independent_se = np.sqrt(after.var(ddof=1) / n + before.var(ddof=1) / n)
    rng = np.random.default_rng(RNG_SEED + 1)
    shuffled_change = rng.permutation(after) - before
    correlation = np.corrcoef(before, after)[0, 1]

    result = pd.DataFrame(
        [
            {"analysis": "Korrekt gepaart", "estimate_after_minus_before": mean_change,
             "standard_error": paired_se, "t_statistic": paired_t,
             "p_value_two_sided": paired_p, "ci_95_low": ci_low, "ci_95_high": ci_high,
             "sd_of_differences": sd_change},
            {"analysis": "Fälschlich unabhängig", "estimate_after_minus_before": mean_change,
             "standard_error": independent_se, "t_statistic": independent.statistic,
             "p_value_two_sided": independent.pvalue, "ci_95_low": np.nan,
             "ci_95_high": np.nan, "sd_of_differences": shuffled_change.std(ddof=1)},
        ]
    )
    result.to_csv(RESULTS / "paired_results.csv", index=False)
    print(f"Korrelation vorher/nachher: {correlation:.4f}")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSciPy-Kontrolle gepaart: t={paired_scipy.statistic:.4f}, "
          f"p={paired_scipy.pvalue:.4g}")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for _, row in data.iloc[:25].iterrows():
        axes[0].plot([0, 1], [row["before_kwh"], row["after_kwh"]],
                     color="#A5A5A5", alpha=0.65)
    axes[0].scatter(np.zeros(25), data.loc[:24, "before_kwh"], color="#4472C4", label="Vorher")
    axes[0].scatter(np.ones(25), data.loc[:24, "after_kwh"], color="#ED7D31", label="Nachher")
    axes[0].set(title="25 verbundene Haushalte", xticks=[0, 1],
                xticklabels=["Vorher", "Nachher"], ylabel="kWh pro Tag")

    axes[1].scatter(before, after, color="#5B9BD5", alpha=0.8)
    limits = [min(before.min(), after.min()), max(before.max(), after.max())]
    axes[1].plot(limits, limits, color="#C00000", linestyle="--", label="Keine Änderung")
    axes[1].set(title=f"Paarstruktur (r={correlation:.2f})", xlabel="Vorher (kWh)",
                ylabel="Nachher (kWh)")
    axes[1].legend(frameon=False)

    shared_bins = np.linspace(min(change.min(), shuffled_change.min()),
                              max(change.max(), shuffled_change.max()), 20)
    axes[2].hist(change, bins=shared_bins, alpha=0.7, color="#4472C4",
                 label="Korrekte Paare")
    axes[2].hist(shuffled_change, bins=shared_bins, alpha=0.55, color="#ED7D31",
                 label="Nachher gemischt")
    axes[2].axvline(0, color="black", linestyle="--")
    axes[2].set(title="Individuelle Differenzen", xlabel="Nachher − vorher (kWh)",
                ylabel="Häufigkeit")
    axes[2].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(RESULTS / "paired_structure.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
