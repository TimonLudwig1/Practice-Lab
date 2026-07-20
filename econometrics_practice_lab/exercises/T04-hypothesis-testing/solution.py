"""Musterlösung für T04: Teststatistik, kritischer Wert und p-Wert."""

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
    times = np.clip(rng.normal(loc=31.6, scale=5.8, size=80), 12, None)
    return pd.DataFrame({"delivery_id": np.arange(1, len(times) + 1),
                         "delivery_minutes": times})


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "delivery_times.csv", index=False)
    values = data["delivery_minutes"].to_numpy()
    null_mean = 30.0
    alpha = 0.05

    sample_mean = values.mean()
    sample_sd = values.std(ddof=1)
    standard_error = sample_sd / np.sqrt(len(values))
    t_statistic = (sample_mean - null_mean) / standard_error
    df = len(values) - 1
    critical_value = stats.t.ppf(1 - alpha / 2, df=df)
    p_value = 2 * stats.t.sf(abs(t_statistic), df=df)
    ci_low = sample_mean - critical_value * standard_error
    ci_high = sample_mean + critical_value * standard_error
    scipy_result = stats.ttest_1samp(values, popmean=null_mean)
    reject = abs(t_statistic) > critical_value

    result = pd.DataFrame(
        [{"n": len(values), "null_mean": null_mean, "sample_mean": sample_mean,
          "sample_sd": sample_sd, "standard_error": standard_error,
          "t_statistic": t_statistic, "degrees_of_freedom": df,
          "critical_value_two_sided": critical_value, "p_value_two_sided": p_value,
          "ci_95_low": ci_low, "ci_95_high": ci_high, "reject_h0": reject}]
    )
    result.to_csv(RESULTS / "test_results.csv", index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSciPy-Kontrolle: t={scipy_result.statistic:.4f}, p={scipy_result.pvalue:.4f}")
    decision = "H0 ablehnen" if reject else "H0 nicht ablehnen"
    print(f"Entscheidung bei alpha={alpha:.2f}: {decision}.")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(values, bins=14, color="#5B9BD5", edgecolor="white")
    axes[0].axvline(null_mean, color="#C00000", linestyle="--", linewidth=2,
                    label="H0-Mittelwert")
    axes[0].axvline(sample_mean, color="black", linestyle=":", linewidth=2,
                    label="Stichprobenmittel")
    axes[0].set(title="Beobachtete Lieferzeiten", xlabel="Minuten", ylabel="Häufigkeit")
    axes[0].legend(frameon=False)

    x_values = np.linspace(-4, 4, 500)
    density = stats.t.pdf(x_values, df=df)
    axes[1].plot(x_values, density, color="#4472C4", linewidth=2)
    left = x_values <= -critical_value
    right = x_values >= critical_value
    axes[1].fill_between(x_values[left], density[left], color="#C00000", alpha=0.35)
    axes[1].fill_between(x_values[right], density[right], color="#C00000", alpha=0.35,
                         label="Ablehnungsbereiche")
    axes[1].axvline(t_statistic, color="black", linewidth=2, label="Beobachtetes t")
    axes[1].axvline(-critical_value, color="#C00000", linestyle="--")
    axes[1].axvline(critical_value, color="#C00000", linestyle="--")
    axes[1].set(title="Testentscheidung unter der t-Nullverteilung",
                xlabel="t-Wert", ylabel="Dichte")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(RESULTS / "test_decision.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
