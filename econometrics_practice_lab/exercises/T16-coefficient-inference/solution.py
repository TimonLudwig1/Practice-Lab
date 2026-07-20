"""Musterlösung für T16: Tests und Konfidenzintervalle für OLS-Koeffizienten."""

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


def make_data(size: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    discount = rng.uniform(0, 20, size=size)
    weekly_units = 120 + 3.0 * discount + rng.normal(0, 25, size=size)
    return pd.DataFrame(
        {"store_id": np.arange(1, size + 1), "discount_percentage": discount,
         "weekly_units": weekly_units}
    )


def test_slope(estimate: float, standard_error: float, null_value: float, df: int) -> dict[str, float]:
    t_statistic = (estimate - null_value) / standard_error
    p_value = 2 * stats.t.sf(abs(t_statistic), df=df)
    return {"null_value": null_value, "t_statistic": t_statistic, "p_value_two_sided": p_value}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "store_discounts.csv", index=False)

    x = data["discount_percentage"].to_numpy()
    y = data["weekly_units"].to_numpy()
    model = sm.OLS(y, sm.add_constant(x)).fit()
    slope = model.params[1]
    sxx = np.sum((x - x.mean()) ** 2)
    sigma_squared = np.sum(model.resid ** 2) / model.df_resid
    manual_se = np.sqrt(sigma_squared / sxx)
    assert np.isclose(manual_se, model.bse[1])

    tests = pd.DataFrame([test_slope(slope, manual_se, null, int(model.df_resid))
                          for null in (0.0, 3.0)])
    tests["estimate"] = slope
    tests["standard_error"] = manual_se
    tests["reject_at_5_percent"] = tests["p_value_two_sided"] < 0.05
    tests.to_csv(RESULTS / "coefficient_tests.csv", index=False)

    intervals = []
    for confidence_level in (0.95, 0.99):
        critical = stats.t.ppf(1 - (1 - confidence_level) / 2, df=model.df_resid)
        intervals.append(
            {"confidence_level": confidence_level, "estimate": slope,
             "critical_value": critical, "ci_low": slope - critical * manual_se,
             "ci_high": slope + critical * manual_se}
        )
    intervals = pd.DataFrame(intervals)
    intervals.to_csv(RESULTS / "coefficient_intervals.csv", index=False)
    print(tests.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nKoeffizientenintervalle:")
    print(intervals.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    x_grid = np.linspace(x.min(), x.max(), 200)
    prediction = model.get_prediction(sm.add_constant(x_grid)).summary_frame(alpha=0.05)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(x, y, color="#5B9BD5", alpha=0.55, s=23)
    axes[0].plot(x_grid, prediction["mean"], color="#C00000", linewidth=2,
                 label="OLS-Mittelwert")
    axes[0].fill_between(x_grid, prediction["mean_ci_lower"], prediction["mean_ci_upper"],
                         color="#4472C4", alpha=0.2, label="95%-KI des Mittelwerts")
    axes[0].set(title="Regression mit Unsicherheit des Mittelwerts", xlabel="Rabatt (Prozentpunkte)",
                ylabel="Wöchentlicher Absatz")
    axes[0].legend(frameon=False)

    y_positions = np.array([0, 1])
    labels = ["95%-KI", "99%-KI"]
    estimates = intervals["estimate"].to_numpy()
    left = estimates - intervals["ci_low"].to_numpy()
    right = intervals["ci_high"].to_numpy() - estimates
    axes[1].errorbar(estimates, y_positions, xerr=np.vstack([left, right]), fmt="o",
                     color="#4472C4", capsize=6, markersize=7)
    axes[1].axvline(0, color="#C00000", linestyle="--", label="H₀: β₁=0")
    axes[1].axvline(3, color="#ED7D31", linestyle=":", linewidth=2, label="H₀: β₁=3")
    axes[1].set(title="Steigung und Nullhypothesen", xlabel="Zusätzliche Einheiten je Rabattpunkt",
                yticks=y_positions, yticklabels=labels, ylim=(-0.7, 1.7))
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(RESULTS / "coefficient_inference.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
