"""Musterlösung für T08: Pearson-Korrelation und ihre Grenzen."""

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
RNG_SEED = 20260720


def make_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)

    x_linear = rng.uniform(0, 10, size=120)
    y_linear = 3 + 1.7 * x_linear + rng.normal(0, 3, size=120)

    x_nonlinear = np.linspace(-3, 3, 120)
    y_nonlinear = x_nonlinear ** 2 + rng.normal(0, 0.65, size=120)

    x_outlier = np.append(rng.normal(size=80), 8.0)
    y_outlier = np.append(rng.normal(size=80), 8.0)

    frames = []
    for scenario, x, y in (
        ("linear", x_linear, y_linear),
        ("nonlinear", x_nonlinear, y_nonlinear),
        ("outlier", x_outlier, y_outlier),
    ):
        frames.append(
            pd.DataFrame(
                {"scenario": scenario, "observation_id": np.arange(1, len(x) + 1),
                 "x": x, "y": y}
            )
        )
    return pd.concat(frames, ignore_index=True)


def pearson_manual(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x und y müssen gleich lang sein und mindestens zwei Werte enthalten.")
    z_x = (x - x.mean()) / x.std(ddof=1)
    z_y = (y - y.mean()) / y.std(ddof=1)
    return float(np.sum(z_x * z_y) / (len(x) - 1))


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    assert not data.isna().any().any()
    data.to_csv(DATA / "correlation_scenarios.csv", index=False)

    rows = []
    scenario_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scenario, group in data.groupby("scenario", sort=False):
        x = group["x"].to_numpy()
        y = group["y"].to_numpy()
        scenario_arrays[scenario] = (x, y)
        scipy_result = stats.pearsonr(x, y)
        manual_r = pearson_manual(x, y)
        covariance_r = np.cov(x, y, ddof=1)[0, 1] / (x.std(ddof=1) * y.std(ddof=1))
        assert np.isclose(manual_r, scipy_result.statistic)
        assert np.isclose(manual_r, covariance_r)
        rows.append(
            {"scenario": scenario, "n": len(x), "covariance": np.cov(x, y, ddof=1)[0, 1],
             "pearson_r": manual_r, "p_value": scipy_result.pvalue,
             "r_without_marked_outlier": np.nan}
        )

    x_linear, y_linear = scenario_arrays["linear"]
    transformed_correlations = {
        "Original X": pearson_manual(x_linear, y_linear),
        "1000 X + 50": pearson_manual(1_000 * x_linear + 50, y_linear),
        "−X": pearson_manual(-x_linear, y_linear),
    }
    transformed_covariances = {
        "Original X": np.cov(x_linear, y_linear, ddof=1)[0, 1],
        "1000 X + 50": np.cov(1_000 * x_linear + 50, y_linear, ddof=1)[0, 1],
    }
    x_outlier, y_outlier = scenario_arrays["outlier"]
    r_without_outlier = pearson_manual(x_outlier[:-1], y_outlier[:-1])
    for row in rows:
        if row["scenario"] == "outlier":
            row["r_without_marked_outlier"] = r_without_outlier

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "correlation_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTransformationen des linearen Szenarios:")
    for label, value in transformed_correlations.items():
        print(f"{label:>12}: r={value:+.6f}")
    covariance_factor = transformed_covariances["1000 X + 50"] / transformed_covariances["Original X"]
    print(f"Kovarianzfaktor nach 1000 X + 50: {covariance_factor:.1f}")

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    labels = {
        "linear": "Linearer Zusammenhang",
        "nonlinear": "U-förmiger Zusammenhang",
        "outlier": "Ausreißergetriebene Korrelation",
    }
    summary_by_scenario = summary.set_index("scenario")
    for axis, scenario in zip(axes.flat[:3], ("linear", "nonlinear", "outlier")):
        x, y = scenario_arrays[scenario]
        if scenario == "outlier":
            axis.scatter(x[:-1], y[:-1], color="#5B9BD5", alpha=0.72, s=28,
                         label="Reguläre Beobachtungen")
            axis.scatter(x[-1], y[-1], color="#C00000", marker="D", s=75,
                         label="Markierter Extremwert")
            axis.legend(frameon=False, fontsize=8)
        else:
            axis.scatter(x, y, color="#5B9BD5", alpha=0.72, s=28)
        r_value = summary_by_scenario.loc[scenario, "pearson_r"]
        axis.set(title=f"{labels[scenario]}\nr = {r_value:+.3f}", xlabel="X", ylabel="Y")

    transform_labels = list(transformed_correlations)
    transform_values = list(transformed_correlations.values())
    colors = ["#4472C4", "#70AD47", "#ED7D31"]
    bars = axes[1, 1].bar(transform_labels, transform_values, color=colors)
    axes[1, 1].axhline(0, color="#595959", linewidth=1)
    axes[1, 1].set(title="Einfluss linearer Transformationen", ylabel="Pearson-r",
                   ylim=(-1, 1))
    for bar, value in zip(bars, transform_values):
        offset = 0.04 if value >= 0 else -0.08
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, value + offset,
                        f"{value:+.3f}", ha="center", va="bottom")

    figure.suptitle("Gleiche Kennzahl, unterschiedliche Datenstruktur")
    figure.tight_layout()
    figure.savefig(RESULTS / "correlation_pitfalls.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
