"""Musterlösung für T11: Log-Transformationen und ihr Definitionsbereich."""

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


def make_data(size: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    employees = np.maximum(1, np.rint(rng.lognormal(mean=4.0, sigma=1.0, size=size))).astype(int)
    revenue = 40_000 * employees ** 0.82 * np.exp(rng.normal(0, 0.42, size=size))
    profit = 0.07 * revenue + rng.normal(0, 250_000, size=size)
    return pd.DataFrame(
        {"firm_id": np.arange(1, size + 1), "employees": employees,
         "revenue_eur": revenue, "profit_eur": profit}
    )


def describe_scale(values: pd.Series, variable: str, scale: str) -> dict[str, float | str]:
    return {
        "variable": variable,
        "scale": scale,
        "n": len(values),
        "mean": values.mean(),
        "median": values.median(),
        "sd": values.std(ddof=1),
        "skewness": stats.skew(values, bias=False),
    }


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    assert (data[["employees", "revenue_eur"]] > 0).all().all()
    data["log_employees"] = np.log(data["employees"])
    data["log_revenue"] = np.log(data["revenue_eur"])
    data["log_profit_if_positive"] = np.where(
        data["profit_eur"] > 0, np.log(data["profit_eur"].clip(lower=1e-12)), np.nan
    )
    data.to_csv(DATA / "firm_scale_data.csv", index=False)

    rows = []
    for raw, logged in (("employees", "log_employees"), ("revenue_eur", "log_revenue")):
        rows.append(describe_scale(data[raw], raw, "raw"))
        rows.append(describe_scale(data[logged], raw, "natural_log"))
    transformation_summary = pd.DataFrame(rows)
    transformation_summary.to_csv(RESULTS / "transformation_summary.csv", index=False)

    nonpositive_profit = int((data["profit_eur"] <= 0).sum())
    domain_summary = pd.DataFrame(
        [{"total_firms": len(data), "positive_profit": len(data) - nonpositive_profit,
          "nonpositive_profit": nonpositive_profit,
          "share_lost_if_positive_only": nonpositive_profit / len(data)}]
    )
    domain_summary.to_csv(RESULTS / "log_domain_summary.csv", index=False)
    print(transformation_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nLog-Domain Gewinn:")
    print(domain_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_specs = [
        ("employees", "Beschäftigte: Rohskala", "Beschäftigte"),
        ("log_employees", "Beschäftigte: Log-Skala", "log(Beschäftigte)"),
        ("revenue_eur", "Umsatz: Rohskala", "Umsatz (€)"),
        ("log_revenue", "Umsatz: Log-Skala", "log(Umsatz)"),
    ]
    colors = ["#4472C4", "#70AD47", "#4472C4", "#70AD47"]
    for axis, (column, title, xlabel), color in zip(axes.flat, plot_specs, colors):
        values = data[column]
        axis.hist(values, bins=30, color=color, edgecolor="white")
        axis.axvline(values.mean(), color="#C00000", linestyle="--", label="Mittelwert")
        axis.axvline(values.median(), color="#595959", linestyle=":", label="Median")
        axis.set(title=f"{title}\nSchiefe = {stats.skew(values, bias=False):.2f}",
                 xlabel=xlabel, ylabel="Häufigkeit")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Logarithmieren komprimiert multiplikative Größenunterschiede")
    figure.tight_layout()
    figure.savefig(RESULTS / "log_distributions.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
