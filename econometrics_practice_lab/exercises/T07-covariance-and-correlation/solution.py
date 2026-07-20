"""Musterlösung für T07: Kovarianz, Korrelation und Maßeinheiten."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    income = np.clip(rng.normal(loc=4_200, scale=1_300, size=size), 1_200, 9_000)
    leisure = np.clip(90 + 0.11 * income + rng.normal(0, 190, size=size), 0, None)
    commute = np.clip(56 - 0.0045 * income + rng.normal(0, 11, size=size), 5, 100)
    return pd.DataFrame(
        {
            "household_id": np.arange(1, size + 1),
            "income_eur": income,
            "leisure_spending_eur": leisure,
            "commute_minutes": commute,
        }
    )


def sample_covariance(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x und y müssen gleich lang sein und mindestens zwei Werte enthalten.")
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (len(x) - 1))


def sample_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return sample_covariance(x, y) / (x.std(ddof=1) * y.std(ddof=1))


def add_mean_lines(axis: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    axis.axvline(x.mean(), color="#595959", linestyle="--", linewidth=1.4)
    axis.axhline(y.mean(), color="#595959", linestyle="--", linewidth=1.4)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    assert data["household_id"].is_unique
    assert not data.isna().any().any()
    data["income_thousand_eur"] = data["income_eur"] / 1_000
    data.to_csv(DATA / "household_finances.csv", index=False)

    income = data["income_eur"].to_numpy()
    income_thousands = data["income_thousand_eur"].to_numpy()
    leisure = data["leisure_spending_eur"].to_numpy()
    commute = data["commute_minutes"].to_numpy()

    rows = [
        {
            "x": "income_eur",
            "y": "leisure_spending_eur",
            "covariance": sample_covariance(income, leisure),
            "correlation": sample_correlation(income, leisure),
        },
        {
            "x": "income_thousand_eur",
            "y": "leisure_spending_eur",
            "covariance": sample_covariance(income_thousands, leisure),
            "correlation": sample_correlation(income_thousands, leisure),
        },
        {
            "x": "income_eur",
            "y": "commute_minutes",
            "covariance": sample_covariance(income, commute),
            "correlation": sample_correlation(income, commute),
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "covariance_summary.csv", index=False)

    matrix_columns = [
        "income_eur", "income_thousand_eur", "leisure_spending_eur", "commute_minutes"
    ]
    covariance_matrix = data[matrix_columns].cov()
    correlation_matrix = data[matrix_columns].corr()
    covariance_matrix.to_csv(RESULTS / "covariance_matrix.csv")
    correlation_matrix.to_csv(RESULTS / "correlation_matrix.csv")

    assert np.isclose(rows[0]["covariance"], data["income_eur"].cov(data["leisure_spending_eur"]))
    assert np.isclose(rows[0]["correlation"], correlation_matrix.loc["income_eur", "leisure_spending_eur"])
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "\nEinheitencheck: Cov(€)/Cov(Tsd. €) = "
        f"{rows[0]['covariance'] / rows[1]['covariance']:.1f}; "
        f"Differenz der Korrelationen = {rows[0]['correlation'] - rows[1]['correlation']:.2e}"
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    contribution_positive = (income - income.mean()) * (leisure - leisure.mean())
    colors_positive = np.where(contribution_positive >= 0, "#4472C4", "#ED7D31")
    axes[0, 0].scatter(income, leisure, c=colors_positive, alpha=0.68, s=24)
    add_mean_lines(axes[0, 0], income, leisure)
    axes[0, 0].set(
        title=f"Positive Kovarianz: {rows[0]['covariance']:,.0f} €²",
        xlabel="Monatliches Einkommen (€)",
        ylabel="Freizeitausgaben (€)",
    )
    contribution_legend = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#4472C4",
               markeredgecolor="#4472C4", label="Positiver Beitrag"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#ED7D31",
               markeredgecolor="#ED7D31", label="Negativer Beitrag"),
    ]
    axes[0, 0].legend(handles=contribution_legend, frameon=False, fontsize=8)

    contribution_negative = (income - income.mean()) * (commute - commute.mean())
    colors_negative = np.where(contribution_negative >= 0, "#4472C4", "#ED7D31")
    axes[0, 1].scatter(income, commute, c=colors_negative, alpha=0.68, s=24)
    add_mean_lines(axes[0, 1], income, commute)
    axes[0, 1].set(
        title=f"Negative Kovarianz: {rows[2]['covariance']:,.0f} €·Min.",
        xlabel="Monatliches Einkommen (€)",
        ylabel="Pendelzeit (Minuten)",
    )

    covariance_values = [abs(rows[0]["covariance"]), abs(rows[1]["covariance"])]
    bars = axes[1, 0].bar(["Einkommen in €", "Einkommen in Tsd. €"], covariance_values,
                          color=["#4472C4", "#70AD47"])
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="Kovarianz ändert sich mit der Einheit",
                   ylabel="Absolute Kovarianz (log-Skala)")
    for bar, value in zip(bars, covariance_values):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:,.1f}",
                        ha="center", va="bottom")

    correlation_values = [rows[0]["correlation"], rows[1]["correlation"]]
    bars = axes[1, 1].bar(["Einkommen in €", "Einkommen in Tsd. €"], correlation_values,
                          color=["#4472C4", "#70AD47"])
    axes[1, 1].set(title="Korrelation bleibt unverändert", ylabel="Korrelation",
                   ylim=(0, 1))
    for bar, value in zip(bars, correlation_values):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"r = {value:.3f}",
                        ha="center", va="bottom")

    figure.suptitle("Kovarianz: mittleres Produkt zentrierter Beobachtungen")
    figure.tight_layout()
    figure.savefig(RESULTS / "covariance_intuition.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
