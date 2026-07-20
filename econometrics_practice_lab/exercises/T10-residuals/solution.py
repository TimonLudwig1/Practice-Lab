"""Musterlösung für T10: Residuen verstehen und diagnostizieren."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    area = rng.uniform(25, 140, size=size)
    rent = 320 + 13.5 * area + rng.normal(0, 150, size=size)
    rent[[17, 88, 153]] += np.array([720, -620, 850])
    return pd.DataFrame(
        {"listing_id": np.arange(1, size + 1), "floor_area_sqm": area, "rent_eur": rent}
    )


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "apartment_rents.csv", index=False)

    model = sm.OLS(data["rent_eur"], sm.add_constant(data["floor_area_sqm"])).fit()
    data["fitted_rent"] = model.fittedvalues
    data["residual"] = model.resid
    data["absolute_residual"] = data["residual"].abs()
    data["absolute_residual_rank"] = data["absolute_residual"].rank(
        method="first", ascending=False
    ).astype(int)
    data.to_csv(RESULTS / "residual_table.csv", index=False)

    properties = {
        "residual_sum": data["residual"].sum(),
        "residual_mean": data["residual"].mean(),
        "cov_residual_area": data["residual"].cov(data["floor_area_sqm"]),
        "cov_residual_fitted": data["residual"].cov(data["fitted_rent"]),
        "rmse": np.sqrt(np.mean(data["residual"] ** 2)),
        "largest_absolute_residual": data["absolute_residual"].max(),
    }
    pd.DataFrame([properties]).to_csv(RESULTS / "residual_summary.csv", index=False)
    reconstruction_error = (
        data["rent_eur"] - data["fitted_rent"] - data["residual"]
    ).abs().max()
    assert reconstruction_error < 1e-9
    print(pd.Series(properties).to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nFünf größte absolute Residuen:")
    print(
        data.nsmallest(5, "absolute_residual_rank")[
            ["listing_id", "floor_area_sqm", "rent_eur", "fitted_rent", "residual"]
        ].to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    order = np.argsort(data["floor_area_sqm"].to_numpy())
    axes[0, 0].scatter(data["floor_area_sqm"], data["rent_eur"], color="#5B9BD5",
                       alpha=0.65, s=24, label="Beobachtung")
    axes[0, 0].plot(data["floor_area_sqm"].to_numpy()[order],
                    data["fitted_rent"].to_numpy()[order], color="#C00000", linewidth=2,
                    label="OLS-Fit")
    highlighted = data.nsmallest(5, "absolute_residual_rank")
    for _, row in highlighted.iterrows():
        axes[0, 0].plot([row["floor_area_sqm"], row["floor_area_sqm"]],
                        [row["fitted_rent"], row["rent_eur"]], color="#ED7D31", linewidth=2)
    axes[0, 0].set(title="Residuum als vertikaler Abstand", xlabel="Wohnfläche (m²)",
                   ylabel="Monatsmiete (€)")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].scatter(data["fitted_rent"], data["residual"], color="#4472C4",
                       alpha=0.68, s=24)
    axes[0, 1].scatter(highlighted["fitted_rent"], highlighted["residual"],
                       color="#ED7D31", marker="D", s=48, label="Top-5 |Residuum|")
    axes[0, 1].axhline(0, color="#C00000", linestyle="--")
    axes[0, 1].set(title="Residuen gegen Fits", xlabel="Vorhergesagte Miete (€)",
                   ylabel="Residuum (€)")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].hist(data["residual"], bins=24, color="#70AD47", edgecolor="white")
    axes[1, 0].axvline(0, color="#C00000", linestyle="--")
    axes[1, 0].set(title="Verteilung der Residuen", xlabel="Residuum (€)", ylabel="Häufigkeit")

    axes[1, 1].scatter(data["fitted_rent"], data["rent_eur"], color="#5B9BD5",
                       alpha=0.68, s=24)
    limits = [min(data["fitted_rent"].min(), data["rent_eur"].min()),
              max(data["fitted_rent"].max(), data["rent_eur"].max())]
    axes[1, 1].plot(limits, limits, color="#C00000", linestyle="--",
                    label="Beobachtung = Fit")
    axes[1, 1].set(title="Beobachtet gegen vorhergesagt", xlabel="Vorhergesagte Miete (€)",
                   ylabel="Beobachtete Miete (€)")
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Residuen zerlegen Beobachtung in Fit und Fehler")
    figure.tight_layout()
    figure.savefig(RESULTS / "residual_diagnostics.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
