"""Musterlösung für T12: Nichtlineare Beziehungen und Residuen."""

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


def make_data(size: int = 365) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    temperature = rng.uniform(-5, 35, size=size)
    energy = 82 + 1.15 * (temperature - 18) ** 2 + rng.normal(0, 20, size=size)
    return pd.DataFrame(
        {"day": np.arange(1, size + 1), "temperature_c": temperature, "energy_kwh": energy}
    )


def model_row(name: str, model: sm.regression.linear_model.RegressionResultsWrapper) -> dict[str, float | str]:
    return {
        "model": name,
        "r_squared": model.rsquared,
        "adjusted_r_squared": model.rsquared_adj,
        "rmse": np.sqrt(np.mean(model.resid ** 2)),
    }


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data["temperature_squared"] = data["temperature_c"] ** 2
    data.to_csv(DATA / "temperature_energy.csv", index=False)

    linear = sm.OLS(data["energy_kwh"], sm.add_constant(data[["temperature_c"]])).fit()
    quadratic = sm.OLS(
        data["energy_kwh"], sm.add_constant(data[["temperature_c", "temperature_squared"]])
    ).fit()
    data["linear_fitted"] = linear.fittedvalues
    data["linear_residual"] = linear.resid
    data["quadratic_fitted"] = quadratic.fittedvalues
    data["quadratic_residual"] = quadratic.resid

    comparison = pd.DataFrame([model_row("linear", linear), model_row("quadratic", quadratic)])
    comparison["estimated_minimum_c"] = [np.nan, -quadratic.params["temperature_c"] /
                                          (2 * quadratic.params["temperature_squared"])]
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

    data["temperature_bin"] = pd.cut(data["temperature_c"], bins=np.arange(-5, 41, 5),
                                      include_lowest=True)
    residual_bins = (
        data.groupby("temperature_bin", observed=True)
        .agg(mean_temperature=("temperature_c", "mean"),
             linear_mean_residual=("linear_residual", "mean"),
             quadratic_mean_residual=("quadratic_residual", "mean"),
             n=("day", "size"))
        .reset_index()
    )
    residual_bins["temperature_bin"] = residual_bins["temperature_bin"].astype(str)
    residual_bins.to_csv(RESULTS / "residuals_by_temperature_bin.csv", index=False)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    order = np.argsort(data["temperature_c"].to_numpy())
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes[0, 0].scatter(data["temperature_c"], data["energy_kwh"], color="#5B9BD5",
                       alpha=0.55, s=22)
    axes[0, 0].plot(data["temperature_c"].to_numpy()[order],
                    data["linear_fitted"].to_numpy()[order], color="#C00000", linewidth=2)
    axes[0, 0].set(title="Lineares Modell verfehlt die U-Form", ylabel="Energieverbrauch (kWh)")

    axes[0, 1].scatter(data["temperature_c"], data["linear_residual"], color="#4472C4",
                       alpha=0.55, s=22)
    axes[0, 1].plot(residual_bins["mean_temperature"], residual_bins["linear_mean_residual"],
                    color="#ED7D31", marker="o", linewidth=2, label="Intervallmittel")
    axes[0, 1].axhline(0, color="#C00000", linestyle="--")
    axes[0, 1].set(title="Systematische lineare Residuen", ylabel="Residuum (kWh)")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].scatter(data["temperature_c"], data["energy_kwh"], color="#5B9BD5",
                       alpha=0.45, s=22)
    axes[1, 0].plot(data["temperature_c"].to_numpy()[order],
                    data["quadratic_fitted"].to_numpy()[order], color="#70AD47", linewidth=2)
    axes[1, 0].set(title="Quadratisches Modell bildet die Kurve ab",
                   xlabel="Außentemperatur (°C)", ylabel="Energieverbrauch (kWh)")

    axes[1, 1].scatter(data["temperature_c"], data["quadratic_residual"], color="#4472C4",
                       alpha=0.55, s=22)
    axes[1, 1].plot(residual_bins["mean_temperature"], residual_bins["quadratic_mean_residual"],
                    color="#70AD47", marker="o", linewidth=2, label="Intervallmittel")
    axes[1, 1].axhline(0, color="#C00000", linestyle="--")
    axes[1, 1].set(title="Quadratische Residuen ohne U-Muster", xlabel="Außentemperatur (°C)",
                   ylabel="Residuum (kWh)")
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Funktionale Form wird in Residuen sichtbar")
    figure.tight_layout()
    figure.savefig(RESULTS / "nonlinearity_diagnostics.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
