"""Musterlösung für T15: Autokorrelation in Regressionsresiduen."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 365, phi: float = 0.78) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    innovation = rng.normal(0, 6, size=size)
    ar_error = np.empty(size)
    ar_error[0] = innovation[0] / np.sqrt(1 - phi ** 2)
    for day in range(1, size):
        ar_error[day] = phi * ar_error[day - 1] + innovation[day]
    day_index = np.arange(1, size + 1)
    orders = 120 + 0.12 * day_index + ar_error
    return pd.DataFrame(
        {"day": day_index, "orders": orders, "structural_error": ar_error}
    )


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "daily_orders.csv", index=False)

    design = sm.add_constant(data[["day"]])
    model = sm.OLS(data["orders"], design).fit()
    hac = model.get_robustcov_results(cov_type="HAC", maxlags=7)
    data["fitted"] = model.fittedvalues
    data["residual"] = model.resid

    residual_acf = acf(data["residual"], nlags=14, fft=False)
    lag1 = data["residual"].autocorr(lag=1)
    dw = durbin_watson(data["residual"])
    ljung_box = acorr_ljungbox(data["residual"], lags=[7], return_df=True).iloc[0]
    autocorrelation_summary = pd.DataFrame(
        [{"lag1_residual_correlation": lag1, "durbin_watson": dw,
          "ljung_box_lag": 7, "ljung_box_statistic": ljung_box["lb_stat"],
          "ljung_box_p_value": ljung_box["lb_pvalue"]}]
    )
    autocorrelation_summary.to_csv(RESULTS / "autocorrelation_summary.csv", index=False)

    conventional_ci = model.conf_int().loc["day"].to_numpy()
    hac_ci = hac.conf_int()[1]
    inference = pd.DataFrame(
        [
            {"method": "Conventional", "trend_estimate": model.params["day"],
             "standard_error": model.bse["day"], "ci_95_low": conventional_ci[0],
             "ci_95_high": conventional_ci[1]},
            {"method": "HAC maxlags=7", "trend_estimate": hac.params[1],
             "standard_error": hac.bse[1], "ci_95_low": hac_ci[0], "ci_95_high": hac_ci[1]},
        ]
    )
    inference.to_csv(RESULTS / "inference_comparison.csv", index=False)
    print(autocorrelation_summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nTrend-Inferenz:")
    print(inference.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(data["day"], data["orders"], color="#5B9BD5", linewidth=1.2,
                    label="Bestellungen")
    axes[0, 0].plot(data["day"], data["fitted"], color="#C00000", linewidth=2,
                    label="Linearer Trend")
    axes[0, 0].set(title="Tägliche Bestellungen", xlabel="Tag", ylabel="Bestellungen")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(data["day"], data["residual"], color="#4472C4", linewidth=1.2)
    axes[0, 1].axhline(0, color="#C00000", linestyle="--")
    axes[0, 1].set(title="Residuen bilden anhaltende Runs", xlabel="Tag", ylabel="Residuum")

    axes[1, 0].scatter(data["residual"].iloc[:-1], data["residual"].iloc[1:],
                       color="#5B9BD5", alpha=0.62, s=24)
    axes[1, 0].set(title=f"Lag-Plot: r₁ = {lag1:.3f}", xlabel="Residuum t−1",
                   ylabel="Residuum t")

    lags = np.arange(1, len(residual_acf))
    axes[1, 1].bar(lags, residual_acf[1:], color="#4472C4")
    confidence = 1.96 / np.sqrt(len(data))
    axes[1, 1].axhline(confidence, color="#C00000", linestyle="--", label="±1,96/√n")
    axes[1, 1].axhline(-confidence, color="#C00000", linestyle="--")
    axes[1, 1].axhline(0, color="#595959", linewidth=1)
    axes[1, 1].set(title="Autokorrelationsfunktion der Residuen", xlabel="Lag",
                   ylabel="Autokorrelation", xticks=lags)
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Autokorrelation: benachbarte Fehler sind nicht unabhängig")
    figure.tight_layout()
    figure.savefig(RESULTS / "autocorrelation_diagnostics.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
