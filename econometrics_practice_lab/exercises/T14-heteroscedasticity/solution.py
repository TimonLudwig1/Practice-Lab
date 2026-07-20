"""Musterlösung für T14: Heteroskedastizität sichtbar machen."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    income = rng.uniform(1_500, 9_000, size=size)
    conditional_sd = 100 + 0.12 * income
    consumption = 700 + 0.42 * income + rng.normal(0, conditional_sd, size=size)
    return pd.DataFrame(
        {"household_id": np.arange(1, size + 1), "income_eur": income,
         "consumption_eur": consumption, "true_conditional_sd": conditional_sd}
    )


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "household_consumption.csv", index=False)

    design = sm.add_constant(data[["income_eur"]])
    model = sm.OLS(data["consumption_eur"], design).fit()
    robust = model.get_robustcov_results(cov_type="HC1")
    data["fitted"] = model.fittedvalues
    data["residual"] = model.resid

    conventional_ci = model.conf_int().loc["income_eur"].to_numpy()
    robust_ci = robust.conf_int()[1]
    inference = pd.DataFrame(
        [
            {"method": "Conventional", "slope": model.params["income_eur"],
             "standard_error": model.bse["income_eur"], "ci_95_low": conventional_ci[0],
             "ci_95_high": conventional_ci[1]},
            {"method": "HC1 robust", "slope": robust.params[1],
             "standard_error": robust.bse[1], "ci_95_low": robust_ci[0],
             "ci_95_high": robust_ci[1]},
        ]
    )
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
    inference["breusch_pagan_lm"] = lm_stat
    inference["breusch_pagan_p_value"] = lm_pvalue
    inference.to_csv(RESULTS / "inference_comparison.csv", index=False)

    data["fitted_decile"] = pd.qcut(data["fitted"], 10, labels=False) + 1
    scale = (
        data.groupby("fitted_decile", observed=True)
        .agg(mean_fitted=("fitted", "mean"), residual_sd=("residual", "std"),
             true_mean_sd=("true_conditional_sd", "mean"), n=("household_id", "size"))
        .reset_index()
    )
    scale.to_csv(RESULTS / "residual_scale_by_decile.csv", index=False)
    print(inference.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    order = np.argsort(data["income_eur"].to_numpy())
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(data["income_eur"], data["consumption_eur"], color="#5B9BD5",
                       alpha=0.55, s=22)
    axes[0, 0].plot(data["income_eur"].to_numpy()[order],
                    data["fitted"].to_numpy()[order], color="#C00000", linewidth=2)
    axes[0, 0].set(title="Streuung wächst mit dem Einkommen", xlabel="Einkommen (€)",
                   ylabel="Konsum (€)")

    axes[0, 1].scatter(data["fitted"], data["residual"], color="#4472C4", alpha=0.55, s=22)
    axes[0, 1].axhline(0, color="#C00000", linestyle="--")
    axes[0, 1].set(title="Trichter im Residuenplot", xlabel="Vorhergesagter Konsum (€)",
                   ylabel="Residuum (€)")

    axes[1, 0].plot(scale["mean_fitted"], scale["residual_sd"], color="#4472C4",
                    marker="o", linewidth=2, label="Empirische Residuen-SD")
    axes[1, 0].plot(scale["mean_fitted"], scale["true_mean_sd"], color="#ED7D31",
                    marker="s", linestyle="--", label="Wahre bedingte SD")
    axes[1, 0].set(title="Fehlerskala nach Fit-Dezil", xlabel="Mittlerer Fit (€)",
                   ylabel="Standardabweichung (€)")
    axes[1, 0].legend(frameon=False)

    y_positions = np.arange(len(inference))
    estimates = inference["slope"].to_numpy()
    left = estimates - inference["ci_95_low"].to_numpy()
    right = inference["ci_95_high"].to_numpy() - estimates
    axes[1, 1].errorbar(estimates, y_positions, xerr=np.vstack([left, right]), fmt="o",
                        color="#4472C4", capsize=6)
    axes[1, 1].axvline(0.42, color="#C00000", linestyle="--", label="Wahre Steigung")
    axes[1, 1].set(title="Gleicher Schätzer, andere Unsicherheit", xlabel="Steigung mit 95%-KI",
                   yticks=y_positions, yticklabels=inference["method"])
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Heteroskedastizität verändert die bedingte Streuung")
    figure.tight_layout()
    figure.savefig(RESULTS / "heteroscedasticity_diagnostics.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
