"""Musterlösung für T43: DiD mit Zeit-Fixed-Effects."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260723
TRUE_EFFECT = 3.5
TIME_SHOCKS = np.array([0.0, 4.0, -2.0, 5.0, 1.0, -3.0, 6.0, 2.0, -1.0, 4.0, 0.0, -4.0])


def make_data(regions: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    treated = np.repeat([0, 1], regions // 2)
    region_effect = rng.normal(0, 2.0, regions)
    rows = []
    for region in range(regions):
        for month, time_shock in enumerate(TIME_SHOCKS, start=1):
            post = int(month >= 7)
            did = treated[region] * post
            visits = (50 + 5 * treated[region] + region_effect[region] + time_shock
                      + TRUE_EFFECT * did + rng.normal(0, 1.5))
            rows.append((region + 1, month, treated[region], post, did, visits, time_shock))
    return pd.DataFrame(rows, columns=["region_id", "month", "treated", "post", "did",
                                       "visits_per_1000", "oracle_time_shock"])


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "regional_voucher_panel.csv", index=False)

    cluster = {"groups": data["region_id"]}
    naive = smf.ols("visits_per_1000 ~ did", data=data).fit(cov_type="cluster", cov_kwds=cluster)
    post_did = smf.ols(
        "visits_per_1000 ~ treated + post + did", data=data
    ).fit(cov_type="cluster", cov_kwds=cluster)
    time_fe = smf.ols(
        "visits_per_1000 ~ treated + did + C(month)", data=data
    ).fit(cov_type="cluster", cov_kwds=cluster)

    model_rows = []
    for name, model in [("Naive: only active treatment", naive),
                        ("DiD with post dummy", post_did),
                        ("DiD with month fixed effects", time_fe)]:
        low, high = model.conf_int().loc["did"]
        model_rows.append({"model": name, "did_estimate": model.params["did"],
                           "cluster_standard_error": model.bse["did"],
                           "ci_95_low": low, "ci_95_high": high,
                           "residual_rmse": np.sqrt(np.mean(model.resid ** 2))})
    estimates = pd.DataFrame(model_rows)
    estimates.to_csv(RESULTS / "time_fe_model_comparison.csv", index=False)
    assert abs(time_fe.params["did"] - TRUE_EFFECT) < 0.4
    assert abs(naive.params["did"] - TRUE_EFFECT) > 2.0
    assert time_fe.mse_resid < post_did.mse_resid

    time_rows = []
    for month in range(1, 13):
        parameter = f"C(month)[T.{month}]"
        estimate = 0.0 if month == 1 else time_fe.params[parameter]
        time_rows.append({"month": month, "estimated_relative_time_fe": estimate,
                          "oracle_time_shock": TIME_SHOCKS[month - 1]})
    time_effects = pd.DataFrame(time_rows)
    time_effects["estimated_centered_time_fe"] = (
        time_effects["estimated_relative_time_fe"] - time_effects["estimated_relative_time_fe"].mean()
    )
    time_effects["oracle_centered_time_shock"] = (
        time_effects["oracle_time_shock"] - time_effects["oracle_time_shock"].mean()
    )
    time_correlation = time_effects["estimated_centered_time_fe"].corr(
        time_effects["oracle_centered_time_shock"]
    )
    assert time_correlation > 0.98
    time_effects.to_csv(RESULTS / "estimated_time_fixed_effects.csv", index=False)

    residuals = data[["month"]].copy()
    residuals["post_dummy_residual"] = post_did.resid
    residuals["time_fe_residual"] = time_fe.resid
    monthly_residuals = residuals.groupby("month", as_index=False).mean()
    monthly_residuals.to_csv(RESULTS / "monthly_residual_means.csv", index=False)

    print(estimates.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nKorrelation geschätzter und wahrer Zeiteffekte: {time_correlation:.4f}")

    means = data.groupby(["month", "treated"], as_index=False)["visits_per_1000"].mean()
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for group, label, color in [(0, "Kontrollregionen", "#595959"),
                                (1, "Gutscheinregionen", "#4472C4")]:
        subset = means.loc[means["treated"].eq(group)]
        axes[0, 0].plot(subset["month"], subset["visits_per_1000"], marker="o",
                        color=color, label=label)
    axes[0, 0].axvline(6.5, color="#C00000", linestyle="--")
    axes[0, 0].set(title="Gemeinsame Monatsschocks bewegen beide Gruppen",
                   xlabel="Monat", ylabel="Besuche je 1.000 Personen")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].scatter(time_effects["oracle_centered_time_shock"],
                       time_effects["estimated_centered_time_fe"], color="#4472C4", s=48)
    limits = [time_effects[["oracle_centered_time_shock", "estimated_centered_time_fe"]].min().min(),
              time_effects[["oracle_centered_time_shock", "estimated_centered_time_fe"]].max().max()]
    axes[0, 1].plot(limits, limits, color="#C00000", linestyle="--")
    for row in time_effects.itertuples(index=False):
        axes[0, 1].annotate(str(row.month),
                            (row.oracle_centered_time_shock, row.estimated_centered_time_fe),
                            xytext=(4, 3), textcoords="offset points", fontsize=8)
    axes[0, 1].set(title=f"Zeit-FE rekonstruieren die Schocks (r={time_correlation:.2f})",
                   xlabel="Wahrer zentrierter Monatsschock", ylabel="Geschätzter zentrierter Zeit-FE")

    axes[1, 0].plot(monthly_residuals["month"], monthly_residuals["post_dummy_residual"],
                    marker="o", color="#C00000", label="Nur Post-Dummy")
    axes[1, 0].plot(monthly_residuals["month"], monthly_residuals["time_fe_residual"],
                    marker="s", color="#4472C4", label="Monats-FE")
    axes[1, 0].axhline(0, color="#595959", linewidth=1)
    axes[1, 0].set(title="Monats-FE entfernen gemeinsame Restmuster",
                   xlabel="Monat", ylabel="Mittleres Residuum")
    axes[1, 0].legend(frameon=False)

    y_pos = np.arange(len(estimates))
    axes[1, 1].errorbar(estimates["did_estimate"], y_pos,
                        xerr=1.96 * estimates["cluster_standard_error"], fmt="o",
                        color="#4472C4", capsize=5)
    axes[1, 1].axvline(TRUE_EFFECT, color="#C00000", linestyle="--", label="Wahrer Effekt")
    axes[1, 1].set(title="Flexible Zeitkontrolle isoliert den Programmeffekt",
                   xlabel="DiD-Effekt", yticks=y_pos,
                   yticklabels=["Naiv", "Post-Dummy", "Monats-FE"])
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Zeit-Fixed-Effects kontrollieren beliebige gemeinsame Zeitschocks")
    figure.tight_layout()
    figure.savefig(RESULTS / "time_fixed_effects_did.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
