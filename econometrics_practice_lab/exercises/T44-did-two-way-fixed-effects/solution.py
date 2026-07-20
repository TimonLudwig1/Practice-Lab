"""Musterlösung für T44: DiD mit individuellen und zeitlichen Fixed Effects."""

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
RNG_SEED = 20260724
TRUE_EFFECT = 4.0
WEEK_SHOCKS = np.array([0.0, 2.5, -1.0, 3.0, -2.0, 5.0, 1.5, 4.0, -1.5, 3.5])


def make_data(stores: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    quality = rng.normal(size=stores)
    treated = (quality >= np.median(quality)).astype(int)
    store_intercept = 75 + 8 * quality + rng.normal(0, 1.2, stores)
    rows = []
    for store in range(stores):
        for week, shock in enumerate(WEEK_SHOCKS, start=1):
            post = int(week >= 6)
            did = treated[store] * post
            sales = store_intercept[store] + shock + TRUE_EFFECT * did + rng.normal(0, 1.6)
            rows.append((store + 1, week, treated[store], post, did, sales, quality[store], shock))
    return pd.DataFrame(rows, columns=["store_id", "week", "treated", "post", "did",
                                       "sales_thousand_eur", "oracle_location_quality",
                                       "oracle_week_shock"])


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "store_twfe_panel.csv", index=False)
    cluster = {"groups": data["store_id"]}

    models = [
        ("Naive", smf.ols("sales_thousand_eur ~ did", data=data).fit(
            cov_type="cluster", cov_kwds=cluster)),
        ("Store FE only", smf.ols("sales_thousand_eur ~ did + C(store_id)", data=data).fit(
            cov_type="cluster", cov_kwds=cluster)),
        ("Week FE only", smf.ols("sales_thousand_eur ~ did + C(week)", data=data).fit(
            cov_type="cluster", cov_kwds=cluster)),
        ("Store + week FE", smf.ols(
            "sales_thousand_eur ~ did + C(store_id) + C(week)", data=data
        ).fit(cov_type="cluster", cov_kwds=cluster)),
    ]
    rows = []
    for name, model in models:
        low, high = model.conf_int().loc["did"]
        rows.append({"model": name, "did_estimate": model.params["did"],
                     "cluster_standard_error": model.bse["did"],
                     "ci_95_low": low, "ci_95_high": high})
    estimates = pd.DataFrame(rows)
    estimates.to_csv(RESULTS / "twfe_model_comparison.csv", index=False)
    twfe_model = models[-1][1]
    assert abs(twfe_model.params["did"] - TRUE_EFFECT) < 0.35
    assert abs(models[1][1].params["did"] - TRUE_EFFECT) > 0.8
    assert abs(models[2][1].params["did"] - TRUE_EFFECT) > 2.0

    outcome = data["sales_thousand_eur"]
    treatment = data["did"]
    data["outcome_tw"] = (outcome - data.groupby("store_id")["sales_thousand_eur"].transform("mean")
                          - data.groupby("week")["sales_thousand_eur"].transform("mean")
                          + outcome.mean())
    data["did_tw"] = (treatment - data.groupby("store_id")["did"].transform("mean")
                      - data.groupby("week")["did"].transform("mean") + treatment.mean())
    manual_twfe = (data["did_tw"] * data["outcome_tw"]).sum() / (data["did_tw"] ** 2).sum()

    group_period = data.groupby(["treated", "post"])["sales_thousand_eur"].mean().unstack()
    classic_did = ((group_period.loc[1, 1] - group_period.loc[1, 0])
                   - (group_period.loc[0, 1] - group_period.loc[0, 0]))
    equivalence = pd.DataFrame([{
        "twfe_regression": twfe_model.params["did"],
        "manual_double_demeaning": manual_twfe,
        "classic_group_time_did": classic_did,
        "max_pairwise_difference": max(abs(twfe_model.params["did"] - manual_twfe),
                                        abs(twfe_model.params["did"] - classic_did)),
    }])
    equivalence.to_csv(RESULTS / "twfe_equivalence.csv", index=False)
    assert equivalence.loc[0, "max_pairwise_difference"] < 1e-10

    diagnostics = pd.DataFrame([{
        "correlation_treatment_location_quality": data[["treated", "oracle_location_quality"]]
        .drop_duplicates()["treated"].corr(data[["treated", "oracle_location_quality"]]
                                            .drop_duplicates()["oracle_location_quality"]),
        "mean_pre_week_shock": WEEK_SHOCKS[:5].mean(),
        "mean_post_week_shock": WEEK_SHOCKS[5:].mean(),
    }])
    diagnostics.to_csv(RESULTS / "twfe_diagnostics.csv", index=False)
    print(estimates.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nÄquivalenz im kanonischen Design:")
    print(equivalence.to_string(index=False, float_format=lambda value: f"{value:.10f}"))

    means = data.groupby(["week", "treated"], as_index=False)["sales_thousand_eur"].mean()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for group, label, color in [(0, "Kontrollfilialen", "#595959"),
                                (1, "Beratungsfilialen", "#4472C4")]:
        subset = means.loc[means["treated"].eq(group)]
        axes[0].plot(subset["week"], subset["sales_thousand_eur"], marker="o",
                     color=color, label=label)
    axes[0].axvline(5.5, color="#C00000", linestyle="--")
    axes[0].set(title="Filialniveaus und Wochenschocks überlagern den Effekt",
                xlabel="Woche", ylabel="Umsatz (Tsd. €)")
    axes[0].legend(frameon=False)

    y_pos = np.arange(len(estimates))
    axes[1].errorbar(estimates["did_estimate"], y_pos,
                     xerr=1.96 * estimates["cluster_standard_error"], fmt="o",
                     color="#4472C4", capsize=5)
    axes[1].axvline(TRUE_EFFECT, color="#C00000", linestyle="--", label="Wahrer Effekt")
    axes[1].set(title="Nur Two-Way FE kontrolliert beide Störquellen",
                xlabel="Geschätzter Treatment-Effekt (Tsd. €)", yticks=y_pos,
                yticklabels=["Naiv", "Nur Filial-FE", "Nur Wochen-FE", "Two-Way FE"])
    axes[1].legend(frameon=False)
    figure.suptitle("Two-Way Fixed Effects: innerhalb der Filiale und relativ zur selben Woche")
    figure.tight_layout()
    figure.savefig(RESULTS / "two_way_fixed_effects.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
