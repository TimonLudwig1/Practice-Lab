"""Musterlösung für T40: klassisches Differences-in-Differences."""

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
RNG_SEED = 20260720
TRUE_EFFECT = -6.0


def make_data(cities: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    treated = np.repeat([0, 1], cities // 2)
    city_effect = rng.normal(0, 2.4, cities)
    year_shocks = {2017: 0.0, 2018: -0.7, 2019: -1.4, 2020: -2.3,
                   2021: -2.9, 2022: -3.8, 2023: -4.5}
    rows = []
    for city in range(cities):
        for year in range(2017, 2024):
            post = int(year >= 2021)
            no2 = (39 + 7.5 * treated[city] + city_effect[city] + year_shocks[year]
                   + TRUE_EFFECT * treated[city] * post + rng.normal(0, 1.8))
            rows.append((city + 1, year, treated[city], post, no2))
    return pd.DataFrame(rows, columns=["city_id", "year", "treated", "post", "no2_ug_m3"])


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "municipality_panel.csv", index=False)
    assert not data.duplicated(["city_id", "year"]).any()

    grouped = (
        data.assign(period=np.where(data["post"].eq(1), "Nachher", "Vorher"))
        .groupby(["treated", "period"], as_index=False)["no2_ug_m3"].mean()
    )
    pivot = grouped.pivot(index="treated", columns="period", values="no2_ug_m3")
    control_change = pivot.loc[0, "Nachher"] - pivot.loc[0, "Vorher"]
    treated_change = pivot.loc[1, "Nachher"] - pivot.loc[1, "Vorher"]
    manual_did = treated_change - control_change
    grouped.to_csv(RESULTS / "group_period_means.csv", index=False)

    model = smf.ols(
        "no2_ug_m3 ~ treated + post + treated:post", data=data
    ).fit(cov_type="cluster", cov_kwds={"groups": data["city_id"]})
    did_name = "treated:post"
    ci_low, ci_high = model.conf_int().loc[did_name]
    estimates = pd.DataFrame([
        {"method": "Manual DiD", "estimate": manual_did, "standard_error": np.nan,
         "ci_95_low": np.nan, "ci_95_high": np.nan},
        {"method": "Regression DiD", "estimate": model.params[did_name],
         "standard_error": model.bse[did_name], "ci_95_low": ci_low, "ci_95_high": ci_high},
    ])
    estimates.to_csv(RESULTS / "did_estimates.csv", index=False)
    assert abs(manual_did - model.params[did_name]) < 1e-10
    assert abs(manual_did - TRUE_EFFECT) < 1.0

    pre = data.loc[data["post"].eq(0)].copy()
    pre["year_index"] = pre["year"] - pre["year"].min()
    pretrend = smf.ols(
        "no2_ug_m3 ~ treated + year_index + treated:year_index", data=pre
    ).fit(cov_type="cluster", cov_kwds={"groups": pre["city_id"]})
    pretrend_table = pd.DataFrame([{
        "differential_pretrend": pretrend.params["treated:year_index"],
        "standard_error": pretrend.bse["treated:year_index"],
        "p_value": pretrend.pvalues["treated:year_index"],
    }])
    pretrend_table.to_csv(RESULTS / "pretrend_diagnostic.csv", index=False)
    assert abs(pretrend.params["treated:year_index"]) < 0.8

    print("Vier Mittelwerte:")
    print(grouped.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nKontrolländerung: {control_change:.3f}")
    print(f"Treatmentänderung: {treated_change:.3f}")
    print(f"DiD (Hand = Regression): {manual_did:.3f}")
    print(f"95%-KI: [{ci_low:.3f}, {ci_high:.3f}]")
    print("\nPre-Trend-Diagnostik:")
    print(pretrend_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    annual = data.groupby(["year", "treated"], as_index=False)["no2_ug_m3"].mean()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = {0: "Kontrollgruppe", 1: "Umweltzone"}
    colors = {0: "#595959", 1: "#4472C4"}
    for group in [0, 1]:
        subset = annual.loc[annual["treated"].eq(group)]
        axes[0].plot(subset["year"], subset["no2_ug_m3"], marker="o",
                     color=colors[group], label=labels[group])
        axes[0].text(subset["year"].iloc[-1] + 0.08, subset["no2_ug_m3"].iloc[-1],
                     labels[group], color=colors[group], va="center")
    axes[0].axvline(2020.5, color="#C00000", linestyle="--", linewidth=1.5)
    axes[0].text(2020.58, axes[0].get_ylim()[1] - 0.7, "Start", color="#C00000")
    axes[0].set(title="Gruppen dürfen verschiedene Niveaus haben",
                xlabel="Jahr", ylabel="NO₂ (µg/m³)")
    axes[0].legend().remove()

    before_control = pivot.loc[0, "Vorher"]
    before_treated = pivot.loc[1, "Vorher"]
    axes[1].plot([0, 1], [before_control, pivot.loc[0, "Nachher"]], marker="o",
                 color=colors[0], label=labels[0])
    axes[1].plot([0, 1], [before_treated, pivot.loc[1, "Nachher"]], marker="o",
                 color=colors[1], label=labels[1])
    counterfactual = before_treated + control_change
    axes[1].plot([0, 1], [before_treated, counterfactual], marker="o",
                 color="#C00000", linestyle="--", label="Gegenfaktum")
    axes[1].annotate(f"DiD = {manual_did:.2f}", xy=(1, pivot.loc[1, "Nachher"]),
                     xytext=(0.55, counterfactual + 1.2),
                     arrowprops={"arrowstyle": "->", "color": "#C00000"})
    axes[1].set(title="DiD rekonstruiert das Gegenfaktum", xlabel="Periode",
                ylabel="NO₂ (µg/m³)", xticks=[0, 1], xticklabels=["Vorher", "Nachher"])
    axes[1].legend(frameon=False)
    figure.suptitle("Umweltzone: Veränderung relativ zur Kontrollgruppe")
    figure.tight_layout()
    figure.savefig(RESULTS / "did_environment_zone.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
