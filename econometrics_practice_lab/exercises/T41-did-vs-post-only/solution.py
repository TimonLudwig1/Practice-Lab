"""Musterlösung für T41: DiD versus Post-Treatment-Daten allein."""

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
RNG_SEED = 20260721
TRUE_EFFECT = 5.0


def make_data(schools: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    treated = np.repeat([0, 1], schools // 2)
    school_effect = rng.normal(0, 2.2, schools)
    rows = []
    for school in range(schools):
        for post in [0, 1]:
            score = (67 - 10 * treated[school] + school_effect[school] + 2.5 * post
                     + TRUE_EFFECT * treated[school] * post + rng.normal(0, 1.2))
            rows.append((school + 1, post, treated[school], score))
    return pd.DataFrame(rows, columns=["school_id", "post", "treated", "math_score"])


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "school_panel.csv", index=False)

    means = data.groupby(["treated", "post"], as_index=False)["math_score"].mean()
    table = means.pivot(index="treated", columns="post", values="math_score")
    baseline_gap = table.loc[1, 0] - table.loc[0, 0]
    post_gap = table.loc[1, 1] - table.loc[0, 1]
    control_change = table.loc[0, 1] - table.loc[0, 0]
    treated_change = table.loc[1, 1] - table.loc[1, 0]
    manual_did = treated_change - control_change

    post_data = data.loc[data["post"].eq(1)]
    post_model = smf.ols("math_score ~ treated", data=post_data).fit()
    did_model = smf.ols("math_score ~ treated + post + treated:post", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["school_id"]}
    )
    wide = data.pivot(index="school_id", columns="post", values="math_score")
    treatment = data.groupby("school_id")["treated"].first()
    changes = pd.DataFrame({"change": wide[1] - wide[0], "treated": treatment})
    change_model = smf.ols("change ~ treated", data=changes).fit()

    comparison = pd.DataFrame([
        {"method": "Post-only difference", "estimate": post_model.params["treated"]},
        {"method": "Change-score difference", "estimate": change_model.params["treated"]},
        {"method": "DiD interaction", "estimate": did_model.params["treated:post"]},
        {"method": "True treatment effect", "estimate": TRUE_EFFECT},
    ])
    comparison.to_csv(RESULTS / "estimator_comparison.csv", index=False)
    decomposition = pd.DataFrame([{
        "baseline_group_gap": baseline_gap,
        "control_group_change": control_change,
        "did_treatment_effect": manual_did,
        "observed_post_gap": post_gap,
        "baseline_gap_plus_did": baseline_gap + manual_did,
    }])
    decomposition.to_csv(RESULTS / "post_gap_decomposition.csv", index=False)
    means.to_csv(RESULTS / "group_time_means.csv", index=False)

    assert post_model.params["treated"] < 0
    assert manual_did > 0
    assert abs(manual_did - did_model.params["treated:post"]) < 1e-10
    assert abs(manual_did - change_model.params["treated"]) < 1e-10
    assert abs(post_gap - (baseline_gap + manual_did)) < 1e-10

    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nZerlegung des Post-Unterschieds:")
    print(decomposition.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {0: "#595959", 1: "#4472C4"}
    labels = {0: "Kontrollschulen", 1: "Förderschulen"}
    for group in [0, 1]:
        axes[0].plot([0, 1], [table.loc[group, 0], table.loc[group, 1]], marker="o",
                     linewidth=2.2, color=colors[group], label=labels[group])
    counterfactual = table.loc[1, 0] + control_change
    axes[0].plot([0, 1], [table.loc[1, 0], counterfactual], marker="o",
                 color="#C00000", linestyle="--", label="Gegenfaktum")
    axes[0].annotate(f"DiD = +{manual_did:.2f}", xy=(1, table.loc[1, 1]),
                     xytext=(0.47, counterfactual + 2.0),
                     arrowprops={"arrowstyle": "->", "color": "#C00000"})
    axes[0].set(title="Die Veränderung zeigt den positiven Effekt", xlabel="Zeitpunkt",
                ylabel="Mathematikscore", xticks=[0, 1], xticklabels=["Vorher", "Nachher"])
    axes[0].legend(frameon=False)

    plot_values = comparison.iloc[[0, 1, 2]].copy()
    bar_colors = ["#C00000", "#4472C4", "#4472C4"]
    bars = axes[1].barh(plot_values["method"], plot_values["estimate"], color=bar_colors)
    axes[1].axvline(0, color="#595959", linewidth=1)
    for bar, value in zip(bars, plot_values["estimate"]):
        x_position = value - 0.25 if value > 0 else value + 0.25
        axes[1].text(x_position, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}",
                     ha="right" if value > 0 else "left", va="center", color="white")
    axes[1].set(title="Post-only liefert sogar das falsche Vorzeichen",
                xlabel="Geschätzter Programmeffekt (Scorepunkte)")
    figure.suptitle("Ohne Baseline verwechselt man Niveauunterschied und Wirkung")
    figure.tight_layout()
    figure.savefig(RESULTS / "post_only_vs_did.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
