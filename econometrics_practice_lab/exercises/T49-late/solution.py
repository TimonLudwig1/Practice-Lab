"""Musterlösung für T49: Local Average Treatment Effect."""

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
RNG_SEED = 20260729


def make_data(observations: int = 20000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    compliance_type = rng.choice(["always", "complier", "never"], observations,
                                 p=[0.30, 0.30, 0.40])
    assignment = rng.binomial(1, 0.5, observations)
    d_if_z0 = (compliance_type == "always").astype(int)
    d_if_z1 = (compliance_type != "never").astype(int)
    treatment = np.where(assignment == 1, d_if_z1, d_if_z0)
    effects = pd.Series(compliance_type).map({"always": 12.0, "complier": 2.0, "never": -3.0}).to_numpy()
    baseline = 50 + pd.Series(compliance_type).map({"always": 7.0, "complier": 0.0, "never": -4.0}).to_numpy()
    outcome = baseline + effects * treatment + rng.normal(0, 2, observations)
    return pd.DataFrame({"assignment_z": assignment, "treatment_d": treatment,
                         "outcome": outcome, "compliance_type": compliance_type,
                         "d_if_z0": d_if_z0, "d_if_z1": d_if_z1,
                         "oracle_treatment_effect": effects})


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "compliance_experiment.csv", index=False)

    by_assignment = data.groupby("assignment_z").agg(
        treatment_rate=("treatment_d", "mean"), mean_outcome=("outcome", "mean")
    )
    always_share_inferred = by_assignment.loc[0, "treatment_rate"]
    complier_share_inferred = (by_assignment.loc[1, "treatment_rate"]
                               - by_assignment.loc[0, "treatment_rate"])
    never_share_inferred = 1 - by_assignment.loc[1, "treatment_rate"]
    first_stage = complier_share_inferred
    reduced_form = by_assignment.loc[1, "mean_outcome"] - by_assignment.loc[0, "mean_outcome"]
    wald_late = reduced_form / first_stage

    oracle_shares = data["compliance_type"].value_counts(normalize=True)
    oracle_late = data.loc[data["compliance_type"].eq("complier"), "oracle_treatment_effect"].mean()
    population_ate = data["oracle_treatment_effect"].mean()
    naive = sm.OLS(data["outcome"], sm.add_constant(data[["treatment_d"]])).fit().params["treatment_d"]
    defiers = int((data["d_if_z1"] < data["d_if_z0"]).sum())
    assert defiers == 0
    assert abs(wald_late - oracle_late) < 0.35
    assert abs(naive - oracle_late) > 5

    shares = pd.DataFrame([
        {"type": "always", "inferred_share": always_share_inferred,
         "oracle_share": oracle_shares["always"]},
        {"type": "complier", "inferred_share": complier_share_inferred,
         "oracle_share": oracle_shares["complier"]},
        {"type": "never", "inferred_share": never_share_inferred,
         "oracle_share": oracle_shares["never"]},
    ])
    shares.to_csv(RESULTS / "compliance_shares.csv", index=False)
    estimands = pd.DataFrame([
        {"estimand": "Naive treated-vs-untreated", "estimate": naive},
        {"estimand": "Wald IV", "estimate": wald_late},
        {"estimand": "Oracle complier effect (LATE)", "estimate": oracle_late},
        {"estimand": "Oracle population ATE", "estimate": population_ate},
    ])
    estimands.to_csv(RESULTS / "late_estimands.csv", index=False)
    diagnostics = pd.DataFrame([{
        "first_stage": first_stage, "reduced_form": reduced_form,
        "wald_late": wald_late, "oracle_late": oracle_late,
        "population_ate": population_ate, "number_of_defiers": defiers,
    }])
    diagnostics.to_csv(RESULTS / "late_diagnostics.csv", index=False)

    print("Compliance-Anteile:")
    print(shares.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nKausale und naive Größen:")
    print(estimands.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nIV-Zerlegung:")
    print(diagnostics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    takeup = [by_assignment.loc[0, "treatment_rate"], by_assignment.loc[1, "treatment_rate"]]
    always = [always_share_inferred, always_share_inferred]
    compliers = [0, complier_share_inferred]
    axes[0].bar([0, 1], always, color="#595959", label="Always-Takers")
    axes[0].bar([0, 1], compliers, bottom=always, color="#4472C4", label="zusätzliche Compliers")
    axes[0].bar_label(axes[0].containers[-1],
                      labels=[f"Take-up {takeup[0]:.2f}", f"Take-up {takeup[1]:.2f}"], padding=4)
    axes[0].set(title="Nur Compliers verändern D wegen Z", xlabel="Zufällige Einladung Z",
                ylabel="Anteil mit Treatment D=1", xticks=[0, 1], ylim=(0, 1))
    axes[0].legend(frameon=False)

    plot_estimands = estimands.copy()
    y_pos = np.arange(len(plot_estimands))
    colors = ["#C00000", "#4472C4", "#4472C4", "#595959"]
    axes[1].scatter(plot_estimands["estimate"], y_pos, c=colors, s=70)
    for position, value in enumerate(plot_estimands["estimate"]):
        axes[1].text(value + 0.25, position, f"{value:.2f}", va="center")
    axes[1].set(title="IV identifiziert den Effekt der Compliers",
                xlabel="Treatment-Effekt", yticks=y_pos,
                yticklabels=["Naiver Vergleich", "Wald IV", "Wahrer Complier-Effekt", "Population ATE"])
    axes[1].axvline(oracle_late, color="#4472C4", linestyle="--", linewidth=1)
    axes[1].set_xlim(min(plot_estimands["estimate"].min() - 1, 0),
                     plot_estimands["estimate"].max() + 2)
    figure.suptitle("LATE ist lokal für Personen, deren Treatmentstatus durch Z verändert wird")
    figure.tight_layout()
    figure.savefig(RESULTS / "late_compliance_types.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
