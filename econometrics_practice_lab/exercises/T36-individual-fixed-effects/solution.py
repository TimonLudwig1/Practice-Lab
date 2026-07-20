"""Musterlösung für T36: Individuelle Fixed Effects."""

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
TRUE_BETA = 1.8


def make_data(people: int = 120, periods: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    ability = rng.normal(size=people)
    degree = (ability + rng.normal(0, 0.7, people) > 0).astype(int)
    rows = []
    for person in range(people):
        for period in range(1, periods + 1):
            training = 5 + 1.4 * ability[person] + 0.25 * period + rng.normal(0, 1.4)
            person_intercept = 50 + 7 * ability[person]
            productivity = person_intercept + TRUE_BETA * training + rng.normal(0, 3.5)
            rows.append((person + 1, period, training, productivity, ability[person], degree[person], person_intercept))
    return pd.DataFrame(rows, columns=["person_id", "period", "training_hours",
                                       "productivity_score", "oracle_ability", "university_degree",
                                       "oracle_person_intercept"])


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "employee_panel.csv", index=False)
    pooled = sm.OLS(
        data["productivity_score"], sm.add_constant(data[["training_hours"]])
    ).fit(cov_type="cluster", cov_kwds={"groups": data["person_id"]})
    person_dummies = pd.get_dummies(data["person_id"], prefix="person", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([data[["training_hours"]], person_dummies], axis=1))
    individual_fe = sm.OLS(data["productivity_score"], design).fit(
        cov_type="cluster", cov_kwds={"groups": data["person_id"]}
    )
    comparison = pd.DataFrame(
        [{"model": "Pooled OLS", "training_coefficient": pooled.params["training_hours"],
          "standard_error": pooled.bse["training_hours"], "true_coefficient": TRUE_BETA},
         {"model": "Individual fixed effects", "training_coefficient": individual_fe.params["training_hours"],
          "standard_error": individual_fe.bse["training_hours"], "true_coefficient": TRUE_BETA}]
    )
    assert abs(individual_fe.params["training_hours"] - TRUE_BETA) < 0.2
    assert abs(individual_fe.params["training_hours"] - TRUE_BETA) < abs(pooled.params["training_hours"] - TRUE_BETA)
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

    person_summary = (
        data.groupby("person_id")
        .agg(mean_training=("training_hours", "mean"),
             within_training_sd=("training_hours", "std"),
             oracle_person_intercept=("oracle_person_intercept", "first"),
             university_degree=("university_degree", "first"))
        .reset_index()
    )
    reference_intercept = individual_fe.params["const"]
    person_summary["estimated_person_intercept"] = [
        reference_intercept if person_id == 1
        else reference_intercept + individual_fe.params[f"person_{person_id}"]
        for person_id in person_summary["person_id"]
    ]
    person_summary["estimated_fe_centered"] = (
        person_summary["estimated_person_intercept"] - person_summary["estimated_person_intercept"].mean()
    )
    person_summary["oracle_fe_centered"] = (
        person_summary["oracle_person_intercept"] - person_summary["oracle_person_intercept"].mean()
    )
    fe_correlation = person_summary["estimated_fe_centered"].corr(person_summary["oracle_fe_centered"])
    assert fe_correlation > 0.85
    person_summary.to_csv(RESULTS / "individual_effects.csv", index=False)

    full_dummies = pd.get_dummies(data["person_id"], prefix="person", dtype=float)
    rank_design = pd.concat([full_dummies, data[["university_degree"]]], axis=1)
    diagnostics = pd.DataFrame(
        [{"columns_with_degree": rank_design.shape[1],
          "rank_with_degree": np.linalg.matrix_rank(rank_design.to_numpy()),
          "rank_deficient": np.linalg.matrix_rank(rank_design.to_numpy()) < rank_design.shape[1],
          "correlation_estimated_true_centered_fe": fe_correlation,
          "minimum_within_training_sd": person_summary["within_training_sd"].min()}]
    )
    assert bool(diagnostics.loc[0, "rank_deficient"])
    diagnostics.to_csv(RESULTS / "individual_fe_diagnostics.csv", index=False)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nDiagnostik:")
    print(diagnostics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    selected_people = [1, 8, 19, 34, 57, 83, 101, 118]
    for person_id in selected_people:
        subset = data[data["person_id"] == person_id]
        axes[0, 0].plot(subset["period"], subset["productivity_score"], marker="o",
                        alpha=0.75, label=f"Person {person_id}")
    axes[0, 0].set(title="Personen haben unterschiedliche Niveaus", xlabel="Periode",
                   ylabel="Produktivitätsscore", xticks=np.arange(1, 7))
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)

    axes[0, 1].scatter(person_summary["oracle_fe_centered"],
                       person_summary["estimated_fe_centered"], color="#4472C4", alpha=0.65, s=28)
    limits = [min(person_summary["oracle_fe_centered"].min(), person_summary["estimated_fe_centered"].min()),
              max(person_summary["oracle_fe_centered"].max(), person_summary["estimated_fe_centered"].max())]
    axes[0, 1].plot(limits, limits, color="#C00000", linestyle="--")
    axes[0, 1].set(title=f"Zentrierte individuelle Effekte (r={fe_correlation:.2f})",
                   xlabel="Wahrer personenspezifischer Intercept", ylabel="Geschätzter Intercept")

    positions = np.arange(len(comparison))
    axes[1, 0].errorbar(comparison["training_coefficient"], positions,
                        xerr=1.96 * comparison["standard_error"], fmt="o", color="#4472C4", capsize=5)
    axes[1, 0].axvline(TRUE_BETA, color="#C00000", linestyle="--", label="Wahrer Effekt")
    axes[1, 0].set(title="Personen-FE korrigieren dauerhafte Fähigkeit",
                   xlabel="Trainingseffekt mit 95%-KI", yticks=positions,
                   yticklabels=comparison["model"])
    axes[1, 0].legend(frameon=False)

    axes[1, 1].scatter(person_summary["mean_training"], person_summary["estimated_fe_centered"],
                       color="#4472C4", alpha=0.65, s=28)
    axes[1, 1].set(title="Trainingsniveau korreliert mit Personeneffekt",
                   xlabel="Mittlere Trainingsstunden", ylabel="Geschätzter zentrierter Personeneffekt")
    figure.suptitle("Individuelle Fixed Effects erlauben jeder Person ein eigenes Ausgangsniveau")
    figure.tight_layout()
    figure.savefig(RESULTS / "individual_fixed_effects.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
