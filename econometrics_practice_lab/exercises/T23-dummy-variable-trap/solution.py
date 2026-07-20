"""Musterlösung für T23: Dummy-Variable-Trap und Matrixrang."""

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
MODES = ["Bus", "Bike", "Car"]


def make_data(size: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    mode = rng.choice(MODES, size=size, p=[0.4, 0.25, 0.35])
    commute = rng.uniform(10, 60, size=size)
    mode_effect = {"Bus": 0, "Bike": 14, "Car": 7}
    satisfaction = 72 - 0.38 * commute + np.array([mode_effect[value] for value in mode])
    satisfaction += rng.normal(0, 6, size=size)
    return pd.DataFrame(
        {"commuter_id": np.arange(1, size + 1), "mode": mode,
         "commute_minutes": commute, "satisfaction_score": satisfaction}
    )


def diagnostics(name: str, design: pd.DataFrame) -> dict[str, float | str | int | bool]:
    singular_values = np.linalg.svd(design.to_numpy(), compute_uv=False)
    rank = np.linalg.matrix_rank(design.to_numpy())
    return {"design": name, "rows": design.shape[0], "columns": design.shape[1],
            "rank": rank, "rank_deficient": rank < design.shape[1],
            "smallest_singular_value": singular_values.min(),
            "condition_number": singular_values.max() / singular_values.min()}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "commuter_modes.csv", index=False)
    dummies = pd.get_dummies(data["mode"], prefix="mode", dtype=float)
    assert np.allclose(dummies.sum(axis=1), 1)

    invalid = sm.add_constant(pd.concat([dummies, data[["commute_minutes"]]], axis=1))
    reference = sm.add_constant(pd.concat(
        [dummies.drop(columns="mode_Bus"), data[["commute_minutes"]]], axis=1
    ))
    no_intercept = pd.concat([dummies, data[["commute_minutes"]]], axis=1)
    designs = {"Intercept + all dummies": invalid, "Reference Bus": reference,
               "No intercept + all dummies": no_intercept}
    diagnostic_table = pd.DataFrame([diagnostics(name, design) for name, design in designs.items()])
    rank_status = diagnostic_table.set_index("design")["rank_deficient"]
    assert bool(rank_status["Intercept + all dummies"])
    assert not bool(rank_status["Reference Bus"])
    assert not bool(rank_status["No intercept + all dummies"])
    diagnostic_table.to_csv(RESULTS / "design_diagnostics.csv", index=False)

    y = data["satisfaction_score"]
    invalid_model = sm.OLS(y, invalid).fit()
    reference_model = sm.OLS(y, reference).fit()
    no_intercept_model = sm.OLS(y, no_intercept).fit()
    shift_vector = pd.Series(0.0, index=invalid.columns)
    shift_vector["const"] = 10
    for column in dummies.columns:
        shift_vector[column] = -10
    shifted_params = invalid_model.params + shift_vector
    assert np.allclose(invalid @ invalid_model.params, invalid @ shifted_params)
    assert np.allclose(reference_model.fittedvalues, no_intercept_model.fittedvalues)

    parameter_rows = []
    for name, params in (
        ("Invalid pseudoinverse", invalid_model.params),
        ("Invalid shifted", shifted_params),
        ("Reference Bus", reference_model.params),
        ("No intercept", no_intercept_model.params),
    ):
        parameter_rows.extend(
            {"parameterization": name, "term": term, "estimate": value}
            for term, value in params.items()
        )
    parameters = pd.DataFrame(parameter_rows)
    parameters.to_csv(RESULTS / "parameterizations.csv", index=False)

    adjusted_rows = []
    for mode in MODES:
        ref_row = pd.DataFrame([{column: 0.0 for column in reference.columns}])
        ref_row["const"] = 1.0
        ref_row["commute_minutes"] = 30.0
        if mode != "Bus":
            ref_row[f"mode_{mode}"] = 1.0
        no_row = pd.DataFrame([{column: 0.0 for column in no_intercept.columns}])
        no_row["commute_minutes"] = 30.0
        no_row[f"mode_{mode}"] = 1.0
        adjusted_rows.append(
            {"mode": mode, "commute_minutes": 30,
             "prediction_reference_model": float(reference_model.predict(ref_row)[0]),
             "prediction_no_intercept_model": float(no_intercept_model.predict(no_row)[0])}
        )
    adjusted = pd.DataFrame(adjusted_rows)
    assert np.allclose(
        adjusted["prediction_reference_model"], adjusted["prediction_no_intercept_model"]
    )
    adjusted.to_csv(RESULTS / "adjusted_mode_means.csv", index=False)
    print(diagnostic_table.to_string(index=False, float_format=lambda value: f"{value:.3e}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].imshow(invalid[["const", *dummies.columns]].iloc[:30].to_numpy(),
                      aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[0, 0].set(title="Intercept und vollständige Dummies",
                   xlabel="Designspalten", ylabel="Erste 30 Beobachtungen",
                   xticks=np.arange(4), xticklabels=["const", "Bike", "Bus", "Car"])

    singular = np.linalg.svd(invalid.to_numpy(), compute_uv=False)
    axes[0, 1].bar(np.arange(1, len(singular) + 1), singular, color="#4472C4")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="Ein Singulärwert ist praktisch null", xlabel="Singulärwert-Index",
                   ylabel="Singulärwert (log)")

    invalid_params = parameters[parameters["parameterization"].isin(
        ["Invalid pseudoinverse", "Invalid shifted"]
    ) & parameters["term"].isin(["const", *dummies.columns])]
    pivot = invalid_params.pivot(index="term", columns="parameterization", values="estimate")
    positions = np.arange(len(pivot))
    width = 0.36
    axes[1, 0].bar(positions - width / 2, pivot["Invalid pseudoinverse"], width,
                   color="#4472C4", label="Pseudoinverse")
    axes[1, 0].bar(positions + width / 2, pivot["Invalid shifted"], width,
                   color="#ED7D31", label="Verschoben")
    axes[1, 0].set(title="Verschiedene Koeffizienten, identische Fits", xticks=positions,
                   xticklabels=pivot.index, ylabel="Koeffizient")
    axes[1, 0].legend(frameon=False)

    positions = np.arange(len(adjusted))
    width = 0.36
    axes[1, 1].bar(positions - width / 2, adjusted["prediction_reference_model"], width,
                   color="#4472C4", label="Referenzmodell")
    axes[1, 1].bar(positions + width / 2, adjusted["prediction_no_intercept_model"], width,
                   color="#70AD47", label="Ohne Intercept")
    axes[1, 1].set(title="Gültige Parametrisierungen stimmen überein",
                   xticks=positions, xticklabels=adjusted["mode"],
                   ylabel="Zufriedenheit bei 30 Minuten")
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Die Dummy-Variable-Trap macht Koeffizienten nicht eindeutig")
    figure.tight_layout()
    figure.savefig(RESULTS / "dummy_trap.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
