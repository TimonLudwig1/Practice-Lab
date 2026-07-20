"""Musterlösung für T48: Reduced Form."""

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
RNG_SEED = 20260728
TRUE_EFFECT = 1.8


def make_data(observations: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    invitation = rng.binomial(1, 0.5, observations)
    drive = rng.normal(size=observations)
    sessions = 1.5 + 1.7 * invitation + 1.1 * drive + rng.normal(0, 1.2, observations)
    job_score = 45 + TRUE_EFFECT * sessions + 3.2 * drive + rng.normal(0, 2.2, observations)
    return pd.DataFrame({"invitation": invitation, "counseling_sessions": sessions,
                         "job_search_score": job_score, "oracle_drive": drive})


def components(data: pd.DataFrame) -> tuple[float, float, float]:
    z = data["invitation"]
    first_stage = data.loc[z.eq(1), "counseling_sessions"].mean() - data.loc[z.eq(0), "counseling_sessions"].mean()
    reduced_form = data.loc[z.eq(1), "job_search_score"].mean() - data.loc[z.eq(0), "job_search_score"].mean()
    return first_stage, reduced_form, reduced_form / first_stage


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "job_counseling_data.csv", index=False)

    first_model = sm.OLS(data["counseling_sessions"], sm.add_constant(data[["invitation"]])).fit()
    reduced_model = sm.OLS(data["job_search_score"], sm.add_constant(data[["invitation"]])).fit()
    first_stage, reduced_form, iv_ratio = components(data)
    identity_error = abs(reduced_form - first_stage * iv_ratio)
    assert abs(first_stage - first_model.params["invitation"]) < 1e-12
    assert abs(reduced_form - reduced_model.params["invitation"]) < 1e-12
    assert identity_error < 1e-12
    assert abs(iv_ratio - TRUE_EFFECT) < 0.3

    component_table = pd.DataFrame([{
        "first_stage_sessions_per_invitation": first_stage,
        "reduced_form_score_points_per_invitation": reduced_form,
        "wald_iv_score_points_per_session": iv_ratio,
        "product_identity_error": identity_error,
        "first_stage_f": first_model.tvalues["invitation"] ** 2,
    }])
    component_table.to_csv(RESULTS / "reduced_form_components.csv", index=False)
    group_means = data.groupby("invitation", as_index=False).agg(
        mean_sessions=("counseling_sessions", "mean"),
        mean_job_score=("job_search_score", "mean"),
        mean_oracle_drive=("oracle_drive", "mean"),
    )
    group_means.to_csv(RESULTS / "instrument_group_means.csv", index=False)

    rng = np.random.default_rng(RNG_SEED + 1)
    bootstrap_rows = []
    for repetition in range(500):
        indices = rng.integers(0, len(data), len(data))
        sample = data.iloc[indices]
        fs, rf, ratio = components(sample)
        bootstrap_rows.append({"repetition": repetition, "first_stage": fs,
                               "reduced_form": rf, "iv_ratio": ratio})
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(RESULTS / "reduced_form_bootstrap.csv", index=False)

    print(component_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nInstrumentgruppen:")
    print(group_means.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot([0, 1], group_means["mean_sessions"], marker="o", color="#4472C4", linewidth=2.2)
    axes[0].annotate(f"First Stage = {first_stage:.2f}", xy=(1, group_means.loc[1, "mean_sessions"]),
                     xytext=(0.18, group_means["mean_sessions"].mean()),
                     arrowprops={"arrowstyle": "->", "color": "#4472C4"})
    axes[0].set(title="Z verändert die Sitzungszahl", xlabel="Einladung Z",
                ylabel="Mittlere Beratungssitzungen", xticks=[0, 1])

    axes[1].plot([0, 1], group_means["mean_job_score"], marker="o", color="#C00000", linewidth=2.2)
    axes[1].annotate(f"Reduced Form = {reduced_form:.2f}",
                     xy=(1, group_means.loc[1, "mean_job_score"]),
                     xytext=(0.05, group_means["mean_job_score"].mean()),
                     arrowprops={"arrowstyle": "->", "color": "#C00000"})
    axes[1].set(title="Z verändert das Outcome insgesamt", xlabel="Einladung Z",
                ylabel="Mittlerer Bewerbungsindex", xticks=[0, 1])

    axes[2].scatter(bootstrap["first_stage"], bootstrap["reduced_form"],
                    color="#9DC3E6", alpha=0.5, s=25, label="Bootstrap")
    x_grid = np.linspace(bootstrap["first_stage"].min(), bootstrap["first_stage"].max(), 100)
    axes[2].plot(x_grid, iv_ratio * x_grid, color="#4472C4", linewidth=2,
                 label=f"Reduced Form = {iv_ratio:.2f} × First Stage")
    axes[2].set(title="Der Wald-Schätzer ist die Steigung des Ratios",
                xlabel="First Stage", ylabel="Reduced Form")
    axes[2].legend(frameon=False)
    figure.suptitle("Reduced Form: Gesamtwirkung des Instruments auf das Outcome")
    figure.tight_layout()
    figure.savefig(RESULTS / "reduced_form_decomposition.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
