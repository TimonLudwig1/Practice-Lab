"""Musterlösung für T47: First Stage und schwache Instrumente."""

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
RNG_SEED = 20260727
TRUE_EFFECT = 1.5


def make_sample(rng: np.random.Generator, strength: float, observations: int = 500) -> pd.DataFrame:
    instrument = rng.binomial(1, 0.5, observations)
    motivation = rng.normal(size=observations)
    coaching = 3 + strength * instrument + 1.2 * motivation + rng.normal(0, 1.5, observations)
    outcome = 10 + TRUE_EFFECT * coaching + 3 * motivation + rng.normal(0, 2, observations)
    return pd.DataFrame({"instrument": instrument, "coaching_hours": coaching,
                         "performance": outcome, "oracle_motivation": motivation})


def first_stage_statistics(data: pd.DataFrame) -> dict[str, float]:
    model = sm.OLS(data["coaching_hours"], sm.add_constant(data[["instrument"]])).fit()
    return {"coefficient": model.params["instrument"],
            "standard_error": model.bse["instrument"],
            "f_statistic": model.tvalues["instrument"] ** 2,
            "partial_r_squared": model.rsquared,
            "p_value": model.pvalues["instrument"]}


def fast_statistics(data: pd.DataFrame) -> tuple[float, float]:
    z = data["instrument"].to_numpy()
    x = data["coaching_hours"].to_numpy()
    y = data["performance"].to_numpy()
    z_centered = z - z.mean()
    denominator = np.sum(z_centered ** 2)
    first_stage = np.sum(z_centered * x) / denominator
    intercept = x.mean() - first_stage * z.mean()
    residual = x - intercept - first_stage * z
    standard_error = np.sqrt(np.sum(residual ** 2) / (len(z) - 2) / denominator)
    f_statistic = (first_stage / standard_error) ** 2
    iv_estimate = np.sum(z_centered * y) / np.sum(z_centered * x)
    return f_statistic, iv_estimate


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    samples = {"Strong instrument": make_sample(rng, strength=2.0),
               "Weak instrument": make_sample(rng, strength=0.2)}
    sample_frames = []
    diagnostic_rows = []
    for label, sample in samples.items():
        sample = sample.copy()
        sample["design"] = label
        sample_frames.append(sample)
        diagnostic_rows.append({"design": label, **first_stage_statistics(sample)})
    pd.concat(sample_frames, ignore_index=True).to_csv(DATA / "first_stage_examples.csv", index=False)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(RESULTS / "first_stage_diagnostics.csv", index=False)
    assert diagnostics.loc[0, "f_statistic"] > 100
    assert diagnostics.loc[1, "f_statistic"] < 10

    simulation_rows = []
    for design, strength in [("Strong instrument", 2.0), ("Weak instrument", 0.2)]:
        for repetition in range(500):
            sample = make_sample(rng, strength=strength, observations=400)
            f_statistic, iv_estimate = fast_statistics(sample)
            simulation_rows.append({"design": design, "repetition": repetition,
                                    "first_stage_f": f_statistic, "iv_estimate": iv_estimate,
                                    "absolute_error": abs(iv_estimate - TRUE_EFFECT)})
    simulations = pd.DataFrame(simulation_rows)
    simulations.to_csv(RESULTS / "weak_instrument_monte_carlo.csv", index=False)
    summary = simulations.groupby("design").agg(
        median_f=("first_stage_f", "median"), median_iv=("iv_estimate", "median"),
        iv_standard_deviation=("iv_estimate", "std"),
        median_absolute_error=("absolute_error", "median"),
        share_abs_estimate_above_10=("iv_estimate", lambda values: np.mean(np.abs(values) > 10)),
    ).reset_index()
    summary.to_csv(RESULTS / "weak_instrument_summary.csv", index=False)
    strong_error = summary.loc[summary["design"].eq("Strong instrument"), "median_absolute_error"].iloc[0]
    weak_error = summary.loc[summary["design"].eq("Weak instrument"), "median_absolute_error"].iloc[0]
    assert weak_error > strong_error * 3

    print("First-Stage-Diagnostik der Beispieldaten:")
    print(diagnostics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMonte-Carlo-Zusammenfassung:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    display_labels = {"Strong instrument": "Starkes Instrument",
                      "Weak instrument": "Schwaches Instrument"}
    for index, (label, sample) in enumerate(samples.items()):
        means = sample.groupby("instrument")["coaching_hours"].mean()
        axes[0, 0].plot([0, 1], means.values, marker="o", linewidth=2,
                        color=["#4472C4", "#C00000"][index], label=display_labels[label])
    axes[0, 0].set(title="First Stage: Wie stark verschiebt Z das Treatment?",
                   xlabel="Instrument Z", ylabel="Mittlere Coachingstunden", xticks=[0, 1])
    axes[0, 0].legend(frameon=False)

    for label, color in [("Strong instrument", "#4472C4"), ("Weak instrument", "#C00000")]:
        subset = simulations.loc[simulations["design"].eq(label), "first_stage_f"]
        axes[0, 1].hist(np.clip(subset, 0.1, 300), bins=np.logspace(-1, np.log10(300), 35),
                        alpha=0.55, color=color, label=display_labels[label])
    axes[0, 1].axvline(10, color="#595959", linestyle="--", label="Faustregel F=10")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set(title="First-Stage-F über Wiederholungen",
                   xlabel="F-Statistik (Log-Skala)", ylabel="Anzahl")
    axes[0, 1].legend(frameon=False)

    bins = np.linspace(-5, 8, 55)
    for label, color in [("Strong instrument", "#4472C4"), ("Weak instrument", "#C00000")]:
        subset = simulations.loc[simulations["design"].eq(label), "iv_estimate"]
        axes[1, 0].hist(subset, bins=bins, alpha=0.55, color=color, label=display_labels[label])
    axes[1, 0].axvline(TRUE_EFFECT, color="#595959", linestyle="--", label="Wahrer Effekt")
    axes[1, 0].set(title="Schwaches IV erzeugt eine breite, schwere Verteilung",
                   xlabel="IV-Schätzer (Ausschnitt −5 bis 8)", ylabel="Anzahl")
    axes[1, 0].legend(frameon=False)

    plot_sample = simulations.sample(500, random_state=RNG_SEED)
    colors = plot_sample["design"].map({"Strong instrument": "#4472C4",
                                         "Weak instrument": "#C00000"})
    axes[1, 1].scatter(plot_sample["first_stage_f"], np.minimum(plot_sample["absolute_error"], 10),
                       c=colors, alpha=0.45, s=22)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set(title="Kleine Nenner machen den Quotienten empfindlich",
                   xlabel="First-Stage-F (Log-Skala)", ylabel="Absoluter Fehler (bei 10 gekappt)")
    figure.suptitle("Instrumentenstärke bestimmt die Stabilität der IV-Schätzung")
    figure.tight_layout()
    figure.savefig(RESULTS / "first_stage_strength.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
