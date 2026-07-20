"""Musterlösung für T17: R² und Varianzzerlegung."""

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


def make_data(size: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    study_hours = rng.uniform(0, 10, size=size)
    exam_score = 45 + 4.5 * study_hours + rng.normal(0, 12, size=size)
    irrelevant_noise = rng.normal(size=size)
    return pd.DataFrame(
        {"student_id": np.arange(1, size + 1), "study_hours": study_hours,
         "irrelevant_noise": irrelevant_noise, "exam_score": exam_score}
    )


def decompose(model, y: np.ndarray, name: str) -> dict[str, float | str]:
    fitted = np.asarray(model.fittedvalues)
    residual = y - fitted
    tss = np.sum((y - y.mean()) ** 2)
    rss = np.sum(residual ** 2)
    ess = np.sum((fitted - y.mean()) ** 2)
    return {
        "model": name,
        "tss": tss,
        "ess": ess,
        "rss": rss,
        "identity_error_tss_minus_ess_rss": tss - ess - rss,
        "r_squared_one_minus_rss_tss": 1 - rss / tss,
        "r_squared_ess_tss": ess / tss,
        "statsmodels_r_squared": model.rsquared,
        "adjusted_r_squared": model.rsquared_adj,
    }


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "study_scores.csv", index=False)
    y = data["exam_score"].to_numpy()

    intercept_only = sm.OLS(y, np.ones((len(data), 1))).fit()
    hours = sm.OLS(y, sm.add_constant(data[["study_hours"]])).fit()
    augmented = sm.OLS(
        y, sm.add_constant(data[["study_hours", "irrelevant_noise"]])
    ).fit()
    models = {
        "Intercept only": intercept_only,
        "Study hours": hours,
        "Hours + irrelevant noise": augmented,
    }
    decomposition = pd.DataFrame([decompose(model, y, name) for name, model in models.items()])
    decomposition.to_csv(RESULTS / "r_squared_decomposition.csv", index=False)
    assert np.allclose(decomposition["r_squared_one_minus_rss_tss"],
                       decomposition["statsmodels_r_squared"])
    print(decomposition.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    order = np.argsort(data["study_hours"].to_numpy())
    axes[0, 0].scatter(data["study_hours"], y, color="#5B9BD5", alpha=0.58, s=23)
    axes[0, 0].axhline(y.mean(), color="#595959", linestyle=":", label="Scoremittel")
    axes[0, 0].plot(data["study_hours"].to_numpy()[order], hours.fittedvalues[order],
                    color="#C00000", linewidth=2, label="OLS-Fit")
    axes[0, 0].set(title="Fit reduziert Abweichungen vom Mittel", xlabel="Lernstunden pro Woche",
                   ylabel="Testscore")
    axes[0, 0].legend(frameon=False)

    model_labels = decomposition["model"].tolist()[1:]
    explained = decomposition["ess"].to_numpy()[1:] / decomposition["tss"].to_numpy()[1:]
    unexplained = decomposition["rss"].to_numpy()[1:] / decomposition["tss"].to_numpy()[1:]
    axes[0, 1].barh(model_labels, explained, color="#4472C4", label="Erklärt: ESS/TSS")
    axes[0, 1].barh(model_labels, unexplained, left=explained, color="#BFBFBF",
                    label="Unerklärt: RSS/TSS")
    axes[0, 1].set(title="Gesamtvariation wird vollständig zerlegt", xlabel="Anteil an TSS",
                   xlim=(0, 1))
    axes[0, 1].legend(frameon=False)

    positions = np.arange(len(model_labels))
    width = 0.36
    r2_values = decomposition["statsmodels_r_squared"].to_numpy()[1:]
    adjusted_values = decomposition["adjusted_r_squared"].to_numpy()[1:]
    r2_bars = axes[1, 0].bar(positions - width / 2, r2_values, width, color="#4472C4",
                             label="R²")
    adjusted_bars = axes[1, 0].bar(positions + width / 2, adjusted_values, width,
                                   color="#70AD47", label="Adjustiertes R²")
    for bar, value in zip([*r2_bars, *adjusted_bars], [*r2_values, *adjusted_values]):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.4f}",
                        ha="center", va="bottom", fontsize=8)
    axes[1, 0].set(title="Irrelevanter Regressor: R² steigt mechanisch", ylabel="Gütemaß",
                   xticks=positions, xticklabels=model_labels, ylim=(0, 1))
    axes[1, 0].legend(frameon=False)

    axes[1, 1].scatter(hours.fittedvalues, y, color="#5B9BD5", alpha=0.58, s=23)
    limits = [min(hours.fittedvalues.min(), y.min()), max(hours.fittedvalues.max(), y.max())]
    axes[1, 1].plot(limits, limits, color="#C00000", linestyle="--",
                    label="Beobachtet = Fit")
    axes[1, 1].set(title=f"Beobachtet gegen Fit (R²={hours.rsquared:.3f})",
                   xlabel="Vorhergesagter Score", ylabel="Beobachteter Score")
    axes[1, 1].legend(frameon=False)
    figure.suptitle("R² misst In-Sample-Anpassung relativ zum Mittelwert")
    figure.tight_layout()
    figure.savefig(RESULTS / "r_squared_visual.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
