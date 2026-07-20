"""Musterlösung für T09: Einfache lineare Regression und Gauss–Markov."""

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


def make_data(size: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    training = rng.uniform(2, 25, size=size)
    structural_error = rng.normal(0, 8, size=size)
    productivity = 50 + 2.2 * training + structural_error
    return pd.DataFrame(
        {"firm_id": np.arange(1, size + 1), "training_hours": training,
         "productivity_index": productivity, "structural_error": structural_error}
    )


def manual_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    intercept = y.mean() - slope * x.mean()
    return float(intercept), float(slope)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "firm_training.csv", index=False)
    x = data["training_hours"].to_numpy()
    y = data["productivity_index"].to_numpy()

    manual_intercept, manual_slope = manual_ols(x, y)
    model = sm.OLS(y, sm.add_constant(x)).fit()
    fitted = model.fittedvalues
    residuals = model.resid
    assert np.allclose([manual_intercept, manual_slope], model.params)

    summary = pd.DataFrame(
        {
            "parameter": ["intercept", "training_hours"],
            "estimate": model.params,
            "standard_error": model.bse,
            "true_value": [50.0, 2.2],
        }
    )
    summary.to_csv(RESULTS / "model_summary.csv", index=False)

    data["training_quintile"] = pd.qcut(data["training_hours"], 5, labels=False) + 1
    gm_groups = (
        data.groupby("training_quintile", observed=True)
        .agg(mean_training=("training_hours", "mean"), mean_error=("structural_error", "mean"),
             variance_error=("structural_error", "var"), n=("firm_id", "size"))
        .reset_index()
    )
    gm_groups.to_csv(RESULTS / "gm_by_training_quintile.csv", index=False)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        f"\nOLS-Bedingungen: Summe Residuen={residuals.sum():.3e}, "
        f"Cov(X, Residuum)={np.cov(x, residuals, ddof=1)[0, 1]:.3e}"
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    order = np.argsort(x)
    axes[0, 0].scatter(x, y, color="#5B9BD5", alpha=0.72, s=26)
    axes[0, 0].plot(x[order], fitted[order], color="#C00000", linewidth=2,
                    label=f"ŷ = {model.params[0]:.1f} + {model.params[1]:.2f}x")
    axes[0, 0].set(title="Produktivität und OLS-Gerade", xlabel="Trainingsstunden",
                   ylabel="Produktivitätsindex")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].scatter(x, residuals, color="#4472C4", alpha=0.72, s=26)
    axes[0, 1].axhline(0, color="#C00000", linestyle="--")
    axes[0, 1].set(title="Residuen ohne systematisches Muster", xlabel="Trainingsstunden",
                   ylabel="Residuum")

    axes[1, 0].errorbar(gm_groups["mean_training"], gm_groups["mean_error"],
                        yerr=np.sqrt(gm_groups["variance_error"] / gm_groups["n"]),
                        fmt="o", capsize=5, color="#4472C4")
    axes[1, 0].axhline(0, color="#C00000", linestyle="--")
    axes[1, 0].set(title="Mittlerer wahrer Fehler nach X-Quintil", xlabel="Mittlere Trainingsstunden",
                   ylabel="Mittlerer struktureller Fehler ± SE")

    axes[1, 1].bar(gm_groups["training_quintile"].astype(str), gm_groups["variance_error"],
                   color="#70AD47")
    axes[1, 1].axhline(np.var(data["structural_error"], ddof=1), color="#595959",
                       linestyle="--", label="Gesamtvarianz")
    axes[1, 1].set(title="Fehlervarianz nach X-Quintil", xlabel="Trainings-Quintil",
                   ylabel="Varianz des strukturellen Fehlers")
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Einfache OLS-Regression unter Gauss–Markov")
    figure.tight_layout()
    figure.savefig(RESULTS / "gm_diagnostics.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
