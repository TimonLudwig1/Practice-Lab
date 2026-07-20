"""Musterlösung für T13: Level- und Log-Modelle."""

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


def make_data(size_per_scenario: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    specs = []

    x = rng.uniform(1, 20, size_per_scenario)
    specs.append(("level_level", x, 12 + 3.2 * x + rng.normal(0, 5, size_per_scenario)))

    x = rng.uniform(9, 20, size_per_scenario)
    specs.append(("log_level", x, np.exp(2.4 + 0.075 * x + rng.normal(0, 0.12, size_per_scenario))))

    x = rng.lognormal(8.0, 0.8, size_per_scenario)
    specs.append(("level_log", x, 30 + 18 * np.log(x) + rng.normal(0, 12, size_per_scenario)))

    x = rng.uniform(5, 30, size_per_scenario)
    specs.append(("log_log", x, np.exp(7.0 - 1.3 * np.log(x) + rng.normal(0, 0.18, size_per_scenario))))

    frames = []
    for scenario, x, y in specs:
        frames.append(pd.DataFrame({"scenario": scenario, "observation_id": np.arange(1, len(x) + 1),
                                    "x": x, "y": y}))
    return pd.concat(frames, ignore_index=True)


def transform_for_model(group: pd.DataFrame, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    x = group["x"].to_numpy()
    y = group["y"].to_numpy()
    if scenario == "level_level":
        return x, y
    if scenario == "log_level":
        return x, np.log(y)
    if scenario == "level_log":
        return np.log(x), y
    if scenario == "log_log":
        return np.log(x), np.log(y)
    raise ValueError(f"Unbekanntes Szenario: {scenario}")


def raw_scale_prediction(model, x: np.ndarray, scenario: str) -> np.ndarray:
    transformed_x = np.log(x) if scenario in {"level_log", "log_log"} else x
    linear_prediction = model.predict(sm.add_constant(transformed_x))
    return np.exp(linear_prediction) if scenario in {"log_level", "log_log"} else linear_prediction


def interpretation(beta: float, scenario: str) -> tuple[float, str]:
    if scenario == "level_level":
        return beta, "Y-Einheiten je +1 X"
    if scenario == "log_level":
        return 100 * (np.exp(beta) - 1), "exakte % Y je +1 X"
    if scenario == "level_log":
        return beta / 100, "Y-Einheiten je +1% X"
    return beta, "% Y je +1% X (Elastizität)"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    assert (data[["x", "y"]] > 0).all().all()
    data.to_csv(DATA / "log_model_scenarios.csv", index=False)

    models = {}
    rows = []
    for scenario, group in data.groupby("scenario", sort=False):
        x_model, y_model = transform_for_model(group, scenario)
        model = sm.OLS(y_model, sm.add_constant(x_model)).fit()
        models[scenario] = model
        effect, effect_unit = interpretation(model.params[1], scenario)
        rows.append(
            {"model": scenario, "intercept": model.params[0], "beta_1": model.params[1],
             "standard_error_beta_1": model.bse[1], "r_squared_transformed": model.rsquared,
             "interpreted_effect": effect, "effect_unit": effect_unit}
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "log_model_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    titles = {
        "level_level": "Level-Level: Einheiten pro Einheit",
        "log_level": "Log-Level: Prozent pro Einheit",
        "level_log": "Level-Log: Einheiten pro Prozent",
        "log_log": "Log-Log: Elastizität",
    }
    axis_labels = {
        "level_level": ("Distanz (km)", "Lieferzeit (Min.)"),
        "log_level": ("Bildungsjahre", "Stundenlohn (€)"),
        "level_log": ("Werbeausgaben (€)", "Neue Kunden"),
        "log_log": ("Preis (€)", "Nachfrage (Einheiten)"),
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, scenario in zip(axes.flat, titles):
        group = data[data["scenario"] == scenario]
        x = group["x"].to_numpy()
        y = group["y"].to_numpy()
        order = np.argsort(x)
        fitted_raw = raw_scale_prediction(models[scenario], x, scenario)
        beta = models[scenario].params[1]
        axis.scatter(x, y, color="#5B9BD5", alpha=0.58, s=23)
        axis.plot(x[order], fitted_raw[order], color="#C00000", linewidth=2)
        axis.set(title=f"{titles[scenario]}\nβ₁ = {beta:.3f}",
                 xlabel=axis_labels[scenario][0], ylabel=axis_labels[scenario][1])
    figure.suptitle("Vier funktionale Formen in Originaleinheiten")
    figure.tight_layout()
    figure.savefig(RESULTS / "log_model_fits.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
