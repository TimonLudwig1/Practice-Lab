"""Musterlösung für T19: Hypothesentests in multipler Regression."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720
REGRESSORS = ["experience_years", "education_years", "certifications", "leadership_training"]


def make_data(size: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    experience = rng.uniform(0, 30, size=size)
    education = rng.uniform(10, 20, size=size)
    certifications = rng.poisson(1.5, size=size)
    training = rng.binomial(1, 0.35, size=size)
    salary = 28_000 + 1_600 * experience + 2_400 * education
    salary += 1_300 * certifications + 4_000 * training + rng.normal(0, 8_000, size=size)
    return pd.DataFrame(
        {"employee_id": np.arange(1, size + 1), "experience_years": experience,
         "education_years": education, "certifications": certifications,
         "leadership_training": training, "annual_salary_eur": salary}
    )


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = make_data()
    data.to_csv(DATA / "employee_salaries.csv", index=False)
    unrestricted = sm.OLS(data["annual_salary_eur"], sm.add_constant(data[REGRESSORS])).fit()
    restricted_terms = ["experience_years", "education_years"]
    restricted = sm.OLS(
        data["annual_salary_eur"], sm.add_constant(data[restricted_terms])
    ).fit()

    ci = unrestricted.conf_int()
    coefficients = pd.DataFrame(
        {"term": unrestricted.params.index, "estimate": unrestricted.params.values,
         "standard_error": unrestricted.bse.values, "t_statistic": unrestricted.tvalues.values,
         "p_value": unrestricted.pvalues.values, "ci_95_low": ci[0].values,
         "ci_95_high": ci[1].values}
    )
    coefficients.to_csv(RESULTS / "coefficients.csv", index=False)

    training_t = unrestricted.tvalues["leadership_training"]
    training_p = unrestricted.pvalues["leadership_training"]
    equality_test = unrestricted.f_test("experience_years = education_years")
    q = 2
    rss_r = np.sum(restricted.resid ** 2)
    rss_u = np.sum(unrestricted.resid ** 2)
    df_denominator = int(unrestricted.df_resid)
    manual_f = ((rss_r - rss_u) / q) / (rss_u / df_denominator)
    f_pvalue = stats.f.sf(manual_f, q, df_denominator)
    f_critical = stats.f.ppf(0.95, q, df_denominator)
    statsmodels_joint = unrestricted.f_test("certifications = 0, leadership_training = 0")
    assert np.isclose(manual_f, float(statsmodels_joint.fvalue))

    tests = pd.DataFrame(
        [
            {"test": "training = 0", "statistic_type": "t", "statistic": training_t,
             "df_numerator": 1, "df_denominator": df_denominator,
             "p_value": training_p, "critical_value_5_percent": stats.t.ppf(0.975, df_denominator)},
            {"test": "experience = education", "statistic_type": "F", "statistic": float(equality_test.fvalue),
             "df_numerator": 1, "df_denominator": df_denominator,
             "p_value": float(equality_test.pvalue),
             "critical_value_5_percent": stats.f.ppf(0.95, 1, df_denominator)},
            {"test": "certifications = training = 0", "statistic_type": "F",
             "statistic": manual_f, "df_numerator": q, "df_denominator": df_denominator,
             "p_value": f_pvalue, "critical_value_5_percent": f_critical},
        ]
    )
    tests["reject_at_5_percent"] = tests["p_value"] < 0.05
    assert (tests["statistic"].abs() > tests["critical_value_5_percent"]).equals(
        tests["reject_at_5_percent"]
    )
    tests.to_csv(RESULTS / "hypothesis_tests.csv", index=False)
    print(tests.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    plot_coefficients = coefficients[coefficients["term"] != "const"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    estimates = plot_coefficients["estimate"].to_numpy()
    left = estimates - plot_coefficients["ci_95_low"].to_numpy()
    right = plot_coefficients["ci_95_high"].to_numpy() - estimates
    positions = np.arange(len(plot_coefficients))
    axes[0].errorbar(estimates, positions, xerr=np.vstack([left, right]), fmt="o",
                     color="#4472C4", capsize=5)
    axes[0].axvline(0, color="#C00000", linestyle="--")
    axes[0].set(title="Koeffizienten mit 95%-KI", xlabel="Euro pro Einheit",
                yticks=positions, yticklabels=plot_coefficients["term"])

    x_values = np.linspace(0, max(manual_f * 1.25, f_critical * 2), 500)
    density = stats.f.pdf(x_values, q, df_denominator)
    axes[1].plot(x_values, density, color="#4472C4", linewidth=2)
    rejection = x_values >= f_critical
    axes[1].fill_between(x_values[rejection], density[rejection], color="#C00000", alpha=0.3,
                         label="Ablehnungsbereich")
    axes[1].axvline(f_critical, color="#C00000", linestyle="--",
                    label=f"Kritisch = {f_critical:.2f}")
    axes[1].axvline(manual_f, color="#595959", linewidth=2,
                    label=f"Beobachtet = {manual_f:.2f}")
    axes[1].set(title="Gemeinsamer F-Test (q=2)", xlabel="F-Statistik", ylabel="Dichte")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(RESULTS / "multiple_tests.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
