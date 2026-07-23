"""Tests for the data generator and the statistical tools.  Call:  python test_study.py"""
import numpy as np
from scipy import stats

from generate_study import (generate_study, generate_naive_without_counterbalancing, TRUTH)
from stats_tools import (cohen_dz, rank_biserial, paired_comparison, bonferroni,
                         holm_bonferroni, order_effect)

DF = generate_study(n_participants=24, seed=3)


def _pairs(df, column):
    a = df[df.condition == "3DoF"].sort_values("participant")[column].values
    b = df[df.condition == "6DoF"].sort_values("participant")[column].values
    return a, b


# ------------------------------ the generator ------------------------------
def test_generator_is_reproducible():
    assert generate_study(seed=3).equals(generate_study(seed=3))
    assert not generate_study(seed=3).equals(generate_study(seed=4))


def test_within_subject_structure():
    assert DF.participant.nunique() == 24
    assert len(DF) == 48                                 # 2 rows per participant
    assert set(DF.condition) == {"3DoF", "6DoF"}
    for _, g in DF.groupby("participant"):
        assert set(g.condition) == {"3DoF", "6DoF"}      # everybody has both


def test_counterbalancing_is_balanced():
    firsts = DF[DF.position == 1].condition.value_counts()
    assert firsts["3DoF"] == firsts["6DoF"] == 12        # 12 start with each condition


def test_values_are_in_valid_ranges():
    assert DF.presence.between(1, 7).all()
    assert DF.comfort.between(1, 7).all()
    assert (DF.sickness >= 0).all()
    assert (DF.time >= 5).all()


# ------------------------------ the effect sizes ------------------------------
def test_cohen_dz_by_hand():
    # diffs [1,2,3] -> mean 2, sd 1 -> dz = 2
    assert np.isclose(cohen_dz([2, 4, 6], [1, 2, 3]), 2.0)


def test_cohen_dz_sign():
    assert cohen_dz([5, 6, 7], [1, 1, 1]) > 0            # x > y (non-degenerate)
    assert cohen_dz([1, 1, 1], [5, 6, 7]) < 0


def test_rank_biserial_extremes():
    assert np.isclose(rank_biserial([2, 3, 4], [1, 1, 1]), 1.0)    # all diffs positive
    assert np.isclose(rank_biserial([1, 1, 1], [2, 3, 4]), -1.0)   # all negative


def test_rank_biserial_symmetric_near_zero():
    x = np.array([1, 2, 3, 4]); y = np.array([2, 1, 4, 3])   # 2 up, 2 down, the same magnitudes
    assert abs(rank_biserial(x, y)) < 1e-9


def test_rank_biserial_is_robust_against_outliers():
    # one extreme outlier hardly changes the ranks -> r stays close to 1
    x = np.array([2, 3, 4, 1000.0]); y = np.array([1, 1, 1, 1.0])
    assert rank_biserial(x, y) == 1.0


# ------------------------------ the corrections ------------------------------
def test_bonferroni_threshold_and_flags():
    threshold, sig = bonferroni([0.01, 0.03, 0.20, 0.04], alpha=0.05)
    assert np.isclose(threshold, 0.0125)                 # 0.05 / 4
    assert sig == [True, False, False, False]            # only 0.01 < 0.0125


def test_holm_is_more_powerful_than_bonferroni():
    pvals = [0.001, 0.013, 0.5]
    _, bonf = bonferroni(pvals, 0.05)                    # threshold 0.0167
    holm = holm_bonferroni(pvals, 0.05)
    # 0.013: Bonferroni 0.013<0.0167 sig; Holm k=1 -> 0.05/2=0.025, 0.013<0.025 sig. Both sig.
    assert bonf[0] and bonf[1] and not bonf[2]
    assert holm[0] and holm[1] and not holm[2]


def test_holm_stops_at_the_first_non_significant_one():
    # Holm works on SORTED p-values: both 0.001 become significant BEFORE the 0.5
    # even comes up and triggers the stop.
    holm = holm_bonferroni([0.001, 0.5, 0.001], 0.05)
    assert holm[0] is True and holm[1] is False and holm[2] is True
    # a real stop: the second smallest is already too large -> everything after it is n.s.
    holm2 = holm_bonferroni([0.001, 0.30, 0.40], 0.05)
    assert holm2[0] is True and holm2[1] is False and holm2[2] is False


# ------------------------------ the analysis finds the truth ------------------------------
def test_paired_comparison_agrees_with_scipy():
    a, b = _pairs(DF, "presence")
    r = paired_comparison(a, b)
    assert np.isclose(r["wilcoxon_p"], stats.wilcoxon(a, b).pvalue)
    assert np.isclose(r["t_p"], stats.ttest_rel(a, b).pvalue)


def test_analysis_finds_the_built_in_effects():
    # 6 DoF: more presence, less sickness, less time - all significant (Wilcoxon)
    a, b = _pairs(DF, "presence"); assert b.mean() > a.mean() and stats.wilcoxon(a, b).pvalue < 0.01
    a, b = _pairs(DF, "sickness"); assert b.mean() < a.mean() and stats.wilcoxon(a, b).pvalue < 0.01
    a, b = _pairs(DF, "time");     assert b.mean() < a.mean() and stats.wilcoxon(a, b).pvalue < 0.01


def test_comfort_is_the_shaky_finding():
    # a small effect: significant raw, but not after Bonferroni
    a, b = _pairs(DF, "comfort")
    p = stats.wilcoxon(a, b).pvalue
    assert 0.0125 < p < 0.05          # between the raw alpha and the Bonferroni threshold
    assert abs(cohen_dz(a, b)) < 0.7  # clearly smaller than the large effects


def test_order_effect_carryover_sickness():
    oe = order_effect(DF, "sickness")
    assert oe["mean_second"] > oe["mean_first"]          # the second session is sicker (carryover)


def test_without_counterbalancing_the_effect_is_masked():
    # THE core finding: the naive analysis clearly underestimates the true sickness effect (12)
    dfn = generate_naive_without_counterbalancing(seed=3)
    a, b = _pairs(dfn, "sickness")
    measured = a.mean() - b.mean()
    true_value = -TRUTH["sickness_effect"]               # 12
    assert measured < 0.7 * true_value                   # clearly underestimated
    assert measured > 0                                  # the direction is still right


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests ...")
    for name, tf in tests:
        tf(); print(f"  {name} ... OK")
    print("All tests passed.")
