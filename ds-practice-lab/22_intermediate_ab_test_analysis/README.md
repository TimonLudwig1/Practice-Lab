# 22 — Experimentation: A/B Test Analysis 🧪

Difficulty: 🟡 Intermediate | Topic: Experimentation & Inference

## 🎯 Project Goal
Analyze a real mobile-game A/B test end to end and deliver a **ship / don't-ship recommendation** that you can defend. The skill here is turning a noisy experiment into an honest decision: effect size, uncertainty, and the difference between "statistically significant" and "actually matters".

## 📊 Dataset Description
**Cookie Cats A/B test** — ~90k players randomly assigned to `gate_30` (control) vs `gate_40` (treatment): the level at which a progression gate appears. Columns: `userid`, `version`, `sum_gamerounds`, `retention_1` (day-1 return), `retention_7` (day-7 return).

Download from Kaggle: *"Mobile Games A/B Testing - Cookie Cats"* → save `cookie_cats.csv` into `data/raw/`. (Small file; the notebook assumes that path.)

## 💡 Suggested Approach (high-level)
1. **Sanity-check the randomization first** before looking at any outcome: are the two groups balanced in size? Any duplicate users? This is the step people skip and regret.
2. Pick your primary metric deliberately (`retention_7` is the business-relevant one) and treat `retention_1` and `sum_gamerounds` as secondary. Decide this *before* peeking.
3. For the binary retention metrics: estimate each group's rate, the **difference**, and a confidence interval for the difference. Run a two-proportion test (look up `statsmodels.stats.proportion.proportions_ztest`). Report the effect size, not just the p-value.
4. `sum_gamerounds` is heavily skewed with extreme outliers — a plain t-test on the mean is fragile. Look at the distribution, consider a bootstrap CI or a rank-based test, and explain your choice.
5. Quantify uncertainty visually: a bootstrap distribution of the retention-7 difference makes the result far more honest than a single p-value.
6. Watch for the **multiple-comparisons** trap: you're testing several metrics. Note how that affects your confidence.
7. Write the verdict: ship or not, the estimated effect with its interval, and what you'd want before committing (sample size / power, longer horizon, guardrail metrics).

## 🏁 Success Criteria
- Randomization / balance check done and reported before any outcome analysis
- Primary metric chosen and justified up front
- Difference in retention-7 with a confidence interval **and** a significance test — effect size emphasized over the bare p-value
- Skewed `sum_gamerounds` handled with a method you justify (not a naive mean t-test)
- One bootstrap-based figure of the treatment effect's uncertainty
- A clear, defensible ship/don't-ship recommendation in plain language

## 🔗 Useful References
- [statsmodels proportions_ztest](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html)
- Look up: *two-proportion z-test*, *bootstrap confidence interval*, *statistical power*, *multiple comparisons / family-wise error*, *practical vs statistical significance*
- This pairs naturally with project 02 (`hypothesis_testing`) — same tools, real decision.
