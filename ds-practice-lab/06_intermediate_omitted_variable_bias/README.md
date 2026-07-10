# 06 — Omitted Variable Bias: When Regression Lies 📉 `[from your lectures]`

Difficulty: 🟡 Intermediate | Topic: Statistics — Regression & Causal Pitfalls

## 🎯 Project Goal
Demonstrate omitted variable bias (OVB) empirically: build a data-generating process where you control the true causal effects, then show how leaving out a confounder systematically distorts regression coefficients — and verify the textbook OVB formula numerically.

## 📊 Dataset Description
Two parts:
1. **Synthetic wage data you generate yourself** — e.g., `ability` (unobserved), `education` (depends on ability), `wage` (depends on both). You choose the true coefficients, so you know exactly what an unbiased regression *should* recover.
2. **A real dataset for the transfer task** — e.g., the `mtcars`-style question "does horsepower lower fuel efficiency once weight is controlled for?", or any dataset with a plausible confounder. Suggested easy option: seaborn's `mpg` dataset (`sns.load_dataset("mpg")`).

The notebook contains a starting snippet for the synthetic data-generating process.

## 💡 Suggested Approach (high-level)
1. Write the data-generating process with explicit true parameters (e.g., wage = β₀ + β₁·education + β₂·ability + noise, with education itself correlated with ability).
2. Fit the **short regression** (wage ~ education) and the **long regression** (wage ~ education + ability) with `statsmodels` OLS. Compare both education coefficients with your true β₁.
3. Verify the OVB formula: bias = β₂ · δ, where δ is the slope from regressing the omitted variable on the included one. Compute it explicitly and confirm it matches the gap you observed.
4. Explore the geometry of the bias: rerun the experiment over a grid of (correlation between education and ability) × (strength of ability's effect). Heatmap the resulting bias. When is the bias zero? When does the sign flip?
5. Transfer to real data: pick a question in the `mpg` dataset where adding a control variable meaningfully changes a coefficient. Tell the story: what is the confounder, which direction did it push the naive estimate, and why?
6. Write the "lessons" cell: what does this mean for every regression coefficient you will ever interpret?

## 🏁 Success Criteria
- Short vs long regression compared against known truth, with statsmodels summary output read and interpreted (coefficient, SE, CI)
- OVB formula verified numerically (analytic bias ≈ observed bias)
- Bias heatmap over the (correlation × effect strength) grid, with the two zero-bias conditions identified
- Real-data example where a coefficient changes sign or magnitude after adding a control, with a written causal interpretation

## 🔗 Useful References
- `statsmodels.formula.api.ols` — R-style formulas (`"wage ~ education"`)
- Look up: *omitted variable bias formula*, *confounder*, *backdoor path*
- Cunningham, *Causal Inference: The Mixtape* — OVB chapter (free online)
- Project [[05_intermediate_sampling_bias]] is a natural companion — biased samples and omitted variables are the two classic ways to fool yourself with data
