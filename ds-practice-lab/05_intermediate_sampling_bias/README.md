# 05 — Sample Bias Laboratory 🎯 `[from your lectures]`

Difficulty: 🟡 Intermediate | Topic: Statistics — Sampling & Selection Bias

## 🎯 Project Goal
Build a simulation that demonstrates, quantifies, and partially corrects sample bias: you create a synthetic "ground truth" population, draw samples with different (flawed) sampling schemes, and measure how badly each one distorts your estimates.

## 📊 Dataset Description
You generate it yourself — that's the point: only with a synthetic population do you *know* the true values your samples should recover. Create a population of ~200,000 people with correlated attributes, e.g. age, income, smartphone ownership, daily screen time, and political/consumer preference. Build in realistic dependencies (income depends on age; smartphone ownership lower for older people; screen time depends on age and smartphone ownership).

The notebook contains a starting snippet for the population generator.

## 💡 Suggested Approach (high-level)
1. Generate the population and record the true population means/proportions — your gold standard.
2. Implement a **simple random sample** as the unbiased reference. Show sampling error shrinking as n grows (this is your sanity check).
3. Implement at least three biased sampling schemes and compare each against the truth:
   - **Convenience sample** (e.g., online survey → only smartphone owners can respond)
   - **Voluntary response / self-selection** (response probability depends on the quantity being measured, e.g. people with extreme opinions respond more)
   - **Survivorship-style filter** (e.g., only sampling people still subscribed/alive in the system)
4. For each scheme: repeat the draw many times, plot the sampling distribution of the estimate vs the true value. Bias = systematic offset, not random scatter — your plots should show exactly that distinction.
5. Attempt a correction: **post-stratification or inverse-probability weighting** on the convenience sample using known population age structure. How much of the bias can you remove? What stays unfixable, and why?
6. Bonus: recreate a mini "Literary Digest 1936" — show that a biased sample of 50,000 loses against an unbiased sample of 500.

## 🏁 Success Criteria
- A figure per sampling scheme showing the sampling distribution against the true value
- A summary table: scheme → bias, standard error, RMSE of the estimate
- Weighting correction implemented and its effect quantified
- A short written conclusion: which kinds of bias can sample size fix? (Hint: none of them — be able to argue this.)

## 🔗 Useful References
- `numpy.random.default_rng` — modern NumPy random API
- Look up: *post-stratification*, *inverse probability weighting*, *Horvitz-Thompson estimator*
- The Literary Digest 1936 poll disaster (any article) — your bonus task in real life
- `pandas.cut` for building age strata
