# 24 — Causal Inference: Did the Treatment Actually Work?

Difficulty: 🟠 Medium-Hard | Topic: Causal Inference

## 🎯 Project Goal
Estimate the causal effect of a job-training program on real earnings using **observational** data, where treated and control groups are not comparable to begin with. Recover a credible average treatment effect and state honestly how much to trust it.

## 📊 Dataset + Evaluation Metric
- **Dataset:** LaLonde / NSW job-training data. Use the **experimental** sample (randomized) and the **observational** sample (NSW treated vs PSID/CPS controls) — both are widely mirrored as `nsw.dta` / `lalonde.csv` (e.g. the `dowhy`, `causaldata`, or R `MatchIt` mirrors). Outcome: `re78` (1978 real earnings). Treatment: `treat`. Covariates: age, education, race, marital status, `re74`, `re75`.
- **Evaluation metric:** there is no single accuracy score. The benchmark is the **experimental ATE (~$1,800 increase in earnings)**. Your observational estimate is "good" to the extent it recovers a similar effect once confounding is addressed — and you are graded on the rigor of the argument, not the closeness alone.

## 🏁 Success Criteria
- The naive difference in means computed first, and an explanation of why it is badly biased here (show the covariate imbalance)
- A stated identification strategy with its assumptions made explicit (unconfoundedness / overlap) and overlap checked, not assumed
- At least two estimators that adjust for confounding, compared against each other and against the experimental benchmark
- A sensitivity / robustness check that asks how strong an unmeasured confounder would have to be to overturn the conclusion
- A final ATE with an uncertainty interval and a one-paragraph verdict on how much you'd trust it for a real funding decision

Relevant techniques (look them up yourself): confounding and back-door adjustment, propensity scores, matching, inverse-probability weighting (IPW), doubly-robust / AIPW estimators, common-support / overlap diagnostics, sensitivity analysis for unmeasured confounding.
