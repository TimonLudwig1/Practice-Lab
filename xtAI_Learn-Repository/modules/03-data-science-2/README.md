# Module 03 — Data Science 2

> **Language note.** This document is bilingual. The English version comes first; the German version (*deutsche Fassung*) follows below the horizontal rule.

**What is this about?** Data Science 1 taught you to understand and describe data. Data Science 2 turns that into **models and reliable statements**: quantifying relationships (regression), analysing temporal structure (time series), constructing features (feature engineering), taming high-dimensional data (PCA) and fetching data from where they actually live (SQL). The module is the bridge between descriptive analysis and the machine learning of modules 04/05.

**Prerequisites:** module 02 (pandas, EDA, descriptive and inferential statistics) is assumed throughout. School mathematics plus a willingness to look at a few matrices.

**To do beforehand:** module 01, module 02.

---

## Learning objectives

After this module you will be able to:

- set up a **linear regression**, interpret its coefficients and goodness-of-fit measures ($R^2$, residuals) and check its assumptions,
- explain why coefficients in multiple regression may only be interpreted "ceteris paribus" and what **multicollinearity** does,
- use a **logistic regression** for yes/no questions and interpret odds ratios,
- decompose **time series** into trend, season and remainder, read autocorrelation and produce simple forecasts (including clean time series validation),
- systematically **construct features** (transformations, category encoding, date features) and avoid data leakage,
- reduce dimensions with **PCA** and explain what principal components are (and what they are not),
- write **basic SQL queries** (SELECT, JOIN, GROUP BY) and combine them with pandas,
- set up an **A/B test design**: sample size, run time, evaluation, typical mistakes.

---

## 1. Basics

### 1.1 Linear regression — the workhorse of data analysis

**Idea:** we model a target quantity $y$ as a linear function of explanatory variables:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \varepsilon$$

$\beta_0$ = intercept, $\beta_i$ = slopes, $\varepsilon$ = remainder (what the model does not explain). "Fitting" means: choose the $\beta$ so that the **sum of squared residuals** is minimal (**ordinary least squares, OLS**):

$$\hat{\beta} = \arg\min_\beta \sum_i (y_i - \hat{y}_i)^2$$

**Why squares?** They penalise large errors disproportionately, make the problem smoothly solvable (set the derivative to zero, giving a closed formula) and correspond to the maximum likelihood solution under normally distributed errors.

**Worked mini example** (simple regression, $y = \beta_0 + \beta_1 x$): for the slope,

$$\hat{\beta}_1 = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2} = r \cdot \frac{s_y}{s_x}, \qquad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

The slope is therefore the **correlation, converted into the units of the data**. Example: an ice cream parlour, $x$ = temperature (degrees C), $y$ = revenue (euros). With $r = 0.8$, $s_y = 300$ euros, $s_x = 5$ degrees: $\hat{\beta}_1 = 0.8 \cdot 300/5 = 48$ euros per degree — one degree more means on average 48 euros more revenue.

**Goodness-of-fit measure $R^2$:** the proportion of the variance of $y$ that the model explains:

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} \in (-\infty, 1]$$

$R^2 = 0.64$ means: 64 % of the spread explained (in simple regression $R^2 = r^2$). Careful: $R^2$ rises *automatically* with every additional variable — hence the **adjusted $R^2$**, and hence model quality is ultimately to be measured on *new* data (see module 04).

**Residual analysis — the underestimated step:** the assumptions of OLS (linearity, constant error spread / "homoscedasticity", independent errors) are checked on the **residual plot** (residuals against predictions):

- pattern/curvature implies the relationship is not linear (transformation or a different model),
- a funnel shape implies the variance grows with the level (often $\log y$ helps),
- outliers with high **leverage** (extreme $x$ values) can tip the whole line.

> **Note:** a high $R^2$ with a broken residual plot is worthless; a moderate $R^2$ with a clean residual plot can be very useful.

### 1.2 Multiple regression: "ceteris paribus" and its pitfalls

With several explanatory variables, $\beta_1$ means: *change in $y$ per unit of $x_1$ when all other variables are held constant.* That is the great strength — and the great trap:

- **Confounder adjustment**: the relationship "ice cream sales → drowning accidents" disappears as soon as "temperature" is in the model. Regression can *partial out* confounders — but only those you know about and have measured!
- **Multicollinearity**: if two predictors are strongly correlated (e.g. living space and number of rooms), the model cannot separate their influence: coefficients become unstable, signs flip seemingly arbitrarily, standard errors explode. Diagnosis: pairwise correlations, variance inflation factor (VIF). Incidentally, *prediction* hardly suffers from this — only *interpretation*.
- **Omitted variable bias**: if a relevant variable is missing that correlates with one that is included, the included one "inherits" its effect. This is why regression coefficients from observational data are *not* causal statements (module 02 script, section 2.4 — it applies here even more strictly).

**Categorical variables** enter the model as **dummy variables** (city = Berlin/Hamburg/Cologne → two 0/1 columns, one category is left out as the reference — otherwise there is perfect collinearity, the "dummy trap").

### 1.3 Logistic regression: when y is a yes/no

For binary targets (buys / does not buy) linear regression does not fit (predictions below 0 or above 1). The **logistic regression** models the *probability* instead, via the sigmoid function:

$$P(y = 1 \mid x) = \sigma(\beta_0 + \beta_1 x_1 + \dots) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots)}}$$

Interpretation via **odds**: $\text{odds} = P/(1-P)$. It holds that $\log(\text{odds}) = \beta_0 + \beta_1 x_1 + \dots$ — the coefficients are additive on the log-odds scale, and $e^{\beta_1}$ is the **odds ratio**: the factor by which the odds change per unit of $x_1$. $e^{\beta_1} = 1.5$ means "50 % higher odds per unit", not "50 % higher probability"! Fitting is done by maximum likelihood (no closed formula, numerical optimisation — conceptually the same "downhill" idea as in module 01, section 3.1).

Logistic regression is at the same time your first "real" classifier besides naive Bayes — and the standard baseline in practice. A deeper treatment (decision boundaries, regularisation, metrics) follows in Machine Learning 1.

---

## 2. Intermediate

### 2.1 Time series analysis

Time series (sales per day, temperature per hour, ...) violate the basic assumption of independent observations: **neighbouring values are related**. This needs its own tools.

**The classical decomposition:**

$$y_t = T_t + S_t + R_t \quad \text{(additive)} \qquad y_t = T_t \cdot S_t \cdot R_t \quad \text{(multiplicative)}$$

- **Trend $T_t$**: the long-term direction (e.g. via a moving average / `rolling`)
- **Season $S_t$**: a recurring pattern of fixed period (day of the week, month, time of day)
- **Remainder $R_t$**: what is left over

You choose multiplicative when the seasonal swings grow with the level (typical under growth: the "December peak" of a shop grows along).

**Autocorrelation (ACF):** the correlation of the series with itself shifted by $k$ steps (lag $k$). The ACF is the ECG of a time series: peaks at lag 7 (daily data) reveal a weekly rhythm, a slowly decaying ACF reveals a trend.

**Simple forecasting methods** (always as a baseline first!):

| Method | Forecast | when |
|--|--|--|
| Naive | the last value | astonishingly hard to beat |
| Seasonal naive | the value from one period ago ("like last Monday") | with strong seasonality |
| Moving average | mean of the last $k$ values | smooth series |
| Exponential smoothing | weighted mean, more recent values count more: $\hat{y}_{t+1} = \alpha y_t + (1-\alpha)\hat{y}_t$ | the standard all-rounder |

**The golden rule of time series validation:** never split into train/test at random! Train on the past, test on the future (a **temporal split**), otherwise the model already "sees" the future during training — the forecast quality is massively overestimated. (This is the most important special case of *leakage*, see 2.2.)

### 2.2 Feature engineering — casting knowledge into columns

Models are only as good as their inputs. Feature engineering = building from raw data the attributes that make the relationship visible:

- **Transformations**: $\log$ for right-skewed quantities (prices, incomes) — this makes multiplicative effects additive and tames outliers. Polynomials/interactions ($x_1 \cdot x_2$) when effects act together.
- **Date features**: a timestamp becomes day of the week, month, hour, a holiday flag, "days since the last event", ... (in the bike sharing project of module 02, half the explanatory power sat in `hr` and `workingday`!). For cyclical quantities (hour 23 is close to hour 0!) one uses a sine/cosine encoding: $\sin(2\pi h/24), \cos(2\pi h/24)$.
- **Category encoding**: one-hot/dummies for nominal attributes; ordinal encoding only where a genuine order exists.
- **Lag and window features** (time series): $y_{t-1}$, $y_{t-7}$, a rolling 7-day mean — this turns a time series into a regression table.
- **Scaling**: standardisation ($z = (x - \bar{x})/s$) — mandatory for regression with regularisation, for PCA and for distance-based methods.

**Leakage — the cardinal error:** a feature contains information that would not exist at prediction time. Classics: the column "cancelled on" when predicting cancellations; scaling parameters computed jointly on train and test; a random split for time series. The symptom: suspiciously good test results that burst in reality. Rule: **everything that is learned (including means for scaling!) is learned only on the training data.**

### 2.3 Principal component analysis (PCA)

With many correlated variables (50 sensors, 1000 questionnaire items) you want to reduce the dimension without losing much information.

**Idea:** find new axes (**principal components**) as linear combinations of the original variables such that the first axis captures the maximum variance of the data, the second the maximum remaining variance perpendicular to it, and so on. Mathematically: eigenvectors of the covariance matrix; the associated eigenvalues are the captured variances.

- **Explained variance**: the scree plot (share of variance per component) shows how many components you need — often 80–90 % of the variance sits in a few components when the variables are strongly correlated.
- **Loadings**: the weights of the original variables in a component — this is how you interpret what a component "stands for" (e.g. PC1 in the penguin data set is roughly "overall body size").
- **Mandatory: standardise beforehand** — otherwise the variable with the largest numerical values (grams beats millimetres) dominates the analysis for purely numerical reasons.

**What PCA is not:** it is not feature *selection* (components mix all variables), it is no guarantee that much variance means much *relevance* for a target quantity (PCA knows no $y$ — it is unsupervised), and with non-linear structures (curved manifolds) it falls short (see t-SNE/UMAP in later modules).

### 2.4 SQL — fetching data where they live

In companies, data live in relational databases, and analysis begins with **SQL**. You already know the mapping to pandas:

| SQL | pandas |
|--|--|
| `SELECT col1, col2 FROM t` | `df[["col1", "col2"]]` |
| `WHERE price > 100` | `df[df.price > 100]` |
| `GROUP BY city` + aggregate | `df.groupby("city").agg(...)` |
| `JOIN ... ON id` | `df.merge(..., on="id")` |
| `ORDER BY x DESC LIMIT 10` | `df.sort_values("x", ascending=False).head(10)` |

The skeleton of every query (and its logical order of evaluation: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY):

```sql
SELECT   c.city, COUNT(*) AS n, AVG(o.price) AS mean_price
FROM     orders o
JOIN     customers c ON c.customer_id = o.customer_id
WHERE    o.order_date >= '2024-01-01'
GROUP BY c.city
HAVING   COUNT(*) >= 10
ORDER BY mean_price DESC;
```

- **Kinds of JOIN**: `INNER` (only matches), `LEFT` (all of the left side, NULL on the right where missing) — the two you need daily.
- `WHERE` filters *rows before* the grouping, `HAVING` filters *groups afterwards* — a popular beginner's mistake.
- **SQLite** is a fully-fledged SQL database in a single file, built into Python (`sqlite3`), and `pd.read_sql` fetches query results directly as a DataFrame — perfect for learning and for small projects.

Rule of thumb for the division of labour: **coarse filtering and joins in SQL** (the database is built for it and keeps the data volume small), **analysis and plots in pandas**.

---

## 3. Advanced topics

### 3.1 Setting up A/B tests properly

The A/B test is the experiment of the digital economy: split users at random into control (A) and variant (B), compare a metric. **Randomisation** makes it a causal instrument — on average the groups differ *only* by the variant (no confounding). You know the statistics from module 02 (section 3.1); here is the design side:

1. **Fix in advance**: the metric, the minimum detectable effect (MDE), the significance level $\alpha$, the power (usually 80 %).
2. **Compute the sample size** (power analysis): small effects need *very* many users — roughly, $n$ grows with $1/\text{effect}^2$. Demonstrating an increase in conversion from 2 % to 2.2 % needs tens of thousands of users per group.
3. **See the run time through**: full weeks (day-of-week effects!), and **no early stopping** at the first significant interim result — constant interim testing massively inflates the false positive rate (the "peeking problem").
4. **Evaluate**: report the effect size + confidence interval; check the randomisation (A/A test, sample ratio mismatch); treat segments only as a hypothesis generator (multiple testing!).

Typical pitfalls in practice: the novelty effect (new things are over-clicked at first), interference between groups (network effects), and the temptation to search through 20 metrics until one is significant.

### 3.2 Simulation and bootstrap — statistics without a formula collection

Modern data analysis replaces many analytical formulas with **computing power**:

- **Bootstrap**: draw from your sample (n values) many new samples *with replacement* (n values each), compute the statistic every time — the spread of these bootstrap statistics estimates the standard error, their quantiles give a confidence interval. This works for the median, quantiles, ratios ... where classical formulas are missing or ugly.
- **Permutation test**: under $H_0$ "no group difference" the group labels are exchangeable. Shuffle the labels a thousand times, compute the test statistic each time — the share of shuffled results that exceed the real one *is* the p-value. No distributional model needed, and the logic of the p-value becomes tangible.

You build both procedures yourself in the medium project — they are the best school of intuition for inferential statistics there is.

### 3.3 From the notebook to the pipeline: reproducibility at a larger scale

- **Scripts instead of cell chaos**: as soon as an analysis is settled, it moves into functions/modules with clear inputs and outputs; the notebook remains as the report.
- **Data versioning light**: raw data immutable ("read-only"), every transformation as code (never edit the CSV by hand!), intermediate states named and dated.
- **Big data outlook**: when data no longer fit into RAM — columnar formats (Parquet), chunked processing, DuckDB/Polars as fast local engines, Spark and friends on a cluster. The *concepts* (filter early, join sparingly, aggregate close to the data) are the same as with SQL.

---

## 4. Summary / cheat sheet

**Linear regression**
- OLS minimises $\sum (y - \hat{y})^2$; simple regression: $\hat\beta_1 = r \cdot s_y / s_x$
- $R^2$ = explained variance; rises automatically with more variables (see adjusted / test data)
- Residual plot: curvature = non-linear, funnel = heteroscedasticity, watch out for leverage points
- Multiple regression: a coefficient is the effect *ceteris paribus*; multicollinearity makes coefficients unstable (VIF); the dummy trap: leave out one reference category

**Logistic regression**
- $P(y=1) = \sigma(\beta^T x)$; $e^\beta$ = odds ratio (odds, not probabilities!)

**Time series**
- Decomposition $y = T + S + R$ (or multiplicative); read the ACF (a lag-7 peak means a weekly rhythm)
- Baselines: naive, seasonal naive, moving average, exponential smoothing
- **Always split temporally** — never at random!

**Feature engineering**
- log for skew, sin/cos for cycles, one-hot for nominal, lags for time series, standardise for PCA and friends
- **Leakage**: use nothing that would not exist at prediction time; learn everything learnable on train only

**PCA**
- Principal components = orthogonal directions of maximum variance (eigenvectors of the covariance matrix)
- Scree plot for the number, loadings for the interpretation, standardise beforehand, unsupervised!

**SQL**
- `SELECT ... FROM ... [JOIN ... ON ...] WHERE ... GROUP BY ... HAVING ... ORDER BY ...`
- WHERE before grouping, HAVING after; LEFT JOIN keeps all left-hand rows
- SQLite + `pd.read_sql` = a learning environment without a server

**A/B and resampling**
- In advance: metric, MDE, $\alpha$, power, run time; no peeking; report effect + CI
- Bootstrap: draw with replacement, giving a CI for (almost) any statistic
- Permutation test: shuffle labels, giving a p-value without a distributional assumption

---

## 5. Self-test

<details><summary><b>1. Your model has R² = 0.92, but the residual plot shows a clear U shape. What does that mean, and what do you do?</b></summary>

The U shape means: the relationship is **not linear** — the model systematically over- or underestimates depending on the range, despite the high $R^2$. Predictions outside the middle range and all coefficient interpretations are unreliable. Remedy: a transformation ($\log$, a quadratic term) or a non-linear model — and afterwards check the residual plot again.
</details>

<details><summary><b>2. In the house price model, "number of rooms" suddenly has a negative coefficient as soon as "living space" is in the model. A blunder?</b></summary>

Not necessarily — two readings. (1) **Ceteris paribus** this is plausible: *at the same living space*, more rooms means smaller rooms, which can push the price down. (2) Number of rooms and living space are strongly correlated (**multicollinearity**), in which case the individual coefficients are unstable and their signs are not very reliable (check the VIF). In both cases: the coefficient does not measure "the effect of rooms" as such, but the additional effect with the other variables held fixed.
</details>

<details><summary><b>3. Logistic regression: β for "subscribed to the newsletter" is 0.69, so e^β is about 2. Formulate the correct interpretation — and a wrong one that is often heard.</b></summary>

Correct: newsletter subscribers have (ceteris paribus) **twice the odds** of buying. Wrong: "twice the probability". At small probabilities, odds and probability are close together (2 % → about 4 %), at large ones they are not: from 50 % ($\text{odds}=1$) a doubling of the odds leads to 66.7 %, not to 100 %.
</details>

<details><summary><b>4. Why may time series data not be split into training and test at random?</b></summary>

With a random split, training points lie temporally *after* test points — the model learns from the future of the test cases (in autocorrelated series, $y_{t+1}$ contains a lot of information about $y_t$). The measured quality is then systematically too optimistic and collapses in real use, where the future really is unknown. Correct: a temporal split (past → future), possibly rolling validation.
</details>

<details><summary><b>5. Name three examples of leakage and the common principle behind them.</b></summary>

(1) The column "cancellation date" when predicting cancellations; (2) standardisation with mean/sd computed jointly on train and test; (3) a random split for time series (or duplicates that end up in both train and test). The principle: **information that would not be available at prediction time flows into training** — the test performance then measures not generalisation but the leak.
</details>

<details><summary><b>6. PCA on raw data with the columns "income (EUR)" and "age (years)" — what goes wrong?</b></summary>

Without standardisation, income (with variance in the tens of thousands) dominates the covariance matrix purely because of its unit — PC1 shows practically only "income" and age disappears. PCA maximises variance in the units present; the variables only become comparable after standardisation ($z$ values). Therefore: scale first, then PCA.
</details>

<details><summary><b>7. What is the difference between WHERE and HAVING — and why does `WHERE COUNT(*) > 10` give an error?</b></summary>

`WHERE` filters **individual rows before** the grouping — at that moment there are no groups yet, and therefore no `COUNT(*)`. `HAVING` filters **groups after** the aggregation and may therefore use aggregate functions. The logical order: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY.
</details>

<details><summary><b>8. Your A/B test has been running for 3 days, the dashboard shows p = 0.04. The product manager wants to roll out immediately. Two objections?</b></summary>

(1) **Peeking**: whoever looks at significance daily and stops at the first p below 0.05 has a real false positive rate far above 5 % — the test has to reach the run time/sample planned in advance. (2) **Three days are not a full week**: day-of-week effects (and the novelty effect) can drive the result. Besides, there is no look at effect size + confidence interval — "significant" alone does not justify a rollout.
</details>

<details><summary><b>9. Explain in two sentences how a bootstrap confidence interval for the median comes about.</b></summary>

You draw from the sample many (e.g. 10,000) new samples of the same size **with replacement** and compute the median each time. The interval between the 2.5 % and the 97.5 % quantile of these bootstrap medians is a 95 % confidence interval — entirely without a distributional assumption or a formula for the standard error of the median.
</details>

<details><summary><b>10. A colleague runs PCA and keeps the components that explain 95 % of the variance, in order to predict a target quantity with them. Which fallacy looms?</b></summary>

PCA is **unsupervised** — it does not know the target quantity. Much variance does not mean much predictive power: the relevant information can sit in a low-variance component (and gets thrown away), while high-variance components can bundle pure noise irrelevant to $y$. For predictions, base the choice of components on predictive performance (validation), not on the share of variance.
</details>

---

## 6. Literature and sources

**Textbooks**

- **James, Witten, Hastie, Tibshirani — "An Introduction to Statistical Learning" (ISLR), 2nd ed.** — chapters 3 (linear regression) and 4.3 (logistic regression) are the best treatment of the material; PCA in chapter 12. **Free**: https://www.statlearning.com *(beginner-friendly to advanced, the reference book for modules 04/05 as well)*
- **Hyndman & Athanasopoulos — "Forecasting: Principles and Practice", 3rd ed.** — *the* time series book, completely **free**: https://otexts.com/fpp3/ *(beginner-friendly; examples in R, concepts language-independent)*
- **Kohavi, Tang & Xu — "Trustworthy Online Controlled Experiments"** — the practical standard on A/B tests, from the people who industrialised them at Microsoft/Amazon. *(advanced)*

**Online courses and interactive material (free)**

- **SQLBolt** (https://sqlbolt.com) — interactive SQL learning in the browser, perfect before the basic project *(beginner-friendly)*
- **Mode SQL Tutorial** (https://mode.com/sql-tutorial/) — SQL from an analyst's perspective *(beginner-friendly)*
- **Seeing Theory**, the chapters "Regression Analysis" and "Frequentist Inference" (https://seeing-theory.brown.edu) *(beginner-friendly)*
- **StatQuest** (YouTube, Josh Starmer): the videos on PCA, linear and logistic regression are gold for intuition *(beginner-friendly)*

**Blog posts / going deeper (free)**

- *explained.ai — "How to interpret PCA plots"* and the Distill article *"How to Use t-SNE Effectively"* (distill.pub) — the latter as an outlook on why dimensionality reduction plots have to be read with care
- Evan Miller: *"How Not To Run an A/B Test"* — the classic text on the peeking problem
- Tim Hesterberg: *"What Teachers Should Know About the Bootstrap"* (arXiv) *(advanced)*

---

**Next step:** `projects/01-basic/` (SQL on a real database) → `projects/02-medium/` (build bootstrap and permutation test yourself) → `projects/03-final/` (regression and time series forecasting on the bike sharing data).

---
---

# Modul 03 — Data Science 2 (deutsche Fassung)

**Worum geht es?** Data Science 1 hat dir beigebracht, Daten zu verstehen und zu beschreiben. Data Science 2 macht daraus **Modelle und belastbare Aussagen**: Zusammenhänge quantifizieren (Regression), zeitliche Strukturen analysieren (Zeitreihen), Merkmale konstruieren (Feature Engineering), hochdimensionale Daten bändigen (PCA) und Daten dort holen, wo sie wirklich liegen (SQL). Das Modul ist die Brücke zwischen deskriptiver Analyse und dem maschinellen Lernen der Module 04/05.

**Vorkenntnisse:** Modul 02 (pandas, EDA, deskriptive & schließende Statistik) wird durchgehend vorausgesetzt. Schulmathematik plus die Bereitschaft, ein paar Matrizen anzuschauen.

**Vorher zu machen:** Modul 01, Modul 02.

---

## Lernziele

Nach diesem Modul kannst du:

- eine **lineare Regression** aufstellen, ihre Koeffizienten und Gütemaße ($R^2$, Residuen) interpretieren und ihre Annahmen prüfen,
- erklären, warum man Koeffizienten in multipler Regression nur „ceteris paribus" interpretieren darf und was **Multikollinearität** anrichtet,
- eine **logistische Regression** für Ja/Nein-Fragen einsetzen und Odds Ratios interpretieren,
- **Zeitreihen** in Trend, Saison und Rest zerlegen, Autokorrelation lesen und einfache Prognosen erstellen (inkl. sauberer Zeitreihen-Validierung),
- systematisch **Features konstruieren** (Transformationen, Kategorien-Encoding, Datums-Features) und Datenlecks (Leakage) vermeiden,
- mit **PCA** Dimensionen reduzieren und erklären, was Hauptkomponenten sind (und was nicht),
- **SQL-Grundabfragen** schreiben (SELECT, JOIN, GROUP BY) und mit pandas kombinieren,
- ein **A/B-Test-Design** aufsetzen: Stichprobengröße, Laufzeit, Auswertung, typische Fehler.

---

## 1. Grundlagen (Basics)

### 1.1 Lineare Regression — das Arbeitstier der Datenanalyse

**Idee:** Wir modellieren eine Zielgröße $y$ als lineare Funktion von Einflussgrößen:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \varepsilon$$

$\beta_0$ = Achsenabschnitt, $\beta_i$ = Steigungen, $\varepsilon$ = Rest (was das Modell nicht erklärt). „Fitten" heißt: die $\beta$ so wählen, dass die **Summe der quadrierten Residuen** minimal wird (**Ordinary Least Squares, OLS**):

$$\hat{\beta} = \arg\min_\beta \sum_i (y_i - \hat{y}_i)^2$$

**Warum Quadrate?** Sie bestrafen große Fehler überproportional, machen das Problem glatt lösbar (Ableitung nullsetzen ⇒ geschlossene Formel) und entsprechen der Maximum-Likelihood-Lösung bei normalverteilten Fehlern.

**Durchgerechnetes Mini-Beispiel** (einfache Regression, $y = \beta_0 + \beta_1 x$): Für die Steigung gilt

$$\hat{\beta}_1 = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2} = r \cdot \frac{s_y}{s_x}, \qquad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

Die Steigung ist also die **Korrelation, umgerechnet in die Einheiten der Daten**. Beispiel: Eiscafé, $x$ = Temperatur (°C), $y$ = Umsatz (€). Mit $r = 0{,}8$, $s_y = 300$ €, $s_x = 5$ °C: $\hat{\beta}_1 = 0{,}8 \cdot 300/5 = 48$ €/°C — pro Grad mehr im Schnitt 48 € mehr Umsatz.

**Gütemaß $R^2$:** der Anteil der Varianz von $y$, den das Modell erklärt:

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} \in (-\infty, 1]$$

$R^2 = 0{,}64$ heißt: 64 % der Streuung erklärt (bei einfacher Regression ist $R^2 = r^2$). Vorsicht: $R^2$ steigt *automatisch* mit jeder zusätzlichen Variable — deshalb gibt es das **adjustierte $R^2$**, und deshalb ist Modellgüte letztlich auf *neuen* Daten zu messen (→ Modul 04).

**Residuenanalyse — der unterschätzte Schritt:** Die Annahmen von OLS (Linearität, konstante Fehlerstreuung/„Homoskedastizität", unabhängige Fehler) prüft man am **Residuenplot** (Residuen gegen Vorhersagen):

- Muster/Krümmung ⇒ Zusammenhang nicht linear (Transformation oder anderes Modell),
- Trichterform ⇒ Varianz wächst mit dem Niveau (oft hilft $\log y$),
- Ausreißer mit großem **Leverage** (extreme $x$-Werte) können die ganze Gerade kippen.

> **Merke:** Ein hohes $R^2$ mit kaputtem Residuenplot ist wertlos, ein moderates $R^2$ mit sauberem Residuenplot kann sehr nützlich sein.

### 1.2 Multiple Regression: „ceteris paribus" und seine Tücken

Mit mehreren Einflussgrößen bedeutet $\beta_1$: *Änderung von $y$ pro Einheit $x_1$, wenn alle anderen Variablen konstant gehalten werden.* Das ist die große Stärke — und die große Falle:

- **Confounder-Adjustierung**: Der Zusammenhang „Eisverkauf → Ertrinkungsunfälle" verschwindet, sobald „Temperatur" mit im Modell steht. Regression kann Confounder *herausrechnen* — aber nur die, die man kennt und gemessen hat!
- **Multikollinearität**: Sind zwei Prädiktoren stark korreliert (z. B. Wohnfläche und Zimmerzahl), kann das Modell ihren Einfluss nicht trennen: Koeffizienten werden instabil, Vorzeichen kippen scheinbar willkürlich, Standardfehler explodieren. Diagnose: paarweise Korrelationen, Variance Inflation Factor (VIF). Die *Vorhersage* leidet darunter übrigens kaum — nur die *Interpretation*.
- **Omitted Variable Bias**: Fehlt eine relevante Variable, die mit einer enthaltenen korreliert, „erbt" die enthaltene deren Effekt. Deshalb sind Regressionskoeffizienten aus Beobachtungsdaten *keine* Kausalaussagen (Skript Modul 02, Abschnitt 2.4 — gilt hier verschärft).

**Kategoriale Variablen** kommen als **Dummy-Variablen** ins Modell (Stadt = Berlin/Hamburg/Köln → zwei 0/1-Spalten, eine Kategorie bleibt als Referenz weg — sonst perfekte Kollinearität, die „Dummy-Falle").

### 1.3 Logistische Regression: wenn y ein Ja/Nein ist

Für binäre Ziele (kauft / kauft nicht) passt lineare Regression nicht (Vorhersagen < 0 oder > 1). Die **logistische Regression** modelliert stattdessen die *Wahrscheinlichkeit* über die Sigmoid-Funktion:

$$P(y = 1 \mid x) = \sigma(\beta_0 + \beta_1 x_1 + \dots) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots)}}$$

Interpretation über **Odds** (Chancen): $\text{Odds} = P/(1-P)$. Es gilt $\log(\text{Odds}) = \beta_0 + \beta_1 x_1 + \dots$ — die Koeffizienten sind additiv auf der Log-Odds-Skala, und $e^{\beta_1}$ ist das **Odds Ratio**: Faktor, um den sich die Chancen pro Einheit $x_1$ ändern. $e^{\beta_1} = 1{,}5$ heißt „50 % höhere Chancen pro Einheit", nicht „50 % höhere Wahrscheinlichkeit"! Gefittet wird per Maximum Likelihood (keine geschlossene Formel, numerische Optimierung — konzeptionell derselbe „bergab"-Gedanke wie in Modul 01, Abschnitt 3.1).

Die logistische Regression ist zugleich dein erster „richtiger" Klassifikator neben Naive Bayes — und der Standard-Baseline in der Praxis. Vertiefung (Entscheidungsgrenzen, Regularisierung, Metriken) folgt in Machine Learning 1.

---

## 2. Aufbau (Intermediate)

### 2.1 Zeitreihenanalyse

Zeitreihen (Verkäufe pro Tag, Temperatur pro Stunde …) verletzen die Grundannahme unabhängiger Beobachtungen: **benachbarte Werte hängen zusammen**. Das braucht eigene Werkzeuge.

**Die klassische Zerlegung:**

$$y_t = T_t + S_t + R_t \quad \text{(additiv)} \qquad y_t = T_t \cdot S_t \cdot R_t \quad \text{(multiplikativ)}$$

- **Trend $T_t$**: langfristige Richtung (z. B. per gleitendem Mittelwert / `rolling`)
- **Saison $S_t$**: wiederkehrendes Muster fester Periode (Wochentag, Monat, Uhrzeit)
- **Rest $R_t$**: was übrig bleibt

Multiplikativ wählt man, wenn die Saisonausschläge mit dem Niveau wachsen (typisch bei Wachstum: die „Dezemberspitze" eines Shops wächst mit).

**Autokorrelation (ACF):** die Korrelation der Reihe mit sich selbst um $k$ Schritte versetzt ($\text{lag } k$). Die ACF ist das EKG einer Zeitreihe: Spitzen bei Lag 7 (Tagesdaten) verraten Wochenrhythmus, langsam abfallende ACF verrät Trend.

**Einfache Prognoseverfahren** (immer zuerst als Baseline!):

| Verfahren | Prognose | wann |
|--|--|--|
| Naiv | letzter Wert | erstaunlich schwer zu schlagen |
| Saisonal naiv | Wert von vor einer Periode ("wie letzten Montag") | bei starker Saison |
| Gleitender Mittelwert | Mittel der letzten $k$ Werte | glatte Reihen |
| Exponentielle Glättung | gewichtetes Mittel, jüngere Werte zählen mehr: $\hat{y}_{t+1} = \alpha y_t + (1-\alpha)\hat{y}_t$ | Standard-Allrounder |

**Die goldene Regel der Zeitreihen-Validierung:** Niemals zufällig in Train/Test splitten! Trainiere auf der Vergangenheit, teste auf der Zukunft (**zeitlicher Split**), sonst „sieht" das Modell beim Training bereits die Zukunft — die Prognosegüte wird massiv überschätzt. (Das ist der wichtigste Spezialfall von *Leakage*, siehe 2.2.)

### 2.2 Feature Engineering — Wissen in Spalten gießen

Modelle sind nur so gut wie ihre Eingaben. Feature Engineering = aus Rohdaten die Merkmale bauen, die den Zusammenhang sichtbar machen:

- **Transformationen**: $\log$ bei rechtsschiefen Größen (Preise, Einkommen) — macht multiplikative Effekte additiv und zähmt Ausreißer. Polynome/Interaktionen ($x_1 \cdot x_2$), wenn Effekte zusammenwirken.
- **Datums-Features**: aus einem Zeitstempel werden Wochentag, Monat, Stunde, Ferien-Flag, „Tage seit letztem Ereignis" … (im Bike-Sharing-Projekt von Modul 02 steckte die halbe Erklärkraft in `hr` und `workingday`!). Für zyklische Größen (Stunde 23 ist nah an Stunde 0!) nutzt man Sinus/Kosinus-Kodierung: $\sin(2\pi h/24), \cos(2\pi h/24)$.
- **Kategorien-Encoding**: One-Hot/Dummies für nominale Merkmale; Ordinal-Encoding nur, wenn echte Ordnung besteht.
- **Lag- und Fenster-Features** (Zeitreihen): $y_{t-1}$, $y_{t-7}$, gleitender 7-Tage-Mittelwert — so wird aus einer Zeitreihe eine Regressionstabelle.
- **Skalierung**: Standardisieren ($z = (x - \bar{x})/s$) — für Regression mit Regularisierung, PCA und abstandsbasierte Verfahren Pflicht.

**Leakage — der Kardinalfehler:** Ein Feature enthält Information, die es zum Vorhersagezeitpunkt nicht gäbe. Klassiker: die Spalte „Storniert am" bei der Vorhersage von Stornos; Skalierungsparameter, die auf Train+Test gemeinsam berechnet wurden; Zufalls-Split bei Zeitreihen. Symptom: verdächtig gute Testergebnisse, die in der Realität zerplatzen. Regel: **Alles, was gelernt wird (auch Mittelwerte zum Skalieren!), wird nur auf den Trainingsdaten gelernt.**

### 2.3 Hauptkomponentenanalyse (PCA)

Bei vielen korrelierten Variablen (50 Sensoren, 1000 Fragebogen-Items) will man die Dimension reduzieren, ohne viel Information zu verlieren.

**Idee:** Finde neue Achsen (**Hauptkomponenten**) als Linearkombinationen der Originalvariablen, sodass die erste Achse die maximale Varianz der Daten einfängt, die zweite die maximale Restvarianz senkrecht dazu, usw. Mathematisch: Eigenvektoren der Kovarianzmatrix; die zugehörigen Eigenwerte sind die eingefangenen Varianzen.

- **Erklärte Varianz**: Der Scree-Plot (Varianzanteil pro Komponente) zeigt, wie viele Komponenten man braucht — oft stecken 80–90 % der Varianz in wenigen Komponenten, wenn die Variablen stark korreliert sind.
- **Loadings**: Die Gewichte der Originalvariablen in einer Komponente — so interpretiert man, „wofür" eine Komponente steht (z. B. PC1 im Pinguin-Datensatz ≈ „Körpergröße insgesamt").
- **Pflicht: vorher standardisieren** — sonst dominiert die Variable mit den größten Zahlenwerten (Gramm schlägt Millimeter) die Analyse aus rein numerischen Gründen.

**Was PCA nicht ist:** keine Feature-*Auswahl* (Komponenten mischen alle Variablen), keine Garantie, dass viel Varianz = viel *Relevanz* für eine Zielgröße (PCA kennt kein $y$ — sie ist unsupervised), und bei nichtlinearen Strukturen (gebogene Mannigfaltigkeiten) greift sie zu kurz (→ t-SNE/UMAP in späteren Modulen).

### 2.4 SQL — Daten holen, wo sie wohnen

In Unternehmen liegen Daten in relationalen Datenbanken, und die Analyse beginnt mit **SQL**. Das Mapping zu pandas kennst du schon:

| SQL | pandas |
|--|--|
| `SELECT spalte1, spalte2 FROM t` | `df[["spalte1", "spalte2"]]` |
| `WHERE preis > 100` | `df[df.preis > 100]` |
| `GROUP BY stadt` + Aggregat | `df.groupby("stadt").agg(...)` |
| `JOIN ... ON id` | `df.merge(..., on="id")` |
| `ORDER BY x DESC LIMIT 10` | `df.sort_values("x", ascending=False).head(10)` |

Grundgerüst jeder Abfrage (und ihre logische Auswertungsreihenfolge: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY):

```sql
SELECT   k.stadt, COUNT(*) AS anzahl, AVG(b.preis) AS mittel
FROM     bestellungen b
JOIN     kunden k ON k.kunden_id = b.kunden_id
WHERE    b.datum >= '2024-01-01'
GROUP BY k.stadt
HAVING   COUNT(*) >= 10
ORDER BY mittel DESC;
```

- **JOIN-Arten**: `INNER` (nur Übereinstimmungen), `LEFT` (alle links, rechts NULL wenn fehlend) — die zwei, die man täglich braucht.
- `WHERE` filtert *Zeilen vor* der Gruppierung, `HAVING` filtert *Gruppen danach* — beliebter Anfängerfehler.
- **SQLite** ist eine vollwertige SQL-Datenbank in einer Datei, in Python eingebaut (`sqlite3`), und `pd.read_sql` holt Abfrageergebnisse direkt als DataFrame — perfekt zum Lernen und für kleine Projekte.

Faustregel für die Arbeitsteilung: **Grobfilterung und Joins in SQL** (die Datenbank ist dafür gebaut und hält die Datenmenge klein), **Analyse und Plots in pandas**.

---

## 3. Advanced-Themen

### 3.1 A/B-Tests richtig aufsetzen

Der A/B-Test ist das Experiment der Digitalwirtschaft: Nutzer zufällig in Kontrolle (A) und Variante (B) teilen, Metrik vergleichen. Die **Randomisierung** macht ihn zum Kausalinstrument — im Schnitt unterscheiden sich die Gruppen *nur* durch die Variante (kein Confounding). Die Statistik dazu kennst du aus Modul 02 (Abschnitt 3.1); hier die Design-Seite:

1. **Vorher festlegen**: Metrik, Mindest-Effektgröße (MDE), Signifikanzniveau $\alpha$, Power (üblich 80 %).
2. **Stichprobengröße berechnen** (Power-Analyse): Kleine Effekte brauchen *sehr* viele Nutzer — grob wächst $n$ mit $1/\text{Effekt}^2$. Eine Konversionssteigerung von 2 % auf 2,2 % zu belegen braucht zehntausende Nutzer pro Gruppe.
3. **Laufzeit durchhalten**: volle Wochen (Wochentagseffekte!), und **kein vorzeitiges Stoppen** beim ersten signifikanten Zwischenstand — das ständige Zwischen-Testen bläst die Falsch-Positiv-Rate massiv auf („peeking problem").
4. **Auswerten**: Effektgröße + Konfidenzintervall berichten; Randomisierung prüfen (A/A-Test, Sample Ratio Mismatch); Segmente nur als Hypothesengenerator (multiple testing!).

Typische Praxisfallen: Novelty-Effekt (Neues wird anfangs überklickt), Interferenz zwischen Gruppen (Netzwerkeffekte), und die Versuchung, nach 20 Metriken zu suchen, bis eine signifikant ist.

### 3.2 Simulation & Bootstrap — Statistik ohne Formelsammlung

Moderne Datenanalyse ersetzt viele analytische Formeln durch **Rechenkraft**:

- **Bootstrap**: Ziehe aus deiner Stichprobe (n Werte) viele neue Stichproben *mit Zurücklegen* (je n Werte), berechne die Kennzahl jedes Mal — die Streuung dieser Bootstrap-Kennzahlen schätzt den Standardfehler, ihre Quantile geben ein Konfidenzintervall. Funktioniert für Median, Quantile, Verhältnisse … wo klassische Formeln fehlen oder hässlich sind.
- **Permutationstest**: Unter $H_0$ „kein Gruppenunterschied" sind die Gruppenlabels austauschbar. Mische die Labels tausendfach, berechne jeweils die Teststatistik — der Anteil der gemischten Ergebnisse, die das echte übertreffen, *ist* der p-Wert. Kein Verteilungsmodell nötig, die Logik des p-Werts wird greifbar.

Beide Verfahren baust du im Medium-Projekt selbst — sie sind die beste Intuitionsschule für Inferenzstatistik überhaupt.

### 3.3 Vom Notebook zur Pipeline: Reproduzierbarkeit im Größeren

- **Skripte statt Zellen-Chaos**: Sobald eine Analyse steht, wandert sie in Funktionen/Module mit klaren Ein-/Ausgaben; das Notebook bleibt als Bericht.
- **Datenversionierung light**: Rohdaten unveränderlich („read-only"), jede Transformation als Code (nie von Hand in der CSV editieren!), Zwischenstände benannt und datiert.
- **Big-Data-Ausblick**: Wenn Daten nicht mehr in den RAM passen — spaltenorientierte Formate (Parquet), chunked Processing, DuckDB/Polars als schnelle lokale Engines, Spark & Co. im Cluster. Die *Konzepte* (Filter früh, Joins sparsam, Aggregation nah an den Daten) sind dieselben wie bei SQL.

---

## 4. Zusammenfassung / Cheat-Sheet

**Lineare Regression**
- OLS minimiert $\sum (y - \hat{y})^2$; einfache Regression: $\hat\beta_1 = r \cdot s_y / s_x$
- $R^2$ = erklärte Varianz; steigt automatisch mit mehr Variablen (→ adjustiert / Testdaten)
- Residuenplot: Krümmung = nichtlinear, Trichter = Heteroskedastizität, Leverage-Punkte beachten
- Multiple Regression: Koeffizient = Effekt *ceteris paribus*; Multikollinearität macht Koeffizienten instabil (VIF); Dummy-Falle: eine Referenzkategorie weglassen

**Logistische Regression**
- $P(y=1) = \sigma(\beta^T x)$; $e^\beta$ = Odds Ratio (Chancen, nicht Wahrscheinlichkeiten!)

**Zeitreihen**
- Zerlegung $y = T + S + R$ (oder multiplikativ); ACF lesen (Lag-7-Spitze = Wochenrhythmus)
- Baselines: naiv, saisonal-naiv, gleitendes Mittel, exponentielle Glättung
- **Split immer zeitlich** — nie zufällig!

**Feature Engineering**
- log bei Schiefe, Sin/Cos bei Zyklen, One-Hot bei Nominal, Lags bei Zeitreihen, standardisieren für PCA & Co.
- **Leakage**: nichts verwenden, was es zum Vorhersagezeitpunkt nicht gibt; alles Lernbare nur auf Train lernen

**PCA**
- Hauptkomponenten = orthogonale Richtungen maximaler Varianz (Eigenvektoren der Kovarianzmatrix)
- Scree-Plot für die Anzahl, Loadings für die Interpretation, vorher standardisieren, unsupervised!

**SQL**
- `SELECT … FROM … [JOIN … ON …] WHERE … GROUP BY … HAVING … ORDER BY …`
- WHERE vor Gruppierung, HAVING danach; LEFT JOIN behält alle linken Zeilen
- SQLite + `pd.read_sql` = Lernumgebung ohne Server

**A/B & Resampling**
- Vorab: Metrik, MDE, $\alpha$, Power, Laufzeit; kein Peeking; Effekt + KI berichten
- Bootstrap: Ziehen mit Zurücklegen → KI für (fast) jede Kennzahl
- Permutationstest: Labels mischen → p-Wert ohne Verteilungsannahme

---

## 5. Selbsttest

<details><summary><b>1. Dein Modell hat R² = 0,92, aber der Residuenplot zeigt eine klare U-Form. Was bedeutet das, und was tust du?</b></summary>

Die U-Form heißt: Der Zusammenhang ist **nicht linear** — das Modell über-/unterschätzt systematisch je nach Bereich, trotz hohem $R^2$. Vorhersagen außerhalb des mittleren Bereichs und alle Koeffizienten-Interpretationen sind unzuverlässig. Abhilfe: Transformation ($\log$, Quadratterm) oder ein nichtlineares Modell — und danach wieder den Residuenplot prüfen.
</details>

<details><summary><b>2. Im Hauspreis-Modell hat „Zimmerzahl" plötzlich einen negativen Koeffizienten, sobald „Wohnfläche" mit im Modell ist. Panne?</b></summary>

Nicht unbedingt — zwei Lesarten. (1) **Ceteris paribus** ist das plausibel: *Bei gleicher Wohnfläche* bedeuten mehr Zimmer kleinere Zimmer, was den Preis drücken kann. (2) Zimmerzahl und Wohnfläche sind stark korreliert (**Multikollinearität**), dann sind die einzelnen Koeffizienten instabil und ihre Vorzeichen wenig belastbar (VIF prüfen). In beiden Fällen gilt: Der Koeffizient misst nicht „den Effekt von Zimmern" schlechthin, sondern den Zusatzeffekt bei festgehaltenen anderen Variablen.
</details>

<details><summary><b>3. Logistische Regression: β für „Newsletter abonniert" ist 0,69, also e^β ≈ 2. Formuliere die korrekte Interpretation — und eine falsche, die man oft hört.</b></summary>

Korrekt: Newsletter-Abonnenten haben (ceteris paribus) **doppelt so hohe Odds** (Chancen) zu kaufen. Falsch: „doppelt so hohe Wahrscheinlichkeit". Bei kleinen Wahrscheinlichkeiten liegen Odds und Wahrscheinlichkeit nah beieinander (2 % → ~4 %), bei großen nicht: von 50 % ($\text{Odds}=1$) führt eine Verdopplung der Odds auf 66,7 %, nicht auf 100 %.
</details>

<details><summary><b>4. Warum darf man Zeitreihendaten nicht zufällig in Training und Test aufteilen?</b></summary>

Beim Zufalls-Split liegen Trainingspunkte zeitlich *nach* Testpunkten — das Modell lernt aus der Zukunft der Testfälle (bei autokorrelierten Reihen steckt in $y_{t+1}$ viel Information über $y_t$). Die gemessene Güte ist dann systematisch zu optimistisch und bricht im echten Einsatz ein, wo die Zukunft wirklich unbekannt ist. Korrekt: zeitlicher Split (Vergangenheit → Zukunft), ggf. rollierende Validierung.
</details>

<details><summary><b>5. Nenne drei Beispiele für Leakage und das gemeinsame Prinzip dahinter.</b></summary>

(1) Spalte „Kündigungsdatum" bei der Kündigungs-Vorhersage; (2) Standardisierung mit Mittel/Std aus Train+Test gemeinsam; (3) Zufalls-Split bei Zeitreihen (bzw. Duplikate, die in Train und Test landen). Prinzip: **Information, die zum Vorhersagezeitpunkt nicht verfügbar wäre, fließt ins Training** — die Testleistung misst dann nicht Generalisierung, sondern das Leck.
</details>

<details><summary><b>6. PCA auf Rohdaten mit den Spalten „Einkommen (€)" und „Alter (Jahre)" — was geht schief?</b></summary>

Ohne Standardisierung dominiert das Einkommen (Varianz in Zehntausenden) die Kovarianzmatrix rein wegen seiner Einheit — PC1 zeigt praktisch nur „Einkommen", das Alter geht unter. PCA maximiert Varianz in den vorliegenden Einheiten; vergleichbar werden die Variablen erst nach Standardisierung ($z$-Werte). Deshalb: erst skalieren, dann PCA.
</details>

<details><summary><b>7. Was ist der Unterschied zwischen WHERE und HAVING — und warum liefert `WHERE COUNT(*) > 10` einen Fehler?</b></summary>

`WHERE` filtert **einzelne Zeilen, bevor** gruppiert wird — zu diesem Zeitpunkt existieren noch keine Gruppen, also auch kein `COUNT(*)`. `HAVING` filtert **Gruppen nach** der Aggregation und darf deshalb Aggregatfunktionen verwenden. Logische Reihenfolge: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY.
</details>

<details><summary><b>8. Dein A/B-Test läuft seit 3 Tagen, das Dashboard zeigt p = 0,04. Der Product Manager will sofort ausrollen. Zwei Einwände?</b></summary>

(1) **Peeking**: Wer täglich auf Signifikanz schaut und beim ersten p < 0,05 stoppt, hat eine reale Falsch-Positiv-Rate weit über 5 % — der Test muss die vorab geplante Laufzeit/Stichprobe erreichen. (2) **Drei Tage sind keine volle Woche**: Wochentagseffekte (und Novelty-Effekt) können das Ergebnis treiben. Außerdem fehlt der Blick auf Effektgröße + Konfidenzintervall — „signifikant" allein rechtfertigt keinen Rollout.
</details>

<details><summary><b>9. Erkläre in zwei Sätzen, wie ein Bootstrap-Konfidenzintervall für den Median entsteht.</b></summary>

Man zieht aus der Stichprobe viele (z. B. 10.000) neue Stichproben gleicher Größe **mit Zurücklegen** und berechnet jeweils den Median. Das Intervall zwischen dem 2,5 %- und dem 97,5 %-Quantil dieser Bootstrap-Mediane ist ein 95 %-Konfidenzintervall — ganz ohne Verteilungsannahme oder Formel für den Standardfehler des Medians.
</details>

<details><summary><b>10. Ein Kollege macht PCA und behält die Komponenten, die 95 % der Varianz erklären, um damit eine Zielgröße vorherzusagen. Welcher Denkfehler droht?</b></summary>

PCA ist **unsupervised** — sie kennt die Zielgröße nicht. Viel Varianz heißt nicht viel Vorhersagekraft: Die relevante Information kann in einer varianzschwachen Komponente stecken (und wird weggeworfen), während varianzstarke Komponenten reines, für $y$ irrelevantes Rauschen bündeln können. Für Vorhersagen die Komponentenwahl an der Vorhersageleistung (Validierung) ausrichten, nicht am Varianzanteil.
</details>

---

## 6. Literatur & Quellen

**Lehrbücher**

- **James, Witten, Hastie, Tibshirani — „An Introduction to Statistical Learning" (ISLR), 2. Aufl.** — Kapitel 3 (lineare Regression) und 4.3 (logistische Regression) sind die beste Behandlung des Stoffs; PCA in Kap. 12. **Kostenlos**: https://www.statlearning.com *(einsteigerfreundlich-bis-vertiefend, das Referenzbuch auch für Module 04/05)*
- **Hyndman & Athanasopoulos — „Forecasting: Principles and Practice", 3. Aufl.** — *das* Zeitreihenbuch, komplett **kostenlos**: https://otexts.com/fpp3/ *(einsteigerfreundlich; Beispiele in R, Konzepte sprachunabhängig)*
- **Kohavi, Tang & Xu — „Trustworthy Online Controlled Experiments"** — der Praxisstandard zu A/B-Tests von den Leuten, die sie bei Microsoft/Amazon industrialisiert haben. *(vertiefend)*

**Onlinekurse & Interaktives (kostenlos)**

- **SQLBolt** (https://sqlbolt.com) — interaktives SQL-Lernen im Browser, perfekt vor dem Basic-Projekt *(einsteigerfreundlich)*
- **Mode SQL Tutorial** (https://mode.com/sql-tutorial/) — SQL aus Analysten-Perspektive *(einsteigerfreundlich)*
- **Seeing Theory**, Kapitel „Regression Analysis" und „Frequentist Inference" (https://seeing-theory.brown.edu) *(einsteigerfreundlich)*
- **StatQuest** (YouTube, Josh Starmer): die Videos zu PCA, linearer und logistischer Regression sind Gold für die Intuition *(einsteigerfreundlich)*

**Blogposts / Vertiefung (kostenlos)**

- *explained.ai — „How to interpret PCA plots"* und der Distill-Artikel *„How to Use t-SNE Effectively"* (distill.pub) — Letzterer als Ausblick, warum Dimensionsreduktions-Plots mit Vorsicht zu lesen sind
- Evan Miller: *„How Not To Run an A/B Test"* — der klassische Text zum Peeking-Problem
- Tim Hesterberg: *„What Teachers Should Know About the Bootstrap"* (arXiv) *(vertiefend)*

---

**Nächster Schritt:** `projects/01-basic/` (SQL auf einer echten Datenbank) → `projects/02-medium/` (Bootstrap & Permutationstest selbst bauen) → `projects/03-final/` (Regression & Zeitreihen-Prognose auf den Bike-Sharing-Daten).
