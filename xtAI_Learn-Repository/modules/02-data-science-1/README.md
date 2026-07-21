# Module 02 — Data Science 1

> **Language note.** This document is bilingual. The English version comes first; the German version (*deutsche Fassung*) follows below the horizontal rule.

**What is this about?** Before any model can learn, somebody has to understand the data: read it in, clean it up, describe it, visualise it and draw first conclusions — and know where the pitfalls lie (outliers, missing values, spurious correlations). This module lays the data craft for all following modules: descriptive statistics, pandas, visualisation and the EDA workflow (exploratory data analysis).

**Prerequisites:** Python basics; module 01 is helpful (notebook workflow), but not needed content-wise. School mathematics is enough.

**To do beforehand:** module 01 (for the Jupyter routine).

---

## Learning objectives

After this module you will be able to:

- distinguish types of data (**nominal, ordinal, metric**) and derive from them which statistic and which plot is permitted or sensible in each case,
- describe distributions with **measures of location and spread** (mean, median, quantiles, standard deviation, IQR) and explain when which measure deceives,
- use **pandas** to load, filter, transform, group and join data,
- systematically **clean** a data set (missing values, duplicates, outliers, inconsistent categories) and justify the decisions made,
- choose meaningful **visualisations** and recognise manipulation tricks (truncated axes and the like),
- compute and interpret **correlations** — and explain why correlation is not causation,
- explain the basic idea of **confidence intervals and hypothesis tests** and interpret a p-value correctly,
- carry out a complete **EDA** and summarise the results comprehensibly.

---

## 1. Basics

### 1.1 What is data science?

Data science is the discipline of extracting reliable insight from data. The typical work cycle:

```
ask a question → obtain data → clean → explore (EDA)
     ↑                                     │
     └── communicate ← model/test ←────────┘
```

Two things about this are chronically underestimated:

1. **Cleaning + exploring eat 60–80 % of the time in practice** — not the modelling. That is exactly why this module is so hands-on.
2. The cycle starts with a **question**, not with the data. "Let's see what the data say" without a question almost always ends in spurious patterns (more on this in section 3.2).

### 1.2 Types of data — the scale of measurement determines everything

| Scale | Description | Examples | Permitted statements |
|--|--|--|--|
| **nominal** | categories without order | blood group, city name | equal/unequal, frequencies, mode |
| **ordinal** | categories with order, distances not interpretable | school grades, satisfaction (1–5) | additionally: greater/smaller, median, quantiles |
| **metric** | numbers with interpretable distances | height, price, temperature | additionally: differences, mean, standard deviation |

> **Why this matters:** the "mean of the postcodes" is nonsense (nominal!), the "mean of school grades" is strictly speaking already questionable (ordinal — are the distances 1→2 and 4→5 really the same?). Many data mishaps start with a number being automatically treated as metric just because it is a number.

Metric data are additionally divided into **discrete** (counts: number of children) and **continuous** (measurements: weight).

### 1.3 Descriptive statistics: describing a distribution

**Measures of location** — where is the "centre"?

- **Arithmetic mean**: $\bar{x} = \frac{1}{n}\sum_i x_i$
- **Median**: the middle value of the sorted data (for even $n$: the mean of the two middle ones)
- **Mode**: the most frequent value (the only location measure for nominal data)

**Worked example — why the median exists:** five people earn (in thousands of euros): 32, 36, 38, 41, 45. Mean = 38.4; median = 38 — both similar, all is well. Now a board member replaces the fifth person: 32, 36, 38, 41, **900**. Mean = **209.4** (!), median = **38**. The mean is dragged away by a single extreme value, the median stays put. One says: the median is **robust**. For skewed distributions (incomes, house prices, waiting times) the median is almost always the more honest summary.

**Measures of spread** — how widely are the data spread?

- **Variance** $s^2 = \frac{1}{n-1}\sum_i (x_i - \bar{x})^2$ and **standard deviation** $s = \sqrt{s^2}$ (same unit as the data!)
- **Range**: max − min (extremely sensitive to outliers)
- **Quartiles and IQR**: $Q_1$ (25 % quantile), $Q_3$ (75 % quantile), **interquartile range** $IQR = Q_3 - Q_1$ — the robust alternative to the standard deviation

*(On the denominator $n-1$ instead of $n$: this is the sample correction — the short justification is that $\bar{x}$ was itself estimated from the data, which makes the deviations systematically too small. Details in section 3.1.)*

**Shape of the distribution:** symmetric vs. **right-skewed** (long tail to the right, e.g. income — then mean > median) vs. **left-skewed**; **unimodal** vs. **bimodal** (two peaks — often a hint that two groups have been mixed!).

### 1.4 The basic plots — and when to use which

| Plot | shows | when |
|--|--|--|
| **Histogram** | distribution of a metric variable | always first! shape, skew, outliers, bimodality |
| **Box plot** | median, quartiles, outliers compactly | *comparing* distributions (e.g. per group) |
| **Bar chart** | frequencies of categories | nominal/ordinal data |
| **Scatter plot** | relationship between two metric variables | correlation, clusters, outliers |
| **Line chart** | development over time | time series |

Box plot convention: box = $Q_1$ to $Q_3$, line = median, "whiskers" up to the last point within $1.5 \cdot IQR$, everything outside is drawn as an outlier point.

> **Rule of thumb:** trust no summary whose distribution you have not seen. The famous **Anscombe quartet** data set contains four data sets with *identical* mean, variance and correlation — which look completely different as scatter plots (a straight line, a curve, an outlier artefact, ...). Statistics without a plot is flying blind.

### 1.5 pandas — the data tool

**pandas** is the standard library for tabular data in Python. The two core objects:

- **`Series`**: a labelled column (values + index)
- **`DataFrame`**: a table (several Series with a common index)

The operations you need in 90 % of cases:

```python
import pandas as pd

df = pd.read_csv("file.csv")         # load (also: read_excel, read_json, ...)
df.head(), df.info(), df.describe()  # first inspection: always these three first!
df["column"]                         # select a column (a Series)
df[df["price"] > 100]                # filter rows (boolean mask)
df.loc[rows, columns]                # selection by label, df.iloc by position
df["new"] = df["a"] / df["b"]        # new column (vectorised, no loop!)
df.sort_values("price")              # sort
df.groupby("city")["price"].mean()   # aggregate per group  <- the workhorse
df.merge(other, on="id")             # join tables (like an SQL join)
```

**The most important mental pattern:** in pandas (as in numpy) you formulate operations **vectorised** — whole columns at once, no `for` loop over rows. This is not only about 100 times faster, but also more readable.

---

## 2. Intermediate

### 2.1 Data cleaning — the underestimated core

Real data are dirty. The four standard problems and their tools:

**(a) Missing values** (`NaN`). First *understand*, then treat — the mechanism decides:

- Missing **at random** (the sensor failed occasionally)? → relatively harmless.
- Missing **systematically** (high earners conceal their income more often)? → any naive treatment distorts the analysis!

Options: delete rows (`dropna` — fine if there are few and they are random), fill in (`fillna` with median/mode — "imputation"), or carry them as their own category "unknown". **Always document what you did and how many values were affected.**

**(b) Duplicates** (`duplicated()`, `drop_duplicates()`). Careful: exact duplicates are easy — the hard ones are *near*-duplicates ("Müller GmbH" vs. "Mueller GmbH").

**(c) Outliers.** The standard detector is the **IQR rule**: everything outside $[Q_1 - 1.5\,IQR,\; Q_3 + 1.5\,IQR]$ is suspicious. But: an outlier is not automatically an error! Three cases with three different reactions:

1. **Measurement error** (body height 17.5 m) → correct or remove,
2. **genuine extreme value** (that one large order) → keep it, possibly use robust measures,
3. **special code** (−999 = "missing") → convert to `NaN` — such codes are only found by reading the data documentation!

**(d) Inconsistencies**: capitalisation ("Berlin" / "berlin"), units (euros vs. thousands of euros), date formats, leading spaces. Tools: `str.strip()`, `str.lower()`, `pd.to_datetime()`, `astype()`, mapping dictionaries.

> **Basic attitude:** cleaning is a chain of *justified decisions*, not a mechanical scrub. A clean analysis notebook documents every decision ("12 rows removed, because ...") — otherwise the result is not reproducible.

### 2.2 Tidy data

A data set is **tidy** if: every variable = one column, every observation = one row, every observational unit = one table. Much raw data comes in "wide" form (one column per year); for analyses you usually need it "long" (columns: year, value). Tools: `melt` (wide → long), `pivot`/`pivot_table` (long → wide). Rule of thumb: if column names feel like *values* (2021, 2022, 2023, ...), the data set is not tidy.

### 2.3 Grouping and aggregating: split–apply–combine

The most powerful analysis pattern: **split** the data (by groups), **evaluate** per group, **combine** the results:

```python
df.groupby("category")["revenue"].agg(["count", "mean", "median", "std"])
df.groupby(["region", "month"])["revenue"].sum().unstack()   # -> pivot table
```

This answers questions like "do the groups differ?" in one line — and this is exactly where **Simpson's paradox** lurks too (section 3.2).

### 2.4 Correlation

The **Pearson correlation coefficient** measures the strength of the *linear* relationship between two metric variables:

$$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2}\sqrt{\sum_i (y_i - \bar{y})^2}} \in [-1, 1]$$

Intuition: you check whether $x$ and $y$ lie *jointly* above or below their means. $r = \pm 1$: a perfect straight line; $r = 0$: no *linear* relationship.

Three classic traps:

1. **$r = 0$ does not mean "no relationship"**: a perfect U shape (e.g. $y = x^2$) has $r \approx 0$. → Look at the scatter plot; for monotone but non-linear relationships use the **Spearman correlation** (= Pearson on the ranks, robust against outliers).
2. **Correlation is not causation.** Ice cream sales correlate with drowning accidents — the cause is summer (**confounder**, a common cause). Causal statements need experiments (randomisation!) or very careful methodology.
3. **Outliers**: a single extreme point can heave $r$ from 0 to 0.9 (see Anscombe).

### 2.5 The EDA workflow

Exploratory data analysis is a systematic procedure, not wild plotting around:

1. **Overview**: `shape`, `info()`, `head()` — what is a row? what do the columns mean? (read the data documentation!)
2. **Quality**: missing values, duplicates, implausible values, special codes → clean (2.1)
3. **Univariate**: every important variable on its own — histogram/box plot or bar chart, key figures
4. **Bivariate**: relationships — scatter plots, correlation matrix, group comparisons (`groupby`)
5. **Time/structure**: trends, seasonal patterns, group differences
6. **Record it**: every figure gets one sentence of interpretation. A figure without a statement is decoration.

**Visualisation ethics** (also for reading figures yourself): truncated y-axes exaggerate effects, dual axes suggest relationships, 3D pie charts distort proportions, cherry-picked periods reverse trends. Whoever knows the tricks falls for them less often.

---

## 3. Advanced topics

### 3.1 From the sample to the statement: inferential statistics

You almost never have *all* the data (the **population**) — only a **sample**. Key figures of the sample (e.g. $\bar{x}$) fluctuate from sample to sample. So how uncertain is a statement?

- The **standard error** of the mean is $SE = s/\sqrt{n}$ — the uncertainty only decreases with the *square root* of the amount of data (4 times as much data → half the error).
- A **95 % confidence interval** $\bar{x} \pm 1.96 \cdot SE$ (for large $n$) is a procedure that, repeated infinitely often, captures the true value in 95 % of cases. *Careful with the interpretation:* it does not mean "the true value lies in exactly this interval with 95 % probability" — the true value is not a random quantity, the interval is.
- Why does this work at all? The **central limit theorem**: means of many independent values are approximately normally distributed — *no matter how the individual values are distributed*. That is why the bell curve appears everywhere.

**Hypothesis tests in 5 steps** (using the example "is variant B of the website better than A?"):

1. **Null hypothesis** $H_0$: no difference (B = A). Alternative hypothesis $H_1$: B ≠ A.
2. Choose a **test statistic** (e.g. the difference in conversion rates, the t statistic).
3. Compute the **p-value**: the probability, *under $H_0$*, of seeing a value at least as extreme as the one observed.
4. If $p < \alpha$ (usually 0.05), reject $H_0$ — "statistically significant".
5. Report **effect size + confidence interval**, not just the p-value!

**The three most common misunderstandings about p-values:**

- $p$ is **not** the probability that $H_0$ is true.
- "significant" does **not** mean "large" or "important": with $n = 10^6$ even a tiny, practically irrelevant difference becomes significant.
- **Multiple testing / p-hacking**: whoever tests 20 hypotheses finds on average one "significant" one purely by chance ($\alpha = 1/20$). Whoever digs in data long enough *always* finds something. Antidotes: fix hypotheses in advance, corrections (e.g. Bonferroni: $\alpha / m$ for $m$ tests), confirmation on fresh data.

### 3.2 Simpson's paradox — when groups reverse the direction

A relationship can point one way in *every subgroup* and the other way *overall*. Real example (Berkeley 1973): overall, men were admitted to study more often than women — but in almost every individual department it was the other way round or equal. Resolution: women applied disproportionately to the toughest subjects. The choice of subject was a confounder.

**Lesson:** aggregated numbers can be fundamentally misleading without a breakdown by group — and *which* level (aggregated or grouped) is the "right" one is not a statistical question but a **substantive, causal** one.

### 3.3 Data ethics, data protection, reproducibility

- **Bias in data**: samples are rarely representative (survivorship bias: you analyse only the customers who *stayed*; selection bias: online surveys only reach online people). The famous bullet-hole case: the places to be reinforced are those where returning aircraft have *no* holes — those hit there never came back.
- **Data protection (GDPR)**: process personal data only for a specified purpose and minimised. Important for the craft: "anonymisation" is harder than it looks — even a few attributes (postcode + date of birth + sex) re-identify most people.
- **Reproducibility**: fixed random seeds, documented cleaning steps, code + data under version control, notebooks that run through "Restart & Run All" without errors. An analysis that ran only once on one laptop is not an analysis but an anecdote.

---

## 4. Summary / cheat sheet

**Scales of measurement**: nominal (mode) → ordinal (+ median, quantiles) → metric (+ mean, sd)

**Key figures**
- Mean $\bar{x}$ (sensitive) vs. median (robust); right-skewed implies mean > median
- $s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$; $IQR = Q_3 - Q_1$ (robust)
- Outlier rule of thumb: outside $[Q_1 - 1.5\,IQR,\ Q_3 + 1.5\,IQR]$

**Plots**: histogram (distribution), box plot (comparison), bars (categories), scatter (relationship), line (time)

**pandas minimum**: `read_csv` → `info`/`describe`/`head` → mask `df[df.x > 0]` → `groupby(...).agg(...)` → `merge` → `melt`/`pivot`

**Cleaning**: NaN (understand the mechanism! dropna/fillna/category), duplicates, outliers (error? extreme value? special code?), inconsistencies — document everything

**Correlation**: Pearson $r$ = linear, Spearman = monotone/robust; $r=0$ is not independence; correlation is not causation (confounders!)

**Inference**: $SE = s/\sqrt{n}$, 95 % CI approximately $\bar{x} \pm 1.96\,SE$, p-value = P(this extreme | $H_0$), significant is not important, correct for multiple tests

**Warnings**: Anscombe (always plot!), Simpson (always group!), survivorship/selection bias (who is missing from the data?)

---

## 5. Self-test

<details><summary><b>1. The mean customer satisfaction (scale 1–5) has risen from 3.8 to 4.1. Two objections?</b></summary>

(1) The scale is **ordinal** — whether the distances between the levels are equal is unclear, so the mean is only interpretable to a limited extent (median/distribution of the levels would be cleaner). (2) Without **uncertainty** (n? confidence interval?) and without the **distribution** (more 5s — or fewer 1s?) the difference cannot be assessed; a changed composition of respondents (selection bias) could also be the cause.
</details>

<details><summary><b>2. When mean, when median — and how do you recognise it in the histogram?</b></summary>

Median for skewed distributions or outliers (incomes, prices, waiting times), mean for approximately symmetric distributions (then both are almost equal — the mean is statistically more efficient and easier to compute with further). In the histogram: a long tail on one side implies skew implies median. Practical quick check: if mean and median differ noticeably, the distribution is skewed or burdened by outliers.
</details>

<details><summary><b>3. A column "income" contains some values of −999. What is this probably, and what do you do?</b></summary>

Probably a **special code for "missing/no answer"** (check the data documentation!). Treatment: convert to `NaN` (`df["income"].replace(-999, np.nan)`), then treat like missing values — and check whether the missingness is systematic. On no account leave it in: −999 would massively distort mean, variance and correlations.
</details>

<details><summary><b>4. Why can a variable with Pearson r about 0 nevertheless be strongly related to another?</b></summary>

Pearson measures only **linear** relationships. A U-shaped relationship (e.g. $y = x^2$ around 0) has $r \approx 0$: for small $x$, $y$ falls, for large ones it rises — the linear parts cancel out. Therefore: look at the scatter plot; for monotone non-linear relationships use Spearman.
</details>

<details><summary><b>5. A study finds: coffee drinkers have more heart attacks. Name a plausible confounder and how causality could be tested.</b></summary>

For example **smoking**: smokers drink (statistically) more coffee *and* have more heart attacks — the coffee may be innocent. Testing: stratify/adjust for confounders (compare within smokers and within non-smokers) or — the gold standard — a **randomised experiment** that decouples coffee assignment from lifestyle (practically difficult here, hence observational studies with careful adjustment).
</details>

<details><summary><b>6. What exactly does a p-value of 0.03 say — and what does it NOT say?</b></summary>

It says: *if* the null hypothesis were true, one would see a result at least this extreme in 3 % of cases. It does **not** say: "$H_0$ is 3 % true", nor "the effect is genuine with 97 % probability", nor "the effect is large/important". And if 20 tests were run, a single $p = 0.03$ is worth little (multiple testing).
</details>

<details><summary><b>7. Drug A has a higher success rate than B in both hospitals, but a lower one overall. How can that be?</b></summary>

**Simpson's paradox**: A was predominantly used in the hospital with the severe cases (low base rate), B predominantly for mild cases. Case severity is a confounder; in aggregate B only wins because it had the easier patients. What is substantively relevant here is the comparison *within* comparable case groups — that is, A.
</details>

<details><summary><b>8. Why is "we removed the outliers" without further details a warning sign?</b></summary>

Because outliers can be three completely different things (measurement errors, genuine extreme values, special codes) and removing them can change results dramatically. Without stating the criterion (e.g. the IQR rule), the number and the justification, the analysis is neither traceable nor reproducible — and possibly inconvenient genuine data points were "cleaned away".
</details>

<details><summary><b>9. Your colleague correlated 47 variables against each other in a customer data set and found 3 "highly significant" relationships. Your assessment?</b></summary>

47 variables give $\binom{47}{2} = 1081$ pairs — at $\alpha = 0.05$ one expects about 54 "significant" correlations purely by chance. Three findings are therefore *fewer* than chance delivers. Procedure: correct for multiple testing, and regard the candidates as *hypotheses* that have to be confirmed on new, independent data.
</details>

<details><summary><b>10. What does "tidy data" mean, and why is `df.groupby` on untidy data often impossible?</b></summary>

Tidy: every variable one column, every observation one row. If, for example, years are present as columns (wide), then "year" is not a variable you could group or filter by — `groupby("year")` only works after `melt` (wide → long). The tidy structure is the precondition for the standard tools (groupby, merge, plotting libraries) to work.
</details>

---

## 6. Literature and sources

**Textbooks**

- **Wickham, Çetinkaya-Rundel & Grolemund — "R for Data Science", 2nd ed.** — free online (r4ds.hadley.nz). Yes, R instead of Python — but the chapters on EDA, tidy data and data quality are, language-independently, the best there is. *(beginner-friendly, free)*
- **McKinney — "Python for Data Analysis", 3rd ed.** — by the creator of pandas; free online: https://wesmckinney.com/book/ *(beginner-friendly to advanced, free)*
- **Freedman, Pisani & Purves — "Statistics"** — the classic for statistical intuition without a desert of formulas. *(beginner-friendly)*
- **Spiegelhalter — "The Art of Statistics"** — statistical thinking through real cases. *(very beginner-friendly)*

**Online courses and documentation (free)**

- **pandas documentation, "10 minutes to pandas"** + user guide: https://pandas.pydata.org/docs/ — the reference you will have permanently open anyway
- **Kaggle Learn**: the courses "Pandas" and "Data Visualization" — short, interactive, in the browser *(beginner-friendly)*
- **Our World in Data** (ourworldindata.org): excellent examples of clean data visualisation and open data sets to practise on

**Interactive / blog posts (free)**

- *Seeing Theory* (Brown University): https://seeing-theory.brown.edu — interactive visualisation of probability, CIs and regression *(beginner-friendly, beautiful)*
- Autodesk Research: *Same Stats, Different Graphs* (the "Datasaurus") — Anscombe on steroids
- Tyler Vigen: *Spurious Correlations* — absurd spurious correlations as an inoculation against credulity about correlation

**Advanced**

- Bickel, Hammel & O'Connell (1975): *Sex Bias in Graduate Admissions: Data from Berkeley* (Science) — the original of the Simpson's paradox case
- Wasserstein & Lazar (2016): *The ASA Statement on p-Values* — the official clarification of what p-values do (not) mean

---

**Next step:** `projects/01-basic/` (pandas basics on real data) → `projects/02-medium/` (rescue a dirty data set) → `projects/03-final/` (a complete EDA on a real data set).

---
---

# Modul 02 — Data Science 1 (deutsche Fassung)

**Worum geht es?** Bevor irgendein Modell lernen kann, muss jemand die Daten verstehen: einlesen, aufräumen, beschreiben, visualisieren und erste Schlüsse ziehen — und wissen, wo die Fallstricke lauern (Ausreißer, fehlende Werte, Scheinkorrelationen). Dieses Modul legt das Daten-Handwerkszeug für alle folgenden Module: deskriptive Statistik, pandas, Visualisierung und den EDA-Workflow (Exploratory Data Analysis).

**Vorkenntnisse:** Python-Grundlagen; Modul 01 ist hilfreich (Notebook-Workflow), aber inhaltlich nicht nötig. Schulmathematik reicht.

**Vorher zu machen:** Modul 01 (wegen Jupyter-Routine).

---

## Lernziele

Nach diesem Modul kannst du:

- Datenarten (**nominal, ordinal, metrisch**) unterscheiden und daraus ableiten, welche Statistik und welcher Plot jeweils erlaubt/sinnvoll ist,
- Verteilungen mit **Lage- und Streuungsmaßen** (Mittelwert, Median, Quantile, Standardabweichung, IQR) beschreiben und erklären, wann welches Maß trügt,
- mit **pandas** Daten laden, filtern, transformieren, gruppieren und zusammenführen,
- einen Datensatz systematisch **bereinigen** (fehlende Werte, Duplikate, Ausreißer, inkonsistente Kategorien) und die getroffenen Entscheidungen begründen,
- aussagekräftige **Visualisierungen** wählen und Manipulationstricks (abgeschnittene Achsen & Co.) erkennen,
- **Korrelationen** berechnen, interpretieren — und erklären, warum Korrelation keine Kausalität ist,
- die Grundidee von **Konfidenzintervallen und Hypothesentests** erklären und einen p-Wert korrekt interpretieren,
- eine vollständige **EDA** durchführen und die Ergebnisse verständlich zusammenfassen.

---

## 1. Grundlagen (Basics)

### 1.1 Was ist Data Science?

Data Science ist die Disziplin, aus Daten belastbare Erkenntnisse zu gewinnen. Der typische Arbeitszyklus:

```
Frage stellen → Daten beschaffen → Bereinigen → Explorieren (EDA)
     ↑                                              │
     └── Kommunizieren ← Modellieren/Testen ←───────┘
```

Zwei Dinge daran werden chronisch unterschätzt:

1. **Bereinigen + Explorieren fressen in der Praxis 60–80 % der Zeit** — nicht das Modellieren. Genau deshalb ist dieses Modul so handwerklich.
2. Der Zyklus beginnt mit einer **Frage**, nicht mit den Daten. „Mal schauen, was die Daten sagen" ohne Frage endet fast immer in Scheinmustern (mehr dazu in Abschnitt 3.2).

### 1.2 Datenarten — das Skalenniveau bestimmt alles

| Skalenniveau | Beschreibung | Beispiele | erlaubte Aussagen |
|--|--|--|--|
| **nominal** | Kategorien ohne Ordnung | Blutgruppe, Stadtname | gleich/ungleich, Häufigkeiten, Modus |
| **ordinal** | Kategorien mit Ordnung, Abstände nicht interpretierbar | Schulnoten, Zufriedenheit (1–5) | zusätzlich: größer/kleiner, Median, Quantile |
| **metrisch** | Zahlen mit interpretierbaren Abständen | Größe, Preis, Temperatur | zusätzlich: Differenzen, Mittelwert, Standardabweichung |

> **Warum das wichtig ist:** Der „Mittelwert der Postleitzahlen" ist Unsinn (nominal!), der „Mittelwert der Schulnoten" ist streng genommen schon fragwürdig (ordinal — sind die Abstände 1→2 und 4→5 wirklich gleich groß?). Viele Datenpannen beginnen damit, dass eine Zahl automatisch als metrisch behandelt wird, nur weil sie eine Zahl ist.

Zusätzlich unterscheidet man metrische Daten in **diskret** (Zählwerte: Kinderzahl) und **stetig** (Messwerte: Gewicht).

### 1.3 Deskriptive Statistik: eine Verteilung beschreiben

**Lagemaße** — wo liegt das „Zentrum"?

- **Arithmetisches Mittel**: $\bar{x} = \frac{1}{n}\sum_i x_i$
- **Median**: der mittlere Wert der sortierten Daten (bei geradem $n$: Mittel der beiden mittleren)
- **Modus**: der häufigste Wert (einziges Lagemaß für nominale Daten)

**Durchgerechnetes Beispiel — warum der Median existiert:** Fünf Personen verdienen (in T€): 32, 36, 38, 41, 45. Mittel = 38,4; Median = 38 — beide ähnlich, alles gut. Jetzt ersetzt ein Vorstand die fünfte Person: 32, 36, 38, 41, **900**. Mittel = **209,4** (!), Median = **38**. Der Mittelwert wird von einem einzigen Extremwert verschleppt, der Median bleibt stehen. Man sagt: der Median ist **robust**. Bei schiefen Verteilungen (Einkommen, Hauspreise, Wartezeiten) ist der Median fast immer die ehrlichere Zusammenfassung.

**Streuungsmaße** — wie breit liegen die Daten?

- **Varianz** $s^2 = \frac{1}{n-1}\sum_i (x_i - \bar{x})^2$ und **Standardabweichung** $s = \sqrt{s^2}$ (gleiche Einheit wie die Daten!)
- **Spannweite**: max − min (extrem ausreißerempfindlich)
- **Quartile & IQR**: $Q_1$ (25 %-Quantil), $Q_3$ (75 %-Quantil), **Interquartilsabstand** $IQR = Q_3 - Q_1$ — die robuste Alternative zur Standardabweichung

*(Zum Nenner $n-1$ statt $n$: Das ist die Stichproben-Korrektur — die Kurzbegründung lautet, dass $\bar{x}$ selbst aus den Daten geschätzt wurde und die Abweichungen dadurch systematisch zu klein ausfallen. Details in Abschnitt 3.1.)*

**Form der Verteilung:** symmetrisch vs. **rechtsschief** (langer Schwanz nach rechts, z. B. Einkommen — dann Mittel > Median) vs. **linksschief**; **unimodal** vs. **bimodal** (zwei Gipfel — oft ein Hinweis, dass zwei Gruppen vermischt sind!).

### 1.4 Die Grundplots — und wann welcher

| Plot | zeigt | wann |
|--|--|--|
| **Histogramm** | Verteilung einer metrischen Variable | immer zuerst! Form, Schiefe, Ausreißer, Bimodalität |
| **Boxplot** | Median, Quartile, Ausreißer kompakt | Verteilungen *vergleichen* (z. B. je Gruppe) |
| **Balkendiagramm** | Häufigkeiten von Kategorien | nominale/ordinale Daten |
| **Scatterplot** | Zusammenhang zweier metrischer Variablen | Korrelation, Cluster, Ausreißer |
| **Liniendiagramm** | Verlauf über die Zeit | Zeitreihen |

Boxplot-Konvention: Box = $Q_1$ bis $Q_3$, Strich = Median, „Whiskers" bis zum letzten Punkt innerhalb von $1{,}5 \cdot IQR$, alles außerhalb wird als Ausreißer-Punkt gezeichnet.

> **Faustregel:** Traue keiner Zusammenfassung, deren Verteilung du nicht gesehen hast. Der berühmte **Anscombe-Quartett**-Datensatz enthält vier Datensätze mit *identischem* Mittelwert, Varianz und Korrelation — die als Scatterplots völlig verschieden aussehen (eine Gerade, eine Kurve, ein Ausreißer-Artefakt …). Statistik ohne Plot ist Blindflug.

### 1.5 pandas — das Datenwerkzeug

**pandas** ist die Standard-Bibliothek für Tabellendaten in Python. Die zwei Kernobjekte:

- **`Series`**: eine beschriftete Spalte (Werte + Index)
- **`DataFrame`**: eine Tabelle (mehrere Series mit gemeinsamem Index)

Die Operationen, die du in 90 % der Fälle brauchst:

```python
import pandas as pd

df = pd.read_csv("datei.csv")        # laden (auch: read_excel, read_json, ...)
df.head(), df.info(), df.describe()  # Erstinspektion: immer diese drei zuerst!
df["spalte"]                         # Spalte auswählen (eine Series)
df[df["preis"] > 100]                # Zeilen filtern (boolesche Maske)
df.loc[zeilen, spalten]              # Auswahl per Label,  df.iloc per Position
df["neu"] = df["a"] / df["b"]        # neue Spalte (vektorisiert, keine Schleife!)
df.sort_values("preis")              # sortieren
df.groupby("stadt")["preis"].mean()  # aggregieren pro Gruppe  ← das Arbeitspferd
df.merge(andere, on="id")            # Tabellen verknüpfen (wie SQL-Join)
```

**Das wichtigste Denkmuster:** In pandas (wie in numpy) formuliert man Operationen **vektorisiert** — ganze Spalten auf einmal, keine `for`-Schleife über Zeilen. Das ist nicht nur ~100× schneller, sondern auch lesbarer.

---

## 2. Aufbau (Intermediate)

### 2.1 Datenbereinigung — der unterschätzte Kern

Echte Daten sind schmutzig. Die vier Standardprobleme und ihre Werkzeuge:

**(a) Fehlende Werte** (`NaN`). Erst *verstehen*, dann behandeln — der Mechanismus entscheidet:

- Fehlt **zufällig** (Sensor fiel gelegentlich aus)? → relativ harmlos.
- Fehlt **systematisch** (Gutverdiener verschweigen ihr Einkommen öfter)? → jede naive Behandlung verzerrt die Analyse!

Optionen: Zeilen löschen (`dropna` — okay, wenn wenige und zufällig), auffüllen (`fillna` mit Median/Modus — „Imputation"), oder als eigene Kategorie „unbekannt" führen. **Immer dokumentieren, was man getan hat und wie viele Werte betroffen waren.**

**(b) Duplikate** (`duplicated()`, `drop_duplicates()`). Achtung: exakte Duplikate sind leicht — schwer sind *Fast*-Duplikate („Müller GmbH" vs. „Mueller GmbH").

**(c) Ausreißer.** Standard-Detektor ist die **IQR-Regel**: verdächtig ist alles außerhalb von $[Q_1 - 1{,}5\,IQR,\; Q_3 + 1{,}5\,IQR]$. Aber: Ein Ausreißer ist nicht automatisch ein Fehler! Drei Fälle mit drei verschiedenen Reaktionen:

1. **Messfehler** (Körpergröße 17,5 m) → korrigieren oder entfernen,
2. **echter Extremwert** (der eine Großauftrag) → drinlassen, ggf. robuste Maße verwenden,
3. **Sondercode** (−999 = „fehlend") → in `NaN` umwandeln — solche Codes findet man nur, wenn man die Datendokumentation liest!

**(d) Inkonsistenzen**: Groß-/Kleinschreibung („Berlin" / „berlin"), Einheiten (€ vs. T€), Datumsformate, führende Leerzeichen. Werkzeuge: `str.strip()`, `str.lower()`, `pd.to_datetime()`, `astype()`, Mapping-Dictionaries.

> **Grundhaltung:** Bereinigung ist eine Kette von *begründeten Entscheidungen*, kein mechanisches Durchputzen. Ein sauberes Analyse-Notebook dokumentiert jede Entscheidung („12 Zeilen entfernt, weil …") — sonst ist das Ergebnis nicht reproduzierbar.

### 2.2 Tidy Data

Ein Datensatz ist **tidy**, wenn: jede Variable = eine Spalte, jede Beobachtung = eine Zeile, jede Beobachtungseinheit = eine Tabelle. Viele Rohdaten kommen „wide" daher (eine Spalte pro Jahr); für Analysen braucht man sie meist „long" (Spalten: Jahr, Wert). Werkzeuge: `melt` (wide → long), `pivot`/`pivot_table` (long → wide). Faustregel: Wenn sich Spaltennamen wie *Werte* anfühlen (2021, 2022, 2023 …), ist der Datensatz nicht tidy.

### 2.3 Gruppieren und Aggregieren: Split–Apply–Combine

Das mächtigste Analyse-Muster: Daten **aufteilen** (nach Gruppen), pro Gruppe **auswerten**, Ergebnisse **zusammensetzen**:

```python
df.groupby("kategorie")["umsatz"].agg(["count", "mean", "median", "std"])
df.groupby(["region", "monat"])["umsatz"].sum().unstack()   # → Pivot-Tabelle
```

Damit beantwortet man Fragen wie „Unterscheiden sich die Gruppen?" in einer Zeile — und genau hier lauert auch **Simpson's Paradox** (Abschnitt 3.2).

### 2.4 Korrelation

Der **Pearson-Korrelationskoeffizient** misst die Stärke des *linearen* Zusammenhangs zweier metrischer Variablen:

$$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2}\sqrt{\sum_i (y_i - \bar{y})^2}} \in [-1, 1]$$

Intuition: Man prüft, ob $x$ und $y$ *gemeinsam* über bzw. unter ihren Mittelwerten liegen. $r = \pm 1$: perfekte Gerade; $r = 0$: kein *linearer* Zusammenhang.

Drei klassische Fallen:

1. **$r = 0$ heißt nicht „kein Zusammenhang"**: Eine perfekte U-Form (z. B. $y = x^2$) hat $r \approx 0$. → Scatterplot anschauen; bei monotonen, aber nichtlinearen Zusammenhängen **Spearman-Korrelation** verwenden (= Pearson auf den Rängen, robust gegen Ausreißer).
2. **Korrelation ≠ Kausalität.** Eisverkauf korreliert mit Ertrinkungsunfällen — Ursache ist der Sommer (**Confounder**, gemeinsame Ursache). Kausale Aussagen brauchen Experimente (Randomisierung!) oder sehr sorgfältige Methodik.
3. **Ausreißer**: Ein einzelner Extrempunkt kann $r$ von 0 auf 0,9 hieven (siehe Anscombe).

### 2.5 Der EDA-Workflow

Explorative Datenanalyse ist ein systematisches Vorgehen, kein wildes Herumplotten:

1. **Überblick**: `shape`, `info()`, `head()` — Was ist eine Zeile? Was bedeuten die Spalten? (Datendokumentation lesen!)
2. **Qualität**: fehlende Werte, Duplikate, unplausible Werte, Sondercodes → bereinigen (2.1)
3. **Univariat**: jede wichtige Variable einzeln — Histogramm/Boxplot bzw. Balkendiagramm, Kennzahlen
4. **Bivariat**: Zusammenhänge — Scatterplots, Korrelationsmatrix, Gruppenvergleiche (`groupby`)
5. **Zeit/Struktur**: Trends, Saisonmuster, Gruppenunterschiede
6. **Festhalten**: Jede Grafik bekommt einen Satz Interpretation. Eine Grafik ohne Aussage ist Deko.

**Visualisierungs-Ethik** (auch fürs eigene Lesen von Grafiken): abgeschnittene y-Achsen übertreiben Effekte, Doppelachsen suggerieren Zusammenhänge, 3D-Tortendiagramme verzerren Anteile, cherry-gepickte Zeiträume drehen Trends. Wer die Tricks kennt, fällt seltener darauf herein.

---

## 3. Advanced-Themen

### 3.1 Von der Stichprobe zur Aussage: schließende Statistik

Fast nie hat man *alle* Daten (die **Population**) — nur eine **Stichprobe**. Kennzahlen der Stichprobe (z. B. $\bar{x}$) schwanken von Stichprobe zu Stichprobe. Wie unsicher ist also eine Aussage?

- Der **Standardfehler** des Mittelwerts ist $SE = s/\sqrt{n}$ — die Unsicherheit sinkt nur mit der *Wurzel* der Datenmenge (4× so viele Daten → halber Fehler).
- Ein **95 %-Konfidenzintervall** $\bar{x} \pm 1{,}96 \cdot SE$ (bei großem $n$) ist ein Verfahren, das bei unendlicher Wiederholung in 95 % der Fälle den wahren Wert einfängt. *Vorsicht bei der Interpretation:* Es heißt nicht „der wahre Wert liegt mit 95 % Wahrscheinlichkeit in genau diesem Intervall" — der wahre Wert ist keine Zufallsgröße, das Intervall ist es.
- Warum funktioniert das überhaupt? Der **zentrale Grenzwertsatz**: Mittelwerte vieler unabhängiger Werte sind näherungsweise normalverteilt — *egal wie die Einzelwerte verteilt sind*. Deshalb taucht die Glockenkurve überall auf.

**Hypothesentests in 5 Schritten** (am Beispiel „Ist Variante B der Website besser als A?"):

1. **Nullhypothese** $H_0$: kein Unterschied (B = A). Alternativhypothese $H_1$: B ≠ A.
2. Wähle eine **Teststatistik** (z. B. Differenz der Konversionsraten, t-Statistik).
3. Berechne den **p-Wert**: die Wahrscheinlichkeit, *unter $H_0$* einen mindestens so extremen Wert zu sehen wie beobachtet.
4. Ist $p < \alpha$ (üblich: 0,05), verwirf $H_0$ — „statistisch signifikant".
5. Berichte **Effektgröße + Konfidenzintervall**, nicht nur den p-Wert!

**Die drei häufigsten p-Wert-Missverständnisse:**

- $p$ ist **nicht** die Wahrscheinlichkeit, dass $H_0$ stimmt.
- „signifikant" heißt **nicht** „groß" oder „wichtig": Bei $n = 10^6$ wird auch ein winziger, praktisch irrelevanter Unterschied signifikant.
- **Multiple Testing / p-Hacking**: Wer 20 Hypothesen testet, findet im Schnitt eine „signifikante" rein durch Zufall ($\alpha = 1/20$). Wer lange genug in Daten wühlt, findet *immer* etwas. Gegenmittel: Hypothesen vorher festlegen, Korrekturen (z. B. Bonferroni: $\alpha / m$ bei $m$ Tests), Bestätigung auf frischen Daten.

### 3.2 Simpson's Paradox — wenn Gruppen die Richtung drehen

Ein Zusammenhang kann in *jeder Untergruppe* in die eine Richtung zeigen und *insgesamt* in die andere. Reales Beispiel (Berkeley 1973): Insgesamt wurden Männer häufiger zum Studium zugelassen als Frauen — in fast jedem einzelnen Fachbereich war es aber umgekehrt oder gleich. Auflösung: Frauen bewarben sich überproportional auf die härtesten Fächer. Die Fachwahl war ein Confounder.

**Lehre:** Aggregierte Zahlen können ohne Gruppenaufschlüsselung fundamental in die Irre führen — und *welche* Ebene (aggregiert oder gruppiert) die „richtige" ist, ist keine statistische, sondern eine **inhaltlich-kausale** Frage.

### 3.3 Datenethik, Datenschutz, Reproduzierbarkeit

- **Bias in Daten**: Stichproben sind selten repräsentativ (Survivorship Bias: Man analysiert nur die Kunden, die *geblieben* sind; Selection Bias: Online-Umfragen erreichen nur Online-Menschen). Der berühmte Bomber-Einschusslöcher-Fall: Verstärkt werden müssen die Stellen, an denen zurückgekehrte Flugzeuge *keine* Löcher haben — die dort Getroffenen kamen nie zurück.
- **Datenschutz (DSGVO)**: personenbezogene Daten nur zweckgebunden und minimiert verarbeiten. Wichtig fürs Handwerk: „Anonymisierung" ist schwerer als gedacht — schon wenige Merkmale (PLZ + Geburtsdatum + Geschlecht) reidentifizieren die meisten Personen.
- **Reproduzierbarkeit**: feste Random-Seeds, dokumentierte Bereinigungsschritte, Code + Daten versioniert, Notebooks laufen „Restart & Run All" fehlerfrei durch. Eine Analyse, die nur einmal auf einem Laptop lief, ist keine Analyse, sondern eine Anekdote.

---

## 4. Zusammenfassung / Cheat-Sheet

**Skalenniveaus**: nominal (Modus) → ordinal (+ Median, Quantile) → metrisch (+ Mittelwert, sd)

**Kennzahlen**
- Mittel $\bar{x}$ (empfindlich) vs. Median (robust); rechtsschief ⇒ Mittel > Median
- $s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$; $IQR = Q_3 - Q_1$ (robust)
- Ausreißer-Faustregel: außerhalb $[Q_1 - 1{,}5\,IQR,\ Q_3 + 1{,}5\,IQR]$

**Plots**: Histogramm (Verteilung) · Boxplot (Vergleich) · Balken (Kategorien) · Scatter (Zusammenhang) · Linie (Zeit)

**pandas-Minimum**: `read_csv` → `info`/`describe`/`head` → Maske `df[df.x > 0]` → `groupby(...).agg(...)` → `merge` → `melt`/`pivot`

**Bereinigung**: NaN (Mechanismus verstehen! dropna/fillna/Kategorie) · Duplikate · Ausreißer (Fehler? Extremwert? Sondercode?) · Inkonsistenzen — alles dokumentieren

**Korrelation**: Pearson $r$ = linear, Spearman = monoton/robust; $r=0$ ≠ unabhängig; Korrelation ≠ Kausalität (Confounder!)

**Inferenz**: $SE = s/\sqrt{n}$ · 95 %-KI ≈ $\bar{x} \pm 1{,}96\,SE$ · p-Wert = P(so extrem | $H_0$) · signifikant ≠ wichtig · multiple Tests korrigieren

**Warnungen**: Anscombe (immer plotten!) · Simpson (immer gruppieren!) · Survivorship/Selection Bias (wer fehlt in den Daten?)

---

## 5. Selbsttest

<details><summary><b>1. Der Mittelwert der Kundenzufriedenheit (Skala 1–5) ist von 3,8 auf 4,1 gestiegen. Zwei Einwände?</b></summary>

(1) Die Skala ist **ordinal** — ob die Abstände zwischen den Stufen gleich sind, ist unklar, damit ist der Mittelwert nur eingeschränkt interpretierbar (Median/Verteilung der Stufen wäre sauberer). (2) Ohne **Unsicherheit** (n? Konfidenzintervall?) und ohne die **Verteilung** (mehr 5er — oder weniger 1er?) ist die Differenz nicht bewertbar; auch eine veränderte Zusammensetzung der Befragten (Selection Bias) könnte die Ursache sein.
</details>

<details><summary><b>2. Wann Mittelwert, wann Median — und woran erkennst du es im Histogramm?</b></summary>

Median bei schiefen Verteilungen oder Ausreißern (Einkommen, Preise, Wartezeiten), Mittelwert bei annähernd symmetrischen Verteilungen (dann sind beide fast gleich — der Mittelwert ist statistisch effizienter und rechnet sich besser weiter). Im Histogramm: langer Schwanz auf einer Seite ⇒ schief ⇒ Median. Praktischer Schnellcheck: Weichen Mittel und Median deutlich voneinander ab, ist die Verteilung schief oder ausreißerbelastet.
</details>

<details><summary><b>3. In einer Spalte „Einkommen" stehen einige Werte von −999. Was ist das vermutlich, und was tust du?</b></summary>

Vermutlich ein **Sondercode für „fehlend/keine Angabe"** (in der Datendokumentation nachprüfen!). Behandlung: in `NaN` umwandeln (`df["einkommen"].replace(-999, np.nan)`), dann wie fehlende Werte behandeln — und prüfen, ob das Fehlen systematisch ist. Auf keinen Fall drinlassen: −999 würde Mittelwert, Varianz und Korrelationen massiv verzerren.
</details>

<details><summary><b>4. Warum kann eine Variable mit Pearson-r ≈ 0 trotzdem stark mit einer anderen zusammenhängen?</b></summary>

Pearson misst nur **lineare** Zusammenhänge. Ein U-förmiger Zusammenhang (z. B. $y = x^2$ um 0) hat $r \approx 0$: Für kleine $x$ fällt $y$, für große steigt es — die linearen Anteile heben sich auf. Deshalb: Scatterplot anschauen; für monotone nichtlineare Zusammenhänge Spearman verwenden.
</details>

<details><summary><b>5. Eine Studie findet: Kaffeetrinker haben mehr Herzinfarkte. Nenne einen plausiblen Confounder und wie man Kausalität prüfen könnte.</b></summary>

Z. B. **Rauchen**: Raucher trinken (statistisch) mehr Kaffee *und* haben mehr Infarkte — der Kaffee kann unschuldig sein. Prüfen: nach Confoundern stratifizieren/adjustieren (Vergleich innerhalb der Raucher und innerhalb der Nichtraucher) oder — Goldstandard — ein **randomisiertes Experiment**, das die Kaffeezuteilung vom Lebensstil entkoppelt (hier praktisch schwierig, daher Beobachtungsstudien mit sorgfältiger Adjustierung).
</details>

<details><summary><b>6. Was genau sagt ein p-Wert von 0,03 — und was sagt er NICHT?</b></summary>

Er sagt: *Wenn* die Nullhypothese wahr wäre, würde man ein mindestens so extremes Ergebnis in 3 % der Fälle sehen. Er sagt **nicht**: „$H_0$ ist zu 3 % wahr", nicht „der Effekt ist mit 97 % Wahrscheinlichkeit echt", und nicht „der Effekt ist groß/wichtig". Und wenn 20 Tests gemacht wurden, ist ein einzelnes $p = 0{,}03$ wenig wert (multiple testing).
</details>

<details><summary><b>7. Medikament A hat in beiden Krankenhäusern eine höhere Erfolgsquote als B, insgesamt aber eine niedrigere. Wie kann das sein?</b></summary>

**Simpson's Paradox**: A wurde überwiegend im Krankenhaus mit den schweren Fällen eingesetzt (niedrige Grundquote), B überwiegend bei leichten Fällen. Die Fallschwere ist ein Confounder; aggregiert gewinnt B nur, weil es die leichteren Patienten hatte. Inhaltlich relevant ist hier der Vergleich *innerhalb* vergleichbarer Fallgruppen — also A.
</details>

<details><summary><b>8. Warum ist „wir haben die Ausreißer entfernt" ohne weitere Angaben ein Warnsignal?</b></summary>

Weil Ausreißer drei völlig verschiedene Dinge sein können (Messfehler, echte Extremwerte, Sondercodes) und ihre Entfernung Ergebnisse dramatisch verändern kann. Ohne Angabe von Kriterium (z. B. IQR-Regel), Anzahl und Begründung ist die Analyse weder nachvollziehbar noch reproduzierbar — und möglicherweise wurden unbequeme echte Datenpunkte „wegbereinigt".
</details>

<details><summary><b>9. Dein Kollege hat in einem Kundendatensatz 47 Variablen gegeneinander korreliert und 3 „hochsignifikante" Zusammenhänge gefunden. Einschätzung?</b></summary>

47 Variablen ergeben $\binom{47}{2} = 1081$ Paare — bei $\alpha = 0{,}05$ erwartet man ~54 „signifikante" Korrelationen rein durch Zufall. Drei Funde sind also eher *weniger* als der Zufall liefert. Vorgehen: Korrektur für multiples Testen, und die Kandidaten als *Hypothesen* betrachten, die auf neuen, unabhängigen Daten bestätigt werden müssen.
</details>

<details><summary><b>10. Was bedeutet „tidy data", und warum ist `df.groupby` auf untidy Daten oft unmöglich?</b></summary>

Tidy: jede Variable eine Spalte, jede Beobachtung eine Zeile. Liegen z. B. Jahre als Spalten vor (wide), ist „Jahr" keine Variable, nach der man gruppieren oder filtern könnte — `groupby("jahr")` geht erst nach `melt` (wide → long). Tidy-Struktur ist die Voraussetzung dafür, dass die Standardwerkzeuge (groupby, merge, Plot-Bibliotheken) greifen.
</details>

---

## 6. Literatur & Quellen

**Lehrbücher**

- **Wickham, Çetinkaya-Rundel & Grolemund — „R for Data Science", 2. Aufl.** — kostenlos online (r4ds.hadley.nz). Ja, R statt Python — aber die Kapitel zu EDA, Tidy Data und Datenqualität sind sprachunabhängig das Beste, was es gibt. *(einsteigerfreundlich, kostenlos)*
- **McKinney — „Python for Data Analysis", 3. Aufl.** — vom pandas-Erfinder; kostenlos online: https://wesmckinney.com/book/ *(einsteigerfreundlich bis vertiefend, kostenlos)*
- **Freedman, Pisani & Purves — „Statistics"** — der Klassiker für Statistik-Intuition ohne Formelwüste. *(einsteigerfreundlich)*
- **Spiegelhalter — „The Art of Statistics"** (dt.: „Die Kunst der Statistik") — Statistikdenken anhand echter Fälle. *(sehr einsteigerfreundlich)*

**Onlinekurse & Dokumentation (kostenlos)**

- **pandas-Dokumentation, „10 minutes to pandas"** + User Guide: https://pandas.pydata.org/docs/ — Referenz, die du ohnehin ständig offen haben wirst
- **Kaggle Learn**: Kurse „Pandas" und „Data Visualization" — kurz, interaktiv, im Browser *(einsteigerfreundlich)*
- **Our World in Data** (ourworldindata.org): hervorragende Beispiele für saubere Datenvisualisierung und offene Datensätze zum Üben

**Interaktiv / Blogposts (kostenlos)**

- *Seeing Theory* (Brown University): https://seeing-theory.brown.edu — interaktive Visualisierung von Wahrscheinlichkeit, KIs und Regression *(einsteigerfreundlich, wunderschön)*
- Autodesk Research: *Same Stats, Different Graphs* (der „Datasaurus") — Anscombe auf Steroiden
- Tyler Vigen: *Spurious Correlations* — absurde Scheinkorrelationen als Impfung gegen Korrelations-Leichtgläubigkeit

**Vertiefend**

- Bickel, Hammel & O'Connell (1975): *Sex Bias in Graduate Admissions: Data from Berkeley* (Science) — das Original zum Simpson-Paradox-Fall
- Wasserstein & Lazar (2016): *The ASA Statement on p-Values* — die offizielle Klarstellung, was p-Werte (nicht) bedeuten

---

**Nächster Schritt:** `projects/01-basic/` (pandas-Grundlagen an echten Daten) → `projects/02-medium/` (einen schmutzigen Datensatz retten) → `projects/03-final/` (komplette EDA an einem realen Datensatz).
