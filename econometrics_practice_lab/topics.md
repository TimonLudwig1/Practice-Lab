# Applied Data Science Lab: Themenkatalog

## Zweck des Labs

Dieses Repository ist ein übungsorientiertes Econometrics- und Applied-Data-Science-Lab. Die späteren Übungen sollen nicht nur Rechenschritte abfragen, sondern durch reale oder realistisch simulierte Daten, Visualisierungen, Interpretation und kleine Modellierungsentscheidungen ökonometrische Intuition aufbauen.

Die Topic-IDs sind stabil und dienen als gemeinsame Referenz für Übungen und für [`progress.md`](progress.md). Neue Übungen sollen immer mindestens einer Topic-ID zugeordnet werden.

## 1. Stichproben, Unsicherheit und Tests

### T01 – Random Samples

Zufallsstichproben verstehen und untersuchen, wie Stichprobengröße, Unabhängigkeit und identische Verteilung die Aussagekraft empirischer Ergebnisse beeinflussen.

### T02 – Central Limit Theorem (CLT)

Durch Simulationen nachvollziehen, warum sich die Stichprobenverteilung vieler Mittelwerte mit wachsender Stichprobe einer Normalverteilung annähert.

### T03 – Sampling Variation vs. Bias

Zufällige Schwankungen zwischen Stichproben von systematischen Verzerrungen eines Schätzers unterscheiden und beide Effekte visuell diagnostizieren.

### T04 – Hypothesis Testing: Test Statistic, Critical Value und p-Value

Null- und Alternativhypothese formulieren, Teststatistiken berechnen und Entscheidungen sowohl über kritische Werte als auch über p-Werte treffen und interpretieren.

### T05 – Testing Differences in Means

Mittelwertsunterschiede zwischen zwei unabhängigen Gruppen schätzen, testen und mit einem Konfidenzintervall inhaltlich interpretieren.

### T06 – Paired Structure of Two Samples

Gepaarte Daten wie Vorher-Nachher-Messungen erkennen, korrekt auswerten und mit einer fälschlich unabhängigen Analyse vergleichen.

## 2. Zusammenhang zwischen Variablen

### T07 – Covariance and Correlation

Richtung und gemeinsame Variation zweier Variablen anhand von Kovarianz und grafischen Darstellungen verstehen, einschließlich ihrer Abhängigkeit von Maßeinheiten.

### T08 – Correlation Coefficient

Den standardisierten Korrelationskoeffizienten berechnen und interpretieren sowie seine Grenzen bei Nichtlinearität, Ausreißern und Kausalitätsaussagen erkennen.

## 3. Einfache Regression und funktionale Form

### T09 – Simple Linear Regression and Gauss–Markov Assumptions

Eine einfache lineare Regression schätzen, Steigung und Achsenabschnitt interpretieren und die Gauss–Markov-Annahmen mit ihrer Bedeutung für OLS einordnen.

### T10 – Residuals

Residuen berechnen, plotten und als Diagnosewerkzeug für Modellfehler, Ausreißer und fehlende Struktur verwenden.

### T11 – Taking Logs

Verstehen, warum Variablen logarithmiert werden, wie sich Verteilungen dadurch verändern und welche Konsequenzen das für die Interpretation hat.

### T12 – Non-linear Relationships and Residuals

Nichtlineare Zusammenhänge erkennen, die ein lineares Modell in systematischen Residuen hinterlässt, und passendere Spezifikationen vergleichen.

### T13 – Different Logarithmic Models

Level-Level-, Log-Level-, Level-Log- und Log-Log-Modelle schätzen und Koeffizienten als Einheitenänderung, prozentuale Änderung oder Elastizität interpretieren.

### T14 – Heteroscedasticity: Data-driven Visual Example

Heteroskedastizität in Daten und Residuenplots sichtbar machen, ihre Folgen für Standardfehler erklären und geeignete Inferenzmethoden kennenlernen.

### T15 – Autocorrelation: Data-driven Visual Example

Serielle Korrelation in zeitlich geordneten Daten und Residuen visualisieren, ihre Ursachen diskutieren und ihre Folgen für OLS-Inferenz verstehen.

### T16 – Significance Tests and Confidence Intervals for Regression Coefficients

Standardfehler, t-Statistiken, p-Werte und Konfidenzintervalle für Regressionskoeffizienten berechnen und statistische von praktischer Signifikanz unterscheiden.

### T17 – R-squared

Bestimmtheitsmaß und adjustiertes Bestimmtheitsmaß als Gütemaße interpretieren, miteinander vergleichen und typische Fehlinterpretationen vermeiden.

## 4. Multiple Regression

### T18 – Multiple Regression Models

Mehrere erklärende Variablen gemeinsam modellieren und Koeffizienten als ceteris-paribus-Zusammenhänge interpretieren.

### T19 – Hypothesis Testing in Multiple Regression

Einzelne und gemeinsame Hypothesen über mehrere Regressionskoeffizienten mit t- und F-Tests prüfen.

### T20 – Multicollinearity

Stark korrelierte Regressoren diagnostizieren und verstehen, wie Multikollinearität Präzision, Standardfehler und Interpretation beeinflusst.

### T21 – Omitted Variable Bias

Anhand von Daten und kausalen Überlegungen erkennen, wann eine ausgelassene Variable OLS-Koeffizienten verzerrt und in welche Richtung der Bias wirkt.

### T22 – Categorical Variables (Dummy Variables)

Kategoriale Merkmale korrekt codieren, Referenzkategorien wählen und Dummy-Koeffizienten interpretieren.

### T23 – Dummy Variable Trap

Perfekte Multikollinearität durch vollständige Dummy-Codierung erkennen, erklären und durch eine geeignete Referenzkategorie vermeiden.

### T24 – Interaction Effects

Interaktionsterme einsetzen, um gruppen- oder niveauabhängige Effekte zu modellieren, und die resultierenden marginalen Zusammenhänge korrekt darstellen.

## 5. Binäre abhängige Variablen

### T25 – Binary Dependent Variables

Die besonderen Modellierungs- und Interpretationsprobleme verstehen, die entstehen, wenn die Zielvariable nur zwei Ausprägungen besitzt.

### T26 – Linear Probability Model

Ein lineares Wahrscheinlichkeitsmodell schätzen, seine Koeffizienten interpretieren und Grenzen wie Heteroskedastizität und Vorhersagen außerhalb von null bis eins untersuchen.

### T27 – Logit Model

Ein Logit-Modell schätzen und den Zusammenhang zwischen linearem Index, Log-Odds und vorhergesagten Wahrscheinlichkeiten nachvollziehen.

### T28 – Marginal Effects

Marginale Effekte für nichtlineare Modelle berechnen, Average Marginal Effects und Marginal Effects at the Mean unterscheiden und verständlich kommunizieren.

## 6. Kausalität und randomisierte Experimente

### T29 – Causality and Selection Bias

Korrelation von Kausalität abgrenzen und mit potenziellen Outcomes verstehen, wie Selbstselektion zu systematischen Unterschieden zwischen Treatment- und Kontrollgruppe führt.

### T30 – Randomized Experiments and RCTs

Nachvollziehen, warum Randomisierung im Erwartungswert vergleichbare Gruppen erzeugt und wie ein Randomized Controlled Trial aufgebaut und analysiert wird.

### T31 – Balance Tests

Pre-Treatment-Merkmale zwischen Treatment- und Kontrollgruppe vergleichen und Balance-Tests sinnvoll nutzen, ohne sie als Beweis perfekter Randomisierung zu missverstehen.

### T32 – Internal vs. External Validity

Interne Verzerrungsquellen von Problemen der Übertragbarkeit unterscheiden und den Zielkonflikt zwischen glaubwürdiger Identifikation und Generalisierbarkeit beurteilen.

### T33 – Estimating Treatment Effects in Experiments

Average Treatment Effects und gruppenspezifische Effekte in Experimenten per Mittelwertsvergleich und Regression schätzen sowie Unsicherheit korrekt quantifizieren.

## 7. Paneldaten und Fixed Effects

### T34 – Panel Data

Die Struktur wiederholter Beobachtungen derselben Einheiten verstehen und Paneldaten von Querschnitts- und reinen Zeitreihendaten unterscheiden.

### T35 – Fixed Effects

Zeitinvariante unbeobachtete Heterogenität als Störfaktor erkennen und die Grundidee von Fixed-Effects-Modellen verstehen.

### T36 – Individual Fixed Effects

Einheitenspezifische Fixed Effects einsetzen, um zeitlich konstante Unterschiede zwischen Personen, Firmen, Ländern oder anderen Einheiten zu kontrollieren.

### T37 – Within Estimator

Den Within-Estimator durch Zentrierung innerhalb jeder Einheit herleiten, anwenden und als Identifikation aus Veränderungen innerhalb derselben Einheit interpretieren.

### T38 – First Differences

Zeitlich aufeinanderfolgende Beobachtungen differenzieren, zeitinvariante unbeobachtete Effekte eliminieren und den Ansatz mit dem Within-Estimator vergleichen.

### T39 – Directly Estimating and Analysing Fixed Effects

Fixed Effects über Dummy-Variablen direkt schätzen, einzelne Effekte extrahieren und ihre Verteilung und inhaltliche Bedeutung vorsichtig analysieren.

## 8. Differences-in-Differences

### T40 – Differences-in-Differences Model

Den DiD-Schätzer aus Gruppen- und Zeitdifferenzen aufbauen, als Interaktion in einer Regression schätzen und die Parallel-Trends-Annahme verstehen.

### T41 – DiD vs. Post-treatment Data Only

Eine vollständige DiD-Analyse mit einem reinen Post-Treatment-Vergleich kontrastieren und zeigen, welche zeitinvarianten Gruppenunterschiede Letzteren verzerren können.

### T42 – DiD with Individual Fixed Effects

DiD mit individuellen Fixed Effects schätzen und verstehen, welche unbeobachteten zeitinvarianten Unterschiede dadurch absorbiert werden.

### T43 – DiD with Time Fixed Effects

Gemeinsame Zeitschocks durch Time Fixed Effects kontrollieren und den verbleibenden Treatment-Effekt interpretieren.

### T44 – DiD with Time and Individual Fixed Effects

Ein Two-Way-Fixed-Effects-Modell formulieren, schätzen und bei geeigneter Treatment-Struktur mit dem klassischen DiD-Schätzer verbinden.

### T45 – Non-binary Treatment

Treatments mit unterschiedlicher Intensität oder Dosis modellieren und klären, welche zusätzliche funktionale Form und Identifikationsannahmen dafür benötigt werden.

## 9. Instrumental Variables

### T46 – Instrumental Variables and Why OLS May Be Biased

Endogenität durch ausgelassene Variablen, Simultanität oder Messfehler erkennen und Relevanz sowie Exogenität eines Instruments als Ausweg begründen.

### T47 – First Stage

Die Wirkung des Instruments auf den endogenen Regressor schätzen und Instrumentenrelevanz sowie schwache Instrumente empirisch beurteilen.

### T48 – Reduced Form

Den Gesamteffekt des Instruments auf das Outcome schätzen und seine Verbindung zu First Stage und IV-Schätzer verstehen.

### T49 – Local Average Treatment Effect (LATE)

Den IV-Effekt unter den passenden Annahmen als lokalen durchschnittlichen Treatment-Effekt für Compliers interpretieren und die Rolle von Always-Takers, Never-Takers und Defiers erklären.

### T50 – Estimating IV Regression and the Covariance Ratio

Die IV-Regression praktisch schätzen und im einfachen Fall die Darstellung \(\hat\beta_{IV}=\operatorname{Cov}(Z,Y)/\operatorname{Cov}(Z,X)\) mit First Stage, Reduced Form und OLS-Bias verbinden.

## Leitlinien für spätere Übungen

Jede Übung soll:

- die zugehörigen Topic-IDs explizit nennen,
- eine konkrete Lernfrage und einen kurzen Intuitionsanker enthalten,
- reale oder reproduzierbar simulierte Daten verwenden,
- aktive Aufgaben zu Datenprüfung, Visualisierung, Schätzung und Interpretation enthalten,
- typische Fehlinterpretationen oder Modellierungsfallen sichtbar machen,
- eine überprüfbare Lösung oder Musterlösung besitzen und
- nach ihrer Erstellung unmittelbar in [`progress.md`](progress.md) eingetragen werden.
