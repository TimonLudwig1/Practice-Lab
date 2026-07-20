# Übungen

Jeder Ordner ist ein eigenständiges, offline ausführbares Mini-Projekt. Die Projekte verwenden feste Zufallsseeds, sodass Daten, Kennzahlen und Grafiken reproduzierbar sind.

## Empfohlener Workflow

1. Die Aufgaben im jeweiligen `README.md` lesen.
2. `starter.py` kopieren oder direkt ergänzen, ohne vorher die Lösung anzusehen.
3. Das bearbeitete Skript aus dem Repository-Root ausführen.
4. Numerische Ergebnisse und Grafiken in eigenen Worten interpretieren.
5. Erst danach mit `solution.py` vergleichen.

Abhängigkeiten können mit `python3 -m pip install -r requirements.txt` installiert werden. Alle Pfade in den Skripten werden relativ zur jeweiligen Python-Datei aufgelöst; die Skripte funktionieren daher unabhängig vom aktuellen Arbeitsverzeichnis.

## Abschnitt 1: Stichproben, Unsicherheit und Tests

| ID | Projekt | Kernidee |
|---|---|---|
| T01 | [Stadtmobilitäts-Stichproben](T01-random-samples/README.md) | Wiederholte Zufallsstichproben und Stichprobengröße |
| T02 | [CLT im Onlinehandel](T02-central-limit-theorem/README.md) | Stichprobenverteilung bei stark schiefen Daten |
| T03 | [Sampling Variation vs. Bias](T03-sampling-variation-vs-bias/README.md) | Zufällige Schwankung gegen systematische Auswahlverzerrung |
| T04 | [Hypothesentest für Lieferzeiten](T04-hypothesis-testing/README.md) | Teststatistik, kritischer Wert und p-Wert |
| T05 | [A/B-Test zweier Checkout-Designs](T05-difference-in-means/README.md) | Welch-Test und Konfidenzintervall für unabhängige Gruppen |
| T06 | [Gepaarte Vorher-Nachher-Daten](T06-paired-samples/README.md) | Paired Test gegen fälschlich unabhängige Analyse |

## Abschnitt 2: Zusammenhang zwischen Variablen

| ID | Projekt | Kernidee |
|---|---|---|
| T07 | [Kovarianz und Maßeinheiten](T07-covariance-and-correlation/README.md) | Zentrierte Produkte, Richtung und Einheitenabhängigkeit |
| T08 | [Korrelationskoeffizient unter Stress](T08-correlation-coefficient/README.md) | Standardisierung, Skalierungsinvarianz und typische Fallen |

## Abschnitt 3: Einfache Regression und funktionale Form

| ID | Projekt | Kernidee |
|---|---|---|
| T09 | [Produktivität und einfache OLS-Regression](T09-simple-linear-regression/README.md) | OLS-Koeffizienten und Gauss–Markov-Annahmen |
| T10 | [Mietpreise durch Residuen verstehen](T10-residuals/README.md) | Beobachtung, Fit und Residuum |
| T11 | [Schiefe Unternehmensdaten logarithmieren](T11-taking-logs/README.md) | Log-Transformation, Verteilungen und Definitionsbereich |
| T12 | [Nichtlinearität in Energieverbrauch](T12-nonlinear-relationships/README.md) | Residuenmuster und quadratische Spezifikation |
| T13 | [Vier logarithmische Regressionsmodelle](T13-logarithmic-models/README.md) | Level-Level, Log-Level, Level-Log und Log-Log |
| T14 | [Heteroskedastizität im Haushaltskonsum](T14-heteroscedasticity/README.md) | Trichterform und robuste Standardfehler |
| T15 | [Autokorrelation in täglichen Bestellungen](T15-autocorrelation/README.md) | Serielle Residuen und HAC-Standardfehler |
| T16 | [Inferenz für Regressionskoeffizienten](T16-coefficient-inference/README.md) | t-Tests und Konfidenzintervalle |
| T17 | [R² als Varianzzerlegung](T17-r-squared/README.md) | TSS, ESS, RSS und adjustiertes R² |

## Abschnitt 4: Multiple Regression

| ID | Projekt | Kernidee |
|---|---|---|
| T18 | [Wohnungsmieten mit mehreren Regressoren](T18-multiple-regression/README.md) | Ceteris-paribus-Effekte und Frisch–Waugh–Lovell |
| T19 | [Gemeinsame Hypothesen im Gehaltsmodell](T19-multiple-hypothesis-testing/README.md) | Einzeltests, lineare Restriktionen und F-Test |
| T20 | [Multikollinearität bei Gebäudeindikatoren](T20-multicollinearity/README.md) | VIF, instabile Koeffizienten und stabile Vorhersagen |
| T21 | [Omitted Variable Bias bei Bildungsrenditen](T21-omitted-variable-bias/README.md) | Ability Bias und Richtung der Verzerrung |
| T22 | [Kategoriale Variablen im Filialmodell](T22-categorical-variables/README.md) | Dummy-Codierung und Referenzkategorien |
| T23 | [Die Dummy-Variable-Trap als Rangproblem](T23-dummy-variable-trap/README.md) | Perfekte Multikollinearität und gültige Parametrisierungen |
| T24 | [Interaktionseffekte in einem Trainingsprogramm](T24-interaction-effects/README.md) | Bedingte Effekte und marginale Treatment-Effekte |

## Abschnitt 5: Binäre abhängige Variablen

| ID | Projekt | Kernidee |
|---|---|---|
| T25 | [Kreditausfälle als binäre Outcomes](T25-binary-dependent-variables/README.md) | Bernoulli-Mittelwert, bedingte Wahrscheinlichkeit und Varianz |
| T26 | [Linear Probability Model für Kursabschlüsse](T26-linear-probability-model/README.md) | Prozentpunkte, Heteroskedastizität und ungültige Vorhersagen |
| T27 | [Logit-Modell für Vertragsverlängerungen](T27-logit-model/README.md) | Log-Odds, Odds Ratios und S-förmige Wahrscheinlichkeiten |
| T28 | [Marginale Effekte bei Jobangeboten](T28-marginal-effects/README.md) | AME, MEM, Ableitungen und diskrete Änderungen |

## Abschnitt 6: Kausalität und randomisierte Experimente

| ID | Projekt | Kernidee |
|---|---|---|
| T29 | [Selbstselektion in ein Weiterbildungsprogramm](T29-causality-selection-bias/README.md) | Potenzielle Outcomes und Zerlegung des naiven Vergleichs |
| T30 | [Randomisiertes Lerncoaching](T30-randomized-experiments/README.md) | Zufallszuweisung und wiederholte Randomisierung |
| T31 | [Balance Tests in einem Onlineexperiment](T31-balance-tests/README.md) | Standardisierte Differenzen, gemeinsame Tests und Zufallstreffer |
| T32 | [Übertragbarkeit einer Gesundheitsstudie](T32-internal-external-validity/README.md) | Interne Validität, Effekt-Heterogenität und Generalisierung |
| T33 | [Treatment-Effekte eines Mentoring-RCTs](T33-treatment-effects-experiments/README.md) | Mittelwertsdifferenz, Regression, Präzision und Randomisierungsinferenz |

## Abschnitt 7: Paneldaten und Fixed Effects

| ID | Projekt | Kernidee |
|---|---|---|
| T34 | [Firmen über acht Quartale](T34-panel-data/README.md) | Panelindex, Balance sowie Within- und Between-Variation |
| T35 | [Fixed Effects im Filialpanel](T35-fixed-effects/README.md) | Zeitinvariante Filialqualität und verzerrtes Pooled OLS |
| T36 | [Individuelle Fixed Effects bei Beschäftigten](T36-individual-fixed-effects/README.md) | Personenspezifische Intercepts und Identifikation innerhalb von Personen |
| T37 | [Within-Estimator für Unternehmensinvestitionen](T37-within-estimator/README.md) | Demeaning und Gleichheit mit der Dummy-Regression |
| T38 | [First Differences bei Haushaltseinkommen](T38-first-differences/README.md) | Differenzieren, zeitinvariante Effekte und Vergleich mit Within |
| T39 | [Direkt geschätzte Krankenhaus-Fixed-Effects](T39-analyzing-fixed-effects/README.md) | Normalisierung, Unsicherheit und Ranking von Einheitseffekten |

## Abschnitt 8: Differences-in-Differences

| ID | Projekt | Kernidee |
|---|---|---|
| T40 | [DiD bei einer kommunalen Umweltzone](T40-differences-in-differences/README.md) | Doppelter Mittelwertvergleich, Regressionsinteraktion und Paralleltrends |
| T41 | [Warum ein Post-only-Vergleich täuscht](T41-did-vs-post-only/README.md) | Dauerhafte Gruppenunterschiede gegen Veränderungen über die Zeit |
| T42 | [DiD mit individuellen Fixed Effects](T42-did-individual-fixed-effects/README.md) | Haushalts-Fixed-Effects und zeitinvariante unbeobachtete Unterschiede |
| T43 | [DiD mit Zeit-Fixed-Effects](T43-did-time-fixed-effects/README.md) | Gemeinsame Monatsschocks und flexible Zeitkontrolle |
| T44 | [Two-Way Fixed Effects im Filialpanel](T44-did-two-way-fixed-effects/README.md) | Individuelle und zeitliche Fixed Effects im selben DiD-Modell |
| T45 | [Nicht-binäre Förderung und Behandlungsintensität](T45-non-binary-treatment/README.md) | Dosis, Linearität und heterogene marginale Effekte |

## Abschnitt 9: Instrumental Variables

| ID | Projekt | Kernidee |
|---|---|---|
| T46 | [Warum OLS bei Weiterbildung verzerrt ist](T46-instrumental-variables-ols-bias/README.md) | Endogenität, Instrumentenannahmen und IV als Gegenentwurf |
| T47 | [Starke und schwache First Stages](T47-first-stage/README.md) | Relevanz, partielles R², F-Statistik und instabile IV-Schätzer |
| T48 | [Reduced Form einer Jobberatung](T48-reduced-form/README.md) | Intention-to-Treat, First Stage und Zerlegung des Wald-Schätzers |
| T49 | [LATE bei freiwilliger Programmteilnahme](T49-late/README.md) | Compliance-Typen, Monotonie und lokaler Treatment-Effekt |
| T50 | [IV-Schätzung als Kovarianzquotient](T50-iv-covariance-ratio/README.md) | Äquivalenz von Kovarianzratio, Wald-Logik und 2SLS |
