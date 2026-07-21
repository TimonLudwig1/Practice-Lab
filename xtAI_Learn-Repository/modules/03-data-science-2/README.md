# Modul 03 — Data Science 2

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
