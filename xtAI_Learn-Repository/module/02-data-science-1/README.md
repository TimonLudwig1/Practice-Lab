# Modul 02 — Data Science 1

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

**Nächster Schritt:** `projekte/01-basic/` (pandas-Grundlagen an echten Daten) → `projekte/02-medium/` (einen schmutzigen Datensatz retten) → `projekte/03-final/` (komplette EDA an einem realen Datensatz).
