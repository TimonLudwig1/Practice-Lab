# Statistik-Intuition durch Projekte — Der komplette Übungskatalog

> **Ziel:** Statistik nicht auswendig können, sondern *fühlen*. Nach diesem Katalog sollst du jedes Konzept aus drei Richtungen beherrschen: (1) Simulation ("Ich kann es nachbauen"), (2) Anwendung auf echte/realistische Daten ("Ich erkenne, wann ich es brauche"), (3) Erklärung ("Ich kann es einem Fünfjährigen und einem Professor erklären").

## Wie du diesen Katalog benutzt

- Die Projekte sind in **10 Phasen** organisiert, die aufeinander aufbauen. Innerhalb einer Phase kannst du die Reihenfolge frei wählen.
- Jedes Projekt hat: **Schwierigkeit** (⭐ bis ⭐⭐⭐⭐⭐), **Kernkonzepte**, eine **Aufgabenbeschreibung**, eine **Datenquelle** (generiert oder real) und **Stretch Goals** für mehr Tiefe.
- **Regel Nr. 1:** Erst selbst versuchen, dann Lösung anschauen. Die `CLAUDE.md` ist so geschrieben, dass Claude dich zuerst arbeiten lässt.
- **Regel Nr. 2:** Jedes Projekt endet mit einer schriftlichen Interpretation in eigenen Worten. Code ohne Interpretation zählt nicht als abgeschlossen.
- **Regel Nr. 3:** Simulation schlägt Formel. Wenn du ein Ergebnis per Simulation reproduzieren kannst, hast du es verstanden.

**Empfohlener Stack:** Python, NumPy, pandas, SciPy, statsmodels, matplotlib/seaborn. Später: scikit-learn, PyMC (Bayes), lifelines (Survival).

---

## Phase 1 — Deskriptive Statistik & EDA (Das Fundament)

### 1.1 Der Lügendetektor für Mittelwerte ⭐
- **Konzepte:** Mittelwert, Median, Modus, Schiefe, Ausreißer
- **Aufgabe:** Generiere drei Gehaltsdatensätze (symmetrisch, rechtsschief mit Superreichen, bimodal mit zwei Berufsgruppen). Berechne für alle drei Mittelwert und Median. Schreibe für jeden Fall eine "Schlagzeile", die mit dem Mittelwert lügt, und eine, die mit dem Median lügt. Visualisiere, *warum* die Kennzahlen auseinanderlaufen.
- **Daten:** Generiert (Lognormal-, Normal-, Mischverteilungen)
- **Stretch:** Baue eine Funktion, die automatisch erkennt, ob Mittelwert oder Median die "ehrlichere" Kennzahl ist, und begründe deine Heuristik.

### 1.2 Streuung zum Anfassen ⭐
- **Konzepte:** Varianz, Standardabweichung, IQR, MAD, Spannweite
- **Aufgabe:** Zwei Kaffeemaschinen füllen Tassen: gleiche mittlere Füllmenge, unterschiedliche Streuung. Simuliere 10.000 Tassen pro Maschine. Berechne alle Streuungsmaße von Hand (eigene Funktionen!) und vergleiche mit NumPy. Beantworte: Welche Maschine würdest du kaufen und ab welcher Streuung läuft eine Tasse über (>250ml)?
- **Daten:** Generiert
- **Stretch:** Füge 5 Ausreißer hinzu und zeige, welche Streuungsmaße robust bleiben und welche explodieren.

### 1.3 Anscombe & der Datasaurus ⭐⭐
- **Konzepte:** Grenzen von Kennzahlen, Bedeutung von Visualisierung
- **Aufgabe:** Lade das Anscombe-Quartett und das Datasaurus-Dozen-Dataset. Zeige, dass alle Datensätze nahezu identische Mittelwerte, Varianzen und Korrelationen haben — aber völlig unterschiedlich aussehen. Schreibe ein Fazit: Wann darfst du Kennzahlen trauen?
- **Daten:** Real (seaborn built-in / Datasaurus CSV, alternativ generierbar)
- **Stretch:** Erzeuge selbst per simulierter Abkühlung (simulated annealing) einen Datensatz mit vorgegebenem Mittelwert/Korrelation, der ein Smiley formt.

### 1.4 Vollständige EDA eines schmutzigen Datensatzes ⭐⭐
- **Konzepte:** EDA-Workflow, fehlende Werte, Datentypen, univariate & bivariate Analyse
- **Aufgabe:** Nimm einen realistischen "schmutzigen" Datensatz (z.B. generierte E-Commerce-Bestellungen mit fehlenden Werten, Duplikaten, unsinnigen Einträgen wie negativem Alter). Führe eine komplette EDA durch: Datenqualität prüfen, jede Variable einzeln verstehen, Beziehungen erkunden, 5 Hypothesen für spätere Tests formulieren.
- **Daten:** Generiert (mit absichtlich eingebauten Fehlern) oder Titanic/Kaggle
- **Stretch:** Schreibe eine wiederverwendbare `eda_report()`-Funktion, die für jeden DataFrame einen Qualitätsbericht erstellt.

### 1.5 Quantile, Perzentile & Boxplots von Hand ⭐⭐
- **Konzepte:** Quantile, Perzentile, Boxplot-Anatomie, Ausreißer-Definition
- **Aufgabe:** Implementiere Quantilberechnung selbst (inkl. Interpolation) und baue einen Boxplot "von Hand" mit matplotlib-Primitiven (Linien, Rechtecke) — ohne `plt.boxplot`. Erkläre die 1,5×IQR-Regel: Wie viel Prozent einer Normalverteilung landen zufällig außerhalb der Whisker?
- **Daten:** Generiert
- **Stretch:** Vergleiche die 9 verschiedenen Quantil-Interpolationsmethoden von NumPy — wann macht der Unterschied praktisch etwas aus?

### 1.6 Das z-Score-Portal ⭐⭐
- **Konzepte:** Standardisierung, z-Scores, Vergleichbarkeit
- **Aufgabe:** Ein Schüler hat 85/100 in Mathe (Klassenschnitt 70, SD 10) und 60/80 in Physik (Schnitt 40, SD 12). In welchem Fach war er relativ besser? Baue ein kleines Tool, das beliebige Leistungen über Fächer/Skalen hinweg vergleichbar macht. Simuliere eine ganze Schule und zeige die Verteilung der z-Scores.
- **Daten:** Generiert
- **Stretch:** Wann versagt der z-Score-Vergleich? (Hinweis: stark schiefe Verteilungen). Zeige es mit einem Gegenbeispiel.

---

## Phase 2 — Wahrscheinlichkeit & Simulation (Denken in Zufall)

### 2.1 Monte-Carlo-Grundausbildung: π schätzen ⭐
- **Konzepte:** Monte-Carlo-Simulation, Gesetz der großen Zahlen, Konvergenz
- **Aufgabe:** Schätze π durch zufällige Punkte im Einheitsquadrat. Plotte den Schätzfehler gegen die Anzahl Punkte (log-log). Beantworte: Wie viele Punkte brauchst du für 2, 3, 4 korrekte Nachkommastellen? Was sagt dir die Steigung der Fehlerkurve?
- **Daten:** Generiert
- **Stretch:** Zeige empirisch, dass der Fehler mit 1/√n fällt — die wichtigste Zahl der gesamten Statistik.

### 2.2 Das Geburtstagsparadoxon & seine Verwandten ⭐⭐
- **Konzepte:** Kombinatorik, Gegenwahrscheinlichkeit, Intuition vs. Rechnung
- **Aufgabe:** Simuliere: Ab wie vielen Personen ist die Wahrscheinlichkeit >50%, dass zwei am selben Tag Geburtstag haben? Vergleiche Simulation mit exakter Rechnung. Erweitere: Wie viele Personen, bis jemand *deinen* Geburtstag teilt? Warum ist das eine völlig andere Zahl?
- **Daten:** Generiert
- **Stretch:** Hash-Kollisionen: Übertrage das Paradoxon auf einen 32-Bit-Hash. Ab wie vielen Einträgen erwartest du eine Kollision? (Relevanz für Data Engineering!)

### 2.3 Monty Hall — beweise es dir selbst ⭐⭐
- **Konzepte:** Bedingte Wahrscheinlichkeit, Simulation als Beweis
- **Aufgabe:** Simuliere das Monty-Hall-Problem 100.000-mal für beide Strategien (wechseln/bleiben). Erweitere dann auf 100 Türen. Schreibe eine Erklärung, die deine Großmutter überzeugen würde.
- **Daten:** Generiert
- **Stretch:** Variante "Monty Fall": Der Moderator öffnet eine *zufällige* Tür (die zufällig eine Ziege zeigt). Ändert sich die Antwort? Warum ist das der eigentliche Kern des Problems?

### 2.4 Bayes im Alltag: Der Medizintest ⭐⭐
- **Konzepte:** Satz von Bayes, Basisraten, Sensitivität/Spezifität, PPV
- **Aufgabe:** Eine Krankheit betrifft 1 von 1000 Menschen. Ein Test hat 99% Sensitivität und 95% Spezifität. Du wirst positiv getestet — wie wahrscheinlich bist du krank? Rechne per Bayes-Formel UND per Simulation von 1 Mio. Menschen UND per "natürlicher Häufigkeiten"-Baumdiagramm. Baue dann einen interaktiven Rechner (Funktion mit Parametern).
- **Daten:** Generiert
- **Stretch:** Was passiert bei einem *zweiten* positiven Test? Verkette Bayes-Updates und zeige, wie Evidenz akkumuliert.

### 2.5 Zufallsvariablen-Zoo: Erwartungswert & Varianz per Simulation ⭐⭐
- **Konzepte:** Erwartungswert, Varianz, Linearität des Erwartungswerts
- **Aufgabe:** Simuliere Glücksspiele: Würfelwurf mit Auszahlung, Roulette (verschiedene Wetten), ein Rubbellos. Berechne für jedes Spiel Erwartungswert und Varianz analytisch und per Simulation. Zeige die Linearität des Erwartungswerts an einem Beispiel, wo sie kontraintuitiv wirkt (Summe abhängiger Variablen!).
- **Daten:** Generiert
- **Stretch:** Kelly-Kriterium: Bei einem vorteilhaften Spiel — welchen Anteil deines Kapitals solltest du setzen? Simuliere verschiedene Strategien über 1000 Runden und zeige, warum "alles setzen" trotz positivem Erwartungswert ruiniert.

### 2.6 Der Gambler's Ruin & Random Walks ⭐⭐⭐
- **Konzepte:** Random Walk, Absorptionswahrscheinlichkeit, Pfadabhängigkeit
- **Aufgabe:** Ein Spieler startet mit 50€, setzt je 1€ auf Kopf/Zahl (fair und unfair p=0.49). Simuliere: Wahrscheinlichkeit, 100€ zu erreichen vs. pleite zu gehen. Wie lange dauern Spiele im Mittel? Visualisiere 100 Pfade. Verbinde mit Aktienkursen: Warum sehen Random Walks aus wie Charts mit "Trends"?
- **Daten:** Generiert
- **Stretch:** Zeige das Arkussinus-Gesetz: In einem fairen Spiel ist es paradox wahrscheinlich, dass einer der Spieler fast die ganze Zeit vorne liegt.

### 2.7 Simpson's Paradox selbst bauen ⭐⭐⭐
- **Konzepte:** Confounding, Aggregation, Simpson's Paradox
- **Aufgabe:** Konstruiere einen Datensatz (z.B. Uni-Zulassungen nach Geschlecht und Fachbereich wie Berkeley 1973), bei dem der Zusammenhang auf Gesamtebene das Gegenteil der Ebene jeder Untergruppe zeigt. Visualisiere beide Ebenen. Formuliere die Regel: Wann darf man aggregieren, wann nicht?
- **Daten:** Generiert (nach realem Vorbild Berkeley)
- **Stretch:** Baue einen "Simpson-Generator": eine Funktion, die zu gegebenen Gruppen-Effekten Daten erzeugt, die aggregiert das Vorzeichen wechseln.

---

## Phase 3 — Verteilungen (Die Sprache des Zufalls)

### 3.1 Verteilungs-Bestiarium ⭐⭐
- **Konzepte:** Bernoulli, Binomial, Poisson, Geometrisch, Uniform, Normal, Exponential, Lognormal
- **Aufgabe:** Erstelle für jede Verteilung ein "Steckbrief-Notebook": PDF/PMF & CDF plotten, Parameter variieren (Slider-artig als Plot-Grid), 3 reale Phänomene benennen, die so verteilt sind, und je eines davon simulieren (z.B. Poisson: Tore pro Fußballspiel; Exponential: Wartezeit an der Kasse; Lognormal: Einkommen).
- **Daten:** Generiert + optional real (z.B. Bundesliga-Tore)
- **Stretch:** Baue ein Quiz-Skript: Es zeigt dir ein Histogramm einer zufälligen Verteilung, du rätst welche — dein Trainingsspiel für Verteilungsintuition.

### 3.2 Poisson in der echten Welt: Tore, Tickets, Todesfälle durch Pferdetritte ⭐⭐
- **Konzepte:** Poisson-Verteilung, Poisson-Prozess, Rate λ
- **Aufgabe:** Simuliere Support-Tickets, die mit λ=4/Stunde eintreffen. Beantworte per Simulation und Formel: P(0 Tickets in einer Stunde)? P(>8)? Wie viele Mitarbeiter brauchst du, damit in 95% der Stunden alle Tickets sofort bearbeitet werden (je Mitarbeiter 1 Ticket/15min)? Prüfe dann echte Daten (z.B. Bundesliga-Tore pro Spiel) auf Poisson-Verteilung.
- **Daten:** Generiert + real (Fußballtore)
- **Stretch:** Zeige die Verbindung Poisson ↔ Exponential: Zwischenankunftszeiten eines Poisson-Prozesses sind exponentialverteilt. Simuliere es in beide Richtungen.

### 3.3 Die Normalverteilung entzaubern ⭐⭐
- **Konzepte:** Normalverteilung, 68-95-99.7-Regel, warum sie überall auftaucht
- **Aufgabe:** Simuliere Körpergrößen (μ=175, σ=7). Beantworte: Wie viel Prozent sind über 190cm? Wie groß muss eine Tür sein, damit 99,9% durchpassen? Verifiziere die 68-95-99.7-Regel per Simulation. Zeige dann an einem Gegenbeispiel (Einkommen), dass NICHT alles normalverteilt ist und was schiefgeht, wenn man es annimmt.
- **Daten:** Generiert
- **Stretch:** QQ-Plots von Hand implementieren und damit 5 Datensätze auf Normalität prüfen.

### 3.4 Fat Tails: Warum die Finanzwelt nicht normal ist ⭐⭐⭐
- **Konzepte:** Kurtosis, heavy tails, t-Verteilung, Extremereignisse
- **Aufgabe:** Lade reale Aktienrenditen (oder simuliere t-verteilte). Fitte eine Normalverteilung und zähle, wie oft "6-Sigma-Ereignisse" tatsächlich vorkommen vs. wie oft die Normalverteilung sie vorhersagt (alle paar Millionen Jahre!). Vergleiche Fit von Normal- vs. t-Verteilung per QQ-Plot.
- **Daten:** Real (yfinance/CSV) oder generiert
- **Stretch:** Simuliere ein Portfolio-Risikomodell einmal mit Normal-, einmal mit t-Annahme und vergleiche den Value-at-Risk. Diskutiere: Was hat das mit 2008 zu tun?

### 3.5 Maximum-Likelihood von Hand ⭐⭐⭐
- **Konzepte:** Likelihood, Log-Likelihood, MLE, numerische Optimierung
- **Aufgabe:** Generiere Daten aus einer Exponentialverteilung mit bekanntem λ. "Vergiss" λ und schätze es: (a) plotte die Log-Likelihood-Kurve über verschiedene λ-Werte, (b) finde das Maximum numerisch mit scipy.optimize, (c) leite die geschlossene Lösung her und vergleiche. Wiederhole für Normal (μ und σ gleichzeitig, Likelihood-Landschaft als Heatmap).
- **Daten:** Generiert
- **Stretch:** Fitte eine Mischung aus zwei Normalverteilungen per MLE (Vorstufe zum EM-Algorithmus) an bimodale Daten.

### 3.6 Ordnungsstatistiken & Extremwerte ⭐⭐⭐
- **Konzepte:** Minimum/Maximum-Verteilungen, Extremwertstatistik
- **Aufgabe:** Ein Deich muss dem höchsten Pegel von 100 Jahren standhalten. Simuliere jährliche Maximalpegel und untersuche die Verteilung des 100-Jahres-Maximums. Wie hoch muss der Deich für 99% Sicherheit sein? Zeige, warum der Mittelwert hier komplett irrelevant ist.
- **Daten:** Generiert
- **Stretch:** Vergleiche empirisch mit der Gumbel-Verteilung — die Extremwerttheorie hat ihr eigenes "zentrales Grenzwerttheorem".

---

## Phase 4 — Stichproben & der Zentrale Grenzwertsatz (Das Herz der Inferenz)

### 4.1 Der ZGS zum Anfassen ⭐⭐
- **Konzepte:** Stichprobenverteilung, Zentraler Grenzwertsatz, Standardfehler
- **Aufgabe:** Nimm drei extrem un-normale Populationen (Uniform, Exponential, bimodal). Ziehe je 10.000 Stichproben der Größen n=2, 5, 30, 100 und plotte die Verteilung der Stichprobenmittelwerte als 3×4-Grid. Beobachte, wie aus jedem Chaos eine Glocke wird. Verifiziere: SE = σ/√n.
- **Daten:** Generiert
- **Stretch:** Finde eine Verteilung, bei der der ZGS versagt oder quälend langsam ist (Cauchy!). Simuliere und staune.

### 4.2 Standardfehler ≠ Standardabweichung ⭐⭐
- **Konzepte:** SE vs. SD, der häufigste Anfängerfehler
- **Aufgabe:** Simuliere eine Population und ziehe Stichproben. Erstelle eine Visualisierung, die glasklar trennt: SD beschreibt die Streuung der *Daten*, SE die Unsicherheit des *Mittelwert-Schätzers*. Zeige: SD bleibt bei wachsendem n konstant, SE schrumpft. Schreibe die Merkregel in eigenen Worten.
- **Daten:** Generiert
- **Stretch:** Finde 3 echte Paper-Abstracts oder Artikel, in denen "±"-Angaben mehrdeutig sind — SD oder SE? Warum ist der Unterschied manchmal Faktor 10?

### 4.3 Stichprobenziehung: Wie man sich selbst belügt ⭐⭐⭐
- **Konzepte:** Sampling-Methoden, Selection Bias, Survivorship Bias
- **Aufgabe:** Erstelle eine synthetische Stadt (100.000 Einwohner mit Alter, Einkommen, Smartphone-Besitz). Simuliere verschiedene Erhebungsmethoden: echte Zufallsstichprobe, Telefonumfrage (erreicht Ältere besser), Online-Umfrage (erreicht Jüngere), Straßenumfrage in der Innenstadt. Vergleiche alle Schätzungen mit der Wahrheit. Quantifiziere den Bias jeder Methode.
- **Daten:** Generiert
- **Stretch:** Implementiere Stratified Sampling und Post-Stratification-Gewichtung, um die verzerrte Online-Umfrage zu "reparieren".

### 4.4 Survivorship Bias: Die Bomber des 2. Weltkriegs ⭐⭐
- **Konzepte:** Survivorship Bias, fehlende Daten sind Information
- **Aufgabe:** Simuliere das berühmte Wald-Problem: Bomber werden zufällig getroffen, aber nur Treffer an unkritischen Stellen lassen sie zurückkehren. Analysiere nur die zurückgekehrten Flugzeuge — wo häufen sich die Einschusslöcher? Zeige, dass die "logische" Panzerungsempfehlung (dort wo Löcher sind) genau falsch ist.
- **Daten:** Generiert
- **Stretch:** Übertrage auf Fonds-Performance: Simuliere 1000 Fonds mit reinem Zufall, lösche die schlechtesten 50% nach 5 Jahren und zeige, dass die "Überlebenden" scheinbar Können beweisen.

---

## Phase 5 — Konfidenzintervalle & Schätzung

### 5.1 Was ein 95%-KI wirklich bedeutet ⭐⭐
- **Konzepte:** Konfidenzintervall, Überdeckungswahrscheinlichkeit, Fehlinterpretationen
- **Aufgabe:** Simuliere 100 Studien (je n=50 aus bekannter Population), berechne für jede das 95%-KI des Mittelwerts und plotte alle 100 Intervalle vertikal mit der wahren Linie. Zähle, wie viele die Wahrheit verfehlen (~5). Schreibe dann: 3 korrekte und 3 falsche Interpretationen eines KIs — und warum die falschen falsch sind.
- **Daten:** Generiert
- **Stretch:** Zeige empirisch, dass die Überdeckung bei kleinem n und schiefer Population unter 95% fällt, und dass die t-Verteilung (statt z) das Problem für Normal-Daten löst.

### 5.2 Konfidenzintervalle für alles: das Kochbuch ⭐⭐⭐
- **Konzepte:** KI für Mittelwert, Anteil, Differenz, Median
- **Aufgabe:** Baue eine kleine Bibliothek `ci.py` mit selbst implementierten KIs: Mittelwert (t-basiert), Anteil (Wald UND Wilson — vergleiche beide bei p nahe 0/1!), Differenz zweier Mittelwerte, Median (Bootstrap). Validiere jede Funktion per Coverage-Simulation: erreicht sie wirklich 95%?
- **Daten:** Generiert
- **Stretch:** Zeige den Wald-Intervall-Skandal: Bei n=50, p=0.02 kollabiert seine Coverage. Wilson rettet den Tag. Wann ist das praktisch relevant? (Seltene Events: Conversion Rates, Fehlerquoten!)

### 5.3 Wahlumfragen nachbauen ⭐⭐⭐
- **Konzepte:** Anteilsschätzung, Fehlermarge, Stichprobengröße
- **Aufgabe:** Simuliere ein Land mit wahren Parteianteilen. Ziehe Umfragen mit n=1000 und berechne die berühmte "±3% Fehlermarge" selbst her. Beantworte: Warum haben fast alle Umfragen n≈1000? Wie ändert sich die Marge bei n=100, 10.000? Warum hilft es NICHT, dass Deutschland 84 Mio. Einwohner hat (Stichprobengröße schlägt Populationsgröße)?
- **Daten:** Generiert
- **Stretch:** Zwei Parteien liegen bei 32% vs. 30% — "Kopf-an-Kopf-Rennen" oder klarer Vorsprung? Berechne das KI der *Differenz* (Achtung: nicht unabhängig!) und simuliere, wie oft die Reihenfolge in Umfragen kippt.

---

## Phase 6 — Hypothesentests (Die große Maschine der Inferenz)

### 6.1 Der Hypothesentest von Null: die Dame mit dem Tee ⭐⭐
- **Konzepte:** Nullhypothese, p-Wert, Signifikanzniveau, exakter Test
- **Aufgabe:** Fishers Original: Eine Dame behauptet, sie schmecke, ob Milch vor oder nach dem Tee eingegossen wurde. Sie bekommt 8 Tassen (4/4) und sortiert alle korrekt. Berechne per Kombinatorik UND Simulation: Wie wahrscheinlich ist das durch Raten? Ab welcher Leistung würdest du ihr glauben? Damit hast du den p-Wert erfunden, bevor du seine Formel kanntest.
- **Daten:** Generiert
- **Stretch:** Sie schafft 3 von 4 Paaren. Signifikant? Diskutiere, warum die Antwort vom Design abhängt (was zählt als "mindestens so extrem"?).

### 6.2 p-Werte per Simulation: Permutationstests ⭐⭐⭐
- **Konzepte:** Permutationstest, Nullverteilung, p-Wert als Anteil
- **Aufgabe:** Zwei Gruppen (Website A/B, Verweildauer). Implementiere einen Permutationstest komplett selbst: Labels 10.000-mal mischen, Differenz der Mittelwerte berechnen, Nullverteilung plotten, beobachtete Differenz einzeichnen, p-Wert als Fläche ablesen. Vergleiche mit dem t-Test von SciPy. Das ist DAS Projekt für p-Wert-Intuition.
- **Daten:** Generiert
- **Stretch:** Teste mit derselben Maschine andere Statistiken: Median-Differenz, Varianz-Verhältnis, 90%-Quantil-Differenz. Merke: Permutation testet alles, wofür es keine Formel braucht.

### 6.3 t-Test-Trilogie ⭐⭐⭐
- **Konzepte:** Ein-Stichproben-, Zwei-Stichproben-, gepaarter t-Test, Welch-Korrektur
- **Aufgabe:** Drei Szenarien, drei Tests: (1) Füllt die Abfüllanlage wirklich 500ml ab? (2) Unterscheiden sich zwei Düngemittel? (3) Blutdruck vorher/nachher bei denselben Patienten. Implementiere jeden t-Test von Hand (Formel!) und vergleiche mit SciPy. Zeige am Szenario 3, wie viel Power man verschenkt, wenn man gepaarte Daten ungepaart testet.
- **Daten:** Generiert
- **Stretch:** Student vs. Welch: Simuliere ungleiche Varianzen und ungleiche Gruppengrößen und zeige, wann der klassische t-Test sein α-Niveau nicht hält. (Spoiler: nimm immer Welch.)

### 6.4 Fehler 1. und 2. Art: die Fehler-Fabrik ⭐⭐⭐
- **Konzepte:** α, β, Power, Effektstärke, das Trade-off-Viereck
- **Aufgabe:** Baue eine Simulations-Fabrik: Ziehe tausende Experimente unter H0 (kein Effekt) und unter H1 (echter Effekt d=0.3). Zähle falsch-positive und falsch-negative Entscheidungen. Erstelle Heatmaps: Power als Funktion von (n, Effektstärke) und von (n, α). Beantworte: Was kostet es, α von 0.05 auf 0.01 zu senken?
- **Daten:** Generiert
- **Stretch:** Baue "Gerichtsprozess-Framing": H0 = unschuldig. Übersetze α und β in "Unschuldige verurteilt" / "Schuldige freigesprochen" und diskutiere, welches α ein Gericht vs. ein Spamfilter vs. ein Krebstest wählen sollte.

### 6.5 Power-Analyse: Wie groß muss meine Studie sein? ⭐⭐⭐
- **Konzepte:** Power, Stichprobenplanung, MDE (Minimum Detectable Effect)
- **Aufgabe:** Du planst ein A/B-Experiment: Baseline-Conversion 5%, du willst eine Verbesserung auf 5.5% mit 80% Power bei α=5% erkennen. Berechne das nötige n per Simulation (Kurve: Power vs. n) und mit statsmodels. Erstelle dann die umgekehrte Sicht: Bei fixem n=2000 pro Gruppe — welcher Effekt ist überhaupt detektierbar (MDE)?
- **Daten:** Generiert
- **Stretch:** Underpowered-Studien-Horror: Simuliere Studien mit 20% Power und zeige, dass signifikante Ergebnisse dort den wahren Effekt systematisch massiv überschätzen (Winner's Curse / Typ-M-Fehler).

### 6.6 Chi-Quadrat: Zusammenhänge in Kategorien ⭐⭐⭐
- **Konzepte:** Chi²-Unabhängigkeitstest, Anpassungstest, erwartete Häufigkeiten
- **Aufgabe:** (a) Ist ein Würfel fair? Wirf einen manipulierten Würfel 600-mal (simuliert) und teste. (b) Hängt Kaufverhalten vom Marketingkanal ab? Baue eine Kontingenztabelle, berechne erwartete Häufigkeiten und Chi²-Statistik von Hand, vergleiche mit SciPy. Visualisiere beobachtet vs. erwartet.
- **Daten:** Generiert
- **Stretch:** Zeige per Simulation, woher die Chi²-Verteilung kommt: Summe quadrierter Standardnormalverteilter. Und: Warum braucht der Test mindestens ~5 erwartete Beobachtungen pro Zelle? (Fisher's exakter Test als Rettung.)

### 6.7 ANOVA: Mehr als zwei Gruppen ⭐⭐⭐
- **Konzepte:** Einweg-ANOVA, F-Statistik, Varianzzerlegung, Post-hoc-Tests
- **Aufgabe:** Vier Lehrmethoden, vier Klassen, Testergebnisse. Warum nicht einfach 6 t-Tests? (Rechne die inflationierte Fehlerrate aus!) Implementiere die ANOVA von Hand: SS_between, SS_within, F-Statistik. Danach Tukey-HSD für die Frage "welche Gruppen genau unterscheiden sich?".
- **Daten:** Generiert
- **Stretch:** Zeige die Identität: Einweg-ANOVA mit 2 Gruppen = t-Test (F = t²). Und: ANOVA als lineare Regression mit Dummy-Variablen — dieselbe Maschine, andere Verkleidung.

### 6.8 Nichtparametrische Tests: Wenn nichts normal ist ⭐⭐⭐
- **Konzepte:** Mann-Whitney-U, Wilcoxon, Kruskal-Wallis, Ränge
- **Aufgabe:** Stark schiefe Daten (z.B. Wartezeiten mit Ausreißern). Zeige per Simulation: Der t-Test verliert Power / hält sein Niveau nicht, Mann-Whitney bleibt stabil. Implementiere den U-Test über Ränge selbst. Wichtig: Was testet Mann-Whitney *wirklich*? (Nicht den Median! Konstruiere ein Gegenbeispiel.)
- **Daten:** Generiert
- **Stretch:** Wann sind Rang-Tests schlechter? Simuliere normalverteilte Daten und quantifiziere den Power-Verlust (ca. 95% Effizienz — überraschend wenig!).

### 6.9 Multiple Testing: Der grüne Jellybean ⭐⭐⭐
- **Konzepte:** Alphafehler-Kumulierung, Bonferroni, Benjamini-Hochberg (FDR)
- **Aufgabe:** Simuliere das xkcd-Szenario: Teste 20 Jellybean-Farben gegen Akne, alle ohne echten Effekt. Wie oft findest du "Signifikanz"? Skaliere auf 10.000 Gene (Genomik-Realität): 100 haben echte Effekte. Vergleiche: keine Korrektur, Bonferroni, Benjamini-Hochberg — je in Bezug auf Entdeckungen und falsche Entdeckungen.
- **Daten:** Generiert
- **Stretch:** Baue eine Visualisierung des BH-Verfahrens (sortierte p-Werte gegen die BH-Gerade) und erkläre, warum FDR-Kontrolle bei Exploration die klügere Philosophie ist als FWER.

### 6.10 p-Hacking-Simulator ⭐⭐⭐⭐
- **Konzepte:** Forscher-Freiheitsgrade, optionales Stoppen, Garden of Forking Paths
- **Aufgabe:** Baue einen "böswilligen Forscher": Er erhebt Daten ohne echten Effekt, aber (a) schaut nach jeder 10. Beobachtung auf den p-Wert und stoppt bei p<0.05, (b) probiert 5 Outcome-Variablen, (c) testet Subgruppen (nur Männer, nur unter 30...). Quantifiziere, wie diese Tricks die reale Falsch-Positiv-Rate von 5% auf 30-60% treiben.
- **Daten:** Generiert
- **Stretch:** Implementiere die Gegenmittel: Prä-Registrierung (fixes n, fixes Outcome) und alpha-spending für legitime Zwischenblicke (Pocock/O'Brien-Fleming-Idee, vereinfacht).

---

## Phase 7 — A/B-Testing & Experimentaldesign (Data-Science-Kernhandwerk)

### 7.1 Das komplette A/B-Test-Framework ⭐⭐⭐⭐
- **Konzepte:** Randomisierung, Conversion-Tests, Praxis-Pipeline
- **Aufgabe:** Baue eine End-to-End-Pipeline als wiederverwendbares Modul: (1) Power-Analyse & Laufzeitplanung, (2) simulierter Traffic mit Zuweisung, (3) Sample Ratio Mismatch Check (Chi²), (4) Auswertung: z-Test für Anteile + KI für Lift, (5) Report mit Empfehlung. Teste die Pipeline mit Szenarien: echter Effekt, kein Effekt, kaputte Randomisierung.
- **Daten:** Generiert (realistischer E-Commerce-Funnel)
- **Stretch:** Füge Neuheitseffekt hinzu (Effekt klingt über 2 Wochen ab) und zeige, warum zu kurze Tests systematisch lügen.

### 7.2 Peeking: Warum du nicht täglich aufs Dashboard schauen darfst ⭐⭐⭐
- **Konzepte:** Optionales Stoppen, sequentielles Testen
- **Aufgabe:** Simuliere einen A/A-Test (kein Effekt) über 30 Tage. Ein ungeduldiger PM prüft täglich und stoppt bei p<0.05. Wie hoch ist die reale Falsch-Positiv-Rate? Plotte den p-Wert-Verlauf einzelner Experimente (er wandert wild!). Vergleiche mit korrektem Verhalten (nur am Ende testen).
- **Daten:** Generiert
- **Stretch:** Implementiere einen einfachen sequentiellen Test (z.B. mSPRT-Idee oder Bonferroni-korrigierte Zwischenanalysen) als legales Peeking.

### 7.3 Metriken-Weisheit: Mittelwert vs. Median vs. Winsorized ⭐⭐⭐
- **Konzepte:** Metrik-Wahl, Ausreißer-Robustheit, Varianzreduktion
- **Aufgabe:** Simuliere Umsatz-pro-User (98% kaufen nichts, 2% lognormal, seltene Whales mit 10.000€). Zeige: Ein einzelner Whale kann einen A/B-Test kippen. Vergleiche Teststrategien: roher Mittelwert, winsorized/getrimmt, Konversionsrate + Umsatz-pro-Käufer getrennt. Welche Metrik hätte bei gegebenem n die beste Power?
- **Daten:** Generiert
- **Stretch:** Implementiere CUPED (Varianzreduktion mit Prä-Experiment-Daten) und zeige, wie viel Laufzeit es spart — Industriestandard bei Microsoft/Netflix & Co.

### 7.4 Bayesianisches A/B-Testing ⭐⭐⭐⭐
- **Konzepte:** Beta-Binomial, Posterior, P(B>A), Expected Loss
- **Aufgabe:** Werte denselben A/B-Test frequentistisch und bayesianisch aus. Bayes-Seite: Beta-Prior, Posterior nach Daten, per Sampling P(B besser als A) und erwarteter Verlust bei Fehlentscheidung. Visualisiere, wie die Posteriors mit wachsendem n schärfer werden (animierbar als Plot-Serie).
- **Daten:** Generiert
- **Stretch:** Entscheide per "Expected Loss < Schwelle" statt p<0.05 und vergleiche, wie oft und wie schnell beide Frameworks in 1000 simulierten Experimenten richtig entscheiden.

### 7.5 Randomisierung & Blockdesign ⭐⭐⭐
- **Konzepte:** Warum Randomisierung wirkt, Blocking, Cluster-Randomisierung
- **Aufgabe:** Simuliere ein Experiment mit starkem Störfaktor (z.B. Gerätetyp beeinflusst Conversion massiv). Vergleiche: naive Zuteilung nach Ankunftszeit (confounded!), einfache Randomisierung, geblockte Randomisierung nach Gerät. Miss Bias und Varianz der Effektschätzung über 1000 Wiederholungen.
- **Daten:** Generiert
- **Stretch:** Cluster-Randomisierung (ganze Schulen statt Schüler): Zeige, warum naive Auswertung die Unsicherheit dramatisch unterschätzt (Intraklassen-Korrelation, Design-Effekt).

---

## Phase 8 — Korrelation, Regression & Kausalität

### 8.1 Korrelations-Werkstatt ⭐⭐
- **Konzepte:** Pearson, Spearman, Kendall, Anscombe-Lektion vertieft
- **Aufgabe:** Implementiere Pearson-Korrelation von Hand (Kovarianz / Produkt der SDs). Erzeuge Datensätze mit r = 0.9, 0.5, 0, -0.7 und trainiere dein Auge: Baue ein "Guess the Correlation"-Spiel (Plot zeigen, r raten, Feedback). Zeige dann Fälle, wo Pearson versagt (nichtlinear, Ausreißer) und Spearman rettet.
- **Daten:** Generiert
- **Stretch:** r = 0, aber perfekter Zusammenhang: konstruiere 3 solche Datensätze (Parabel, Kreis, ...). Merksatz formulieren.

### 8.2 Einfache lineare Regression: dreimal hergeleitet ⭐⭐⭐
- **Konzepte:** OLS, Kleinste Quadrate, Residuen, R²
- **Aufgabe:** Wohnungsgröße → Miete. Finde die Regressionsgerade auf drei Wegen: (1) brute force über ein Grid von (Steigung, Achsenabschnitt) mit SSE-Heatmap, (2) analytische Formel von Hand, (3) Gradient Descent selbst implementiert. Alle drei müssen dasselbe liefern. Interpretiere Koeffizienten in echten Worten und zerlege R² (erklärte/Gesamt-Varianz) per eigener Rechnung.
- **Daten:** Generiert
- **Stretch:** Zeige geometrisch, warum "Regression von y auf x" ≠ "x auf y" (zwei verschiedene Geraden!) und wann das praktisch relevant ist.

### 8.3 Regressionsdiagnostik: Der Residuen-Detektiv ⭐⭐⭐
- **Konzepte:** Residuenplots, Heteroskedastizität, Nichtlinearität, Einflusspunkte
- **Aufgabe:** Erzeuge 5 Datensätze, in denen je eine Annahme verletzt ist (Nichtlinearität, Heteroskedastizität, Autokorrelation, Ausreißer, Hebelpunkt). Fitte jeweils eine Linie und lerne, jede Krankheit im Residuenplot zu erkennen. Erstelle einen "Diagnose-Spickzettel": Muster im Plot → Krankheit → Therapie.
- **Daten:** Generiert
- **Stretch:** Cook's Distance implementieren und zeigen, wie ein einziger Punkt eine Regression kapern kann (mit/ohne-Vergleich).

### 8.4 Multiple Regression & das Confounder-Drama ⭐⭐⭐⭐
- **Konzepte:** Multiple Regression, Adjustierung, omitted variable bias
- **Aufgabe:** Simuliere: Eisverkauf und Ertrinkungsunfälle korrelieren (Confounder: Temperatur). Zeige: einfache Regression findet Scheineffekt, Aufnahme der Temperatur lässt ihn verschwinden. Baue dann das Gegenteil: einen Fall, wo naive Regression *keinen* Effekt zeigt, obwohl einer da ist (Suppression). Interpretiere "Koeffizient bei Konstanthaltung der anderen Variablen" präzise.
- **Daten:** Generiert (du kennst die Wahrheit, weil du sie gebaut hast!)
- **Stretch:** Collider Bias: Konstruiere den Fall, wo Adjustieren den Bias *erzeugt* statt entfernt (z.B. "adjustiere auf Krankenhausaufenthalt"). Regel: Nicht alles kontrollieren, was rumliegt!

### 8.5 Multikollinearität ⭐⭐⭐
- **Konzepte:** Kollinearität, VIF, instabile Koeffizienten
- **Aufgabe:** Simuliere Regression mit zwei fast identischen Prädiktoren (Größe in cm und in Inch mit Messrauschen). Zeige: Vorhersage bleibt gut, Koeffizienten werden wild und wechseln Vorzeichen zwischen Stichproben. Berechne VIF von Hand (R² der Regression jedes Prädiktors auf die anderen).
- **Daten:** Generiert
- **Stretch:** Ridge-Regression als Medizin: Zeige, wie L2-Regularisierung die Koeffizienten stabilisiert (Bias-Varianz-Deal).

### 8.6 Logistische Regression von Grund auf ⭐⭐⭐⭐
- **Konzepte:** Odds, Log-Odds, Sigmoid, MLE für Klassifikation
- **Aufgabe:** Kunden-Churn (Alter, Nutzung → kündigt ja/nein). Implementiere logistische Regression selbst: Sigmoid, Log-Likelihood, Gradientenaufstieg. Vergleiche mit sklearn. Der Kern: Interpretiere einen Koeffizienten korrekt in Odds Ratios und übersetze für einen Manager in Wahrscheinlichkeiten (an konkreten Beispielpersonen!).
- **Daten:** Generiert
- **Stretch:** Kalibrierungskurve: Sagt dein Modell "70%", passiert es dann auch in 70% der Fälle? Implementiere Reliability-Diagramm und Brier Score.

### 8.7 Regression zur Mitte: Der Fluch der Sports Illustrated ⭐⭐⭐
- **Konzepte:** Regression to the mean, Messfehler, Scheinwirkungen
- **Aufgabe:** Simuliere Schüler mit wahrem Können + Prüfungsglück (zwei Tests). Wähle die schlechtesten 10% aus Test 1 aus, gib ihnen ein wirkungsloses "Förderprogramm" und zeige: In Test 2 sind sie deutlich besser — ganz ohne Wirkung! Quantifiziere den Effekt als Funktion der Test-Retest-Korrelation.
- **Daten:** Generiert
- **Stretch:** Finde 3 reale Phänomene, die durch Regression zur Mitte erklärbar sind (Blitzer an Unfallschwerpunkten, Fluch der zweiten Saison, ...) und simuliere eines.

### 8.8 Kausal-Inferenz-Einstieg: Vier Wege zur Wahrheit ⭐⭐⭐⭐⭐
- **Konzepte:** Potential Outcomes, ATE, Confounding, DAGs (informell)
- **Aufgabe:** Erzeuge eine Welt mit bekanntem wahren Behandlungseffekt (Nachhilfe → Note), aber Selektion (bessere Schüler nehmen eher Nachhilfe). Schätze den Effekt mit vier Methoden und vergleiche mit der Wahrheit: (1) naiver Vergleich (verzerrt!), (2) Regression mit Confounder-Adjustierung, (3) Matching, (4) simuliertes RCT. Erstelle eine Tabelle Bias/Varianz je Methode.
- **Daten:** Generiert
- **Stretch:** Difference-in-Differences: Simuliere zwei Städte, eine führt eine Maßnahme ein. Schätze den Effekt aus den vier Zellwerten und zeige, welche Annahme (Parallel Trends) alles trägt — und wie man sie bricht.

### 8.9 Instrumentvariablen light: Wenn du nicht randomisieren kannst ⭐⭐⭐⭐⭐
- **Konzepte:** Endogenität, Instrumente, 2SLS-Idee
- **Aufgabe:** Simuliere den Klassiker Bildung → Einkommen mit unbeobachteter "Fähigkeit" als Confounder. Zeige, dass OLS verzerrt ist. Erzeuge ein Instrument (z.B. Entfernung zur nächsten Uni) und implementiere Two-Stage Least Squares von Hand. Zeige, dass 2SLS die Wahrheit trifft — aber mit viel größerer Varianz.
- **Daten:** Generiert
- **Stretch:** Schwaches Instrument: reduziere die Instrument-Stärke und zeige, wie 2SLS zusammenbricht.

---

## Phase 9 — Bootstrap, Resampling & Bayes (Die moderne Werkzeugkiste)

### 9.1 Bootstrap von Grund auf ⭐⭐⭐
- **Konzepte:** Resampling mit Zurücklegen, Bootstrap-KI, Standardfehler ohne Formel
- **Aufgabe:** Implementiere den Bootstrap selbst (keine Bibliothek): Schätze KIs für Mittelwert, Median, Korrelation und das 90%-Quantil aus einem schiefen Datensatz. Validiere per Coverage-Simulation gegen die Wahrheit. Die Erleuchtung: Für Median & Quantil gibt es keine einfache Formel — der Bootstrap braucht keine.
- **Daten:** Generiert
- **Stretch:** Percentile- vs. BCa-Intervall vergleichen; zeige einen Fall, wo Percentile schlecht überdeckt.

### 9.2 Bootstrap in der Praxis: Modell-Unsicherheit ⭐⭐⭐⭐
- **Konzepte:** Bootstrap für Regressionskoeffizienten, Vorhersageintervalle
- **Aufgabe:** Fitte eine Regression und bootstrappe die Koeffizienten: Plotte 500 Bootstrap-Regressionsgeraden als transparentes Band über die Daten ("Unsicherheits-Schlauch"). Vergleiche Bootstrap-KIs mit den analytischen aus statsmodels. Wo weichen sie ab (kleines n, Heteroskedastizität)?
- **Daten:** Generiert
- **Stretch:** Unterschied Konfidenzintervall (für die Gerade) vs. Vorhersageintervall (für neue Punkte) — implementiere beides und visualisiere beide Bänder.

### 9.3 Bayes-Denken mit Gittern: Der Münzwurf ⭐⭐⭐
- **Konzepte:** Prior, Likelihood, Posterior, Grid Approximation
- **Aufgabe:** Eine verdächtige Münze zeigt 7× Kopf in 10 Würfen. Berechne die Posterior über p per Grid-Approximation (kein PyMC!): Prior × Likelihood, normalisieren, fertig. Zeige die Posterior-Entwicklung nach 1, 5, 10, 100 Würfen als Plot-Serie. Vergleiche drei Priors (uniform, skeptisch um 0.5, absurd) und zeige: Daten überstimmen irgendwann jeden vernünftigen Prior.
- **Daten:** Generiert
- **Stretch:** Beta-Binomial-Konjugation: Zeige, dass dein Grid-Ergebnis exakt Beta(α+k, β+n−k) entspricht — und warum Konjugation ein Geschenk ist.

### 9.4 Bayes vs. Frequentismus: Das Duell ⭐⭐⭐⭐
- **Konzepte:** Credible vs. Confidence Interval, Interpretationsunterschiede
- **Aufgabe:** Ein Szenario, zwei Welten: Schätze eine Conversion Rate aus n=100, k=8. Frequentistisch (Wilson-KI) und bayesianisch (Beta-Posterior, Credible Interval). Simuliere dann 1000 Wiederholungen und prüfe: Was verspricht jedes Intervall und hält es das? Schreibe ein Streitgespräch (2 Absätze pro Seite) zwischen einem Frequentisten und einem Bayesianer.
- **Daten:** Generiert
- **Stretch:** Der Fall, wo sie dramatisch auseinanderlaufen: n=3, k=0 ("noch nie ist etwas schiefgegangen"). Was sagt jede Schule und wem glaubst du?

### 9.5 Naive Bayes Spam-Filter ⭐⭐⭐
- **Konzepte:** Bayes als Klassifikator, bedingte Unabhängigkeit, Laplace-Glättung
- **Aufgabe:** Baue einen Spam-Filter komplett selbst (kein sklearn): Wortwahrscheinlichkeiten aus Trainingsdaten, Log-Wahrscheinlichkeiten summieren, klassifizieren. Zeige, warum Laplace-Glättung nötig ist (ein einziges nie gesehenes Wort macht sonst alles kaputt). Werte mit Confusion Matrix aus.
- **Daten:** Generiert (Templates) oder real (SMS Spam Collection)
- **Stretch:** Verschiebe die Entscheidungsschwelle und zeichne die ROC-Kurve von Hand — Brücke zur ML-Welt.

### 9.6 MCMC verstehen: Metropolis von Hand ⭐⭐⭐⭐⭐
- **Konzepte:** MCMC, Metropolis-Algorithmus, Konvergenz, Burn-in
- **Aufgabe:** Implementiere den Metropolis-Algorithmus in ~30 Zeilen und sample damit aus einer Posterior, deren Normalisierung du nicht kennst. Visualisiere den Random Walk durch den Parameterraum, Trace-Plots, Burn-in, Effekt der Schrittweite (zu klein/zu groß/richtig). Validiere gegen die Grid-Lösung aus 9.3.
- **Daten:** Generiert
- **Stretch:** Sample eine 2D-Posterior (μ, σ gemeinsam) und plotte den Pfad der Kette über der Likelihood-Landschaft — eines der schönsten Bilder der Statistik.

---

## Phase 10 — Zeitreihen, Survival & Capstones

### 10.1 Zeitreihen-Anatomie ⭐⭐⭐
- **Konzepte:** Trend, Saisonalität, Rauschen, Dekomposition, Autokorrelation
- **Aufgabe:** Baue eine synthetische Zeitreihe aus Bausteinen (Trend + Jahressaison + Wochensaison + Rauschen), dann zerlege sie wieder (statsmodels seasonal_decompose) — kommt raus, was du reingesteckt hast? Implementiere die Autokorrelationsfunktion von Hand und lies daraus die Saisonlängen ab.
- **Daten:** Generiert + real (z.B. Temperatur- oder Verkaufsdaten)
- **Stretch:** Spurious Correlation bei Zeitreihen: Zeige, dass zwei unabhängige Random Walks fast immer "hochsignifikant korrelieren" — warum Zeitreihenregression eigene Regeln hat.

### 10.2 Forecasting-Basics ehrlich evaluiert ⭐⭐⭐⭐
- **Konzepte:** Baselines, gleitender Durchschnitt, exponentielle Glättung, Backtesting
- **Aufgabe:** Prognostiziere eine Verkaufszeitreihe mit: naive Prognose (letzter Wert), saisonale naive Prognose, Moving Average, exponentieller Glättung. Entscheidend: korrekte zeitliche Validierung (Rolling-Origin, niemals zufälliger Split!). Vergleiche MAE/MAPE. Lektion: Wie oft schlägt die dumme saisonale Baseline die schlauen Methoden?
- **Daten:** Generiert oder real
- **Stretch:** Prognoseintervalle per Residuen-Bootstrap — eine Punktprognose ohne Unsicherheit ist wertlos.

### 10.3 Survival Analysis: Wann kündigen Kunden? ⭐⭐⭐⭐
- **Konzepte:** Zensierung, Kaplan-Meier, Hazard
- **Aufgabe:** Simuliere Kunden-Abos: manche kündigen, viele sind noch aktiv (zensiert!). Zeige zuerst, wie naive Analysen (nur Gekündigte betrachten / Zensierte ignorieren) systematisch lügen. Implementiere Kaplan-Meier von Hand und vergleiche zwei Kundengruppen (Log-Rank-Test via lifelines).
- **Daten:** Generiert
- **Stretch:** Cox-Regression: Welche Kundeneigenschaften treiben das Kündigungsrisiko? Interpretiere Hazard Ratios sauber.

### 10.4 Fehlende Werte: Die MCAR/MAR/MNAR-Trilogie ⭐⭐⭐⭐
- **Konzepte:** Missingness-Mechanismen, Imputation, Bias durch Löschen
- **Aufgabe:** Erzeuge einen vollständigen Datensatz (die Wahrheit), dann lösche Werte auf drei Arten: völlig zufällig (MCAR), abhängig von beobachteten Variablen (MAR: Ältere verschweigen Einkommen), abhängig vom Wert selbst (MNAR: Hohe Einkommen verschweigen Einkommen). Vergleiche für jede Welt: Complete-Case-Analyse, Mittelwert-Imputation, Regressions-Imputation — jeweils gegen die bekannte Wahrheit.
- **Daten:** Generiert
- **Stretch:** Multiple Imputation (Grundidee mit m=5 Imputationen) und warum Mittelwert-Imputation die Varianz systematisch tötet.

### 10.5 Statistik für ML: Bias-Varianz zum Anfassen ⭐⭐⭐⭐
- **Konzepte:** Bias-Varianz-Zerlegung, Overfitting, Kreuzvalidierung
- **Aufgabe:** Wahre Funktion sin(x) + Rauschen. Fitte Polynome vom Grad 1 bis 15 auf je 100 verschiedene Trainingssets. Visualisiere pro Grad: alle 100 Fits übereinander (Varianz sichtbar!) und den mittleren Fit vs. Wahrheit (Bias sichtbar!). Zerlege den Testfehler numerisch in Bias² + Varianz + Rauschen. Danach: k-fold Cross-Validation von Hand implementieren und den optimalen Grad finden.
- **Daten:** Generiert
- **Stretch:** Zeige Data Leakage konkret: Feature-Skalierung vor dem Split vs. korrekt in der CV-Pipeline — quantifiziere die zu optimistische Schätzung.

### 10.6 Unsicherheit in ML-Metriken ⭐⭐⭐⭐
- **Konzepte:** KIs für Accuracy/AUC, Modellvergleich als Hypothesentest
- **Aufgabe:** Zwei Klassifikatoren: 87% vs. 89% Accuracy auf 500 Testpunkten. Ist B wirklich besser? Bootstrap-KIs für beide Accuracies UND für die Differenz (gepaart, auf denselben Testpunkten!). McNemar-Test als Alternative. Zeige, wie groß das Testset sein müsste, damit 2 Prozentpunkte Unterschied zuverlässig erkennbar sind.
- **Daten:** Generiert
- **Stretch:** Dieselbe Analyse für AUC-Differenzen — und die unbequeme Wahrheit über die meisten Kaggle-Leaderboard-Unterschiede.

---

## Capstone-Projekte (alles zusammen)

### C1 Der komplette Analyse-Zyklus: Produktexperiment ⭐⭐⭐⭐⭐
Simuliere ein komplettes Produkt-Feature-Experiment von der Idee bis zur Entscheidung: Hypothese formulieren → Power-Analyse & Laufzeit → Datensimulation mit realistischen Effekten (Neuheitseffekt, Segment-Heterogenität, ein paar Bots) → Datenqualitätsprüfung (SRM!) → Primär- und Sekundärmetriken mit Multiple-Testing-Korrektur → Segment-Analyse (Vorsicht: post-hoc!) → schriftliche Entscheidung mit Unsicherheitsquantifizierung, als würde sie an die Geschäftsführung gehen.

### C2 Statistik-Mythbusters: Ein Paper zerlegen ⭐⭐⭐⭐⭐
Nimm eine reale populärwissenschaftliche Behauptung ("Schokolade macht schlank", "Weintrinker leben länger") und (1) simuliere, wie so ein Ergebnis durch Confounding/Multiple Testing/kleine n entstehen kann, (2) rechne nach, welche Studiengröße für den behaupteten Effekt nötig wäre, (3) schreibe eine faire Kritik: Was müsste eine Studie zeigen, damit du sie glaubst?

### C3 Die Simulations-Bibliothek ⭐⭐⭐⭐⭐
Verwandle deine besten Projekte in ein sauberes, installierbares Python-Paket `statlab`: Module für Verteilungen, Tests (inkl. Permutation & Bootstrap), Power-Analyse, A/B-Auswertung. Mit Docstrings, Tests (pytest — deine Coverage-Simulationen werden zu Unit-Tests!), README. Das ist gleichzeitig dein Portfolio-Stück und dein Data-Engineering-Training.

### C4 "Erklär's mir"-Sammlung ⭐⭐⭐⭐
Schreibe zu den 15 wichtigsten Konzepten (p-Wert, KI, Power, ZGS, Bootstrap, Bayes, Confounding, Regression zur Mitte, ...) je eine Ein-Seiten-Erklärung mit genau einer selbst erzeugten Grafik, verständlich für einen Nicht-Statistiker. Feynman-Test: Wenn du es nicht einfach erklären kannst, zurück zur Simulation.

### C5 Echte Daten, echtes Chaos ⭐⭐⭐⭐⭐
Wähle einen echten, ungeputzten Datensatz (öffentliche Verwaltungsdaten, Kaggle, eigene Scrapes) und führe eine vollständige statistische Analyse durch: Fragen formulieren → EDA → Missing-Data-Strategie → mindestens 3 formale Analysen (Tests/Regression/Bootstrap) mit Annahmen-Prüfung → ehrlicher Limitations-Abschnitt. Keine Simulation als Sicherheitsnetz — die Realität ist der Endgegner.

---

## Anhang A — Selbsttest: Bist du fertig?

Du beherrschst Statistik intuitiv, wenn du folgende Fragen ohne Nachschlagen, in eigenen Worten und mit einem Simulations-Skizzenplan beantworten kannst:

1. Warum schrumpft der Standardfehler mit √n und nicht mit n?
2. Was genau ist ein p-Wert — und was ist er *nicht*? (3 verbreitete Fehlinterpretationen nennen)
3. Ein 95%-KI enthält den wahren Wert mit 95% Wahrscheinlichkeit — richtig oder falsch, und warum ist die Frage gemein?
4. Warum ist "signifikant" nicht "wichtig" und "nicht signifikant" nicht "kein Effekt"?
5. Dein A/B-Test ist nach 3 Tagen signifikant. Warum darfst du (vermutlich) nicht stoppen?
6. Wann erzeugt das Kontrollieren einer Variable Bias, statt ihn zu entfernen?
7. Warum überschätzen signifikante Ergebnisse aus kleinen Studien den wahren Effekt?
8. Erkläre den Bootstrap so, dass klar wird, warum er funktioniert — nicht nur wie.
9. Was ist der Unterschied zwischen der Verteilung der Daten und der Verteilung eines Schätzers?
10. Ein Kollege sagt "die Daten sind normalverteilt, ich hab den Shapiro-Test gemacht, p=0.2". Was entgegnest du?

## Anhang B — Empfohlene Reihenfolge für 8 Wochen Semesterferien

| Woche | Fokus | Projekte (Kern) |
|---|---|---|
| 1 | Fundament & Simulation | 1.1–1.4, 2.1, 2.4 |
| 2 | Wahrscheinlichkeit & Verteilungen | 2.2, 2.3, 2.6, 3.1–3.3 |
| 3 | ZGS & Schätzung | 4.1–4.3, 5.1, 5.2 |
| 4 | Hypothesentests I | 6.1–6.4 |
| 5 | Hypothesentests II & Fallen | 6.5–6.10 |
| 6 | A/B-Testing | 7.1–7.4, 5.3 |
| 7 | Regression & Kausalität | 8.1–8.4, 8.6, 8.8 |
| 8 | Bootstrap, Bayes & Capstone-Start | 9.1, 9.3, 9.4, C1 oder C3 |

Restliche Projekte (Zeitreihen, Survival, MCMC, 8.9, Capstones) als Vertiefung danach — sie lohnen sich alle, aber das Fundament geht vor.
