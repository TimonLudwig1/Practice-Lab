# Modul 01 — Introduction in AI

**Worum geht es?** Dieses Modul ist das Fundament des gesamten Studiengangs. Es beantwortet: Was ist Künstliche Intelligenz überhaupt? Wie formalisiert man „intelligentes Verhalten" so, dass ein Computer es ausführen kann? Du lernst die klassischen Kernideen der KI — Agenten, Problemlösen durch Suche, Spiele, Logik, Schließen unter Unsicherheit — und bekommst einen Überblick, wie das moderne maschinelle Lernen (bis hin zu großen Sprachmodellen) darauf aufbaut.

**Vorkenntnisse:** Schulmathematik (Mengen, Funktionen, ein wenig Wahrscheinlichkeit) und Python-Grundlagen (Variablen, Schleifen, Funktionen, Listen/Dictionaries). Keine KI-Vorkenntnisse nötig.

**Vorher zu machen:** Nichts — das hier ist Modul 1.

---

## Lernziele

Nach diesem Modul kannst du:

- erklären, was ein **rationaler Agent** ist und eine Aufgabe mit dem **PEAS-Schema** beschreiben,
- ein Problem als **Zustandsraum-Suche** formulieren (Zustände, Aktionen, Zieltest, Kosten),
- die wichtigsten **Suchalgorithmen** (BFS, DFS, Uniform-Cost, Greedy, A\*) anwenden und ihre Stärken/Schwächen vergleichen,
- erklären, was eine **zulässige Heuristik** ist und warum A\* damit optimal ist,
- Zwei-Personen-Spiele mit **Minimax** und **Alpha-Beta-Pruning** lösen,
- ein Problem als **Constraint Satisfaction Problem (CSP)** modellieren,
- einfache Schlüsse in **Aussagenlogik** ziehen und die Idee der Wissensrepräsentation erklären,
- mit der **Bayes-Regel** unter Unsicherheit schließen und erklären, wie ein Naive-Bayes-Klassifikator funktioniert,
- die drei Hauptarten des **maschinellen Lernens** unterscheiden und grob einordnen, wo Deep Learning und LLMs in die KI-Landschaft gehören.

---

## 1. Grundlagen (Basics)

### 1.1 Was ist KI? Vier Sichtweisen

„Künstliche Intelligenz" wird seit den 1950ern unterschiedlich definiert. Das Standardlehrbuch (Russell & Norvig) sortiert die Definitionen in vier Quadranten:

|  | **menschlich** | **rational** |
|--|--|--|
| **Denken** | kognitive Modellierung („denkt wie ein Mensch") | Logik, korrektes Schließen („denkt richtig") |
| **Handeln** | Turing-Test („verhält sich wie ein Mensch") | **rationale Agenten („handelt bestmöglich")** |

Die moderne KI-Forschung arbeitet fast durchgängig mit der vierten Sichtweise: **KI = das Bauen rationaler Agenten**. „Rational" heißt dabei nicht „allwissend", sondern: *Der Agent wählt die Aktion, die seinen erwarteten Erfolg maximiert — gegeben das, was er wahrnimmt und weiß.*

> **Intuition:** Ein Navi ist rational, wenn es die (nach seinen Karten) schnellste Route wählt. Dass es einen unangekündigten Stau nicht kennt, macht es nicht irrational — es hat mit seinem Wissen das Beste getan.

**Kurze Geschichte in fünf Etappen:**

1. **1950–1956**: Turing stellt die Frage „Can machines think?" (Turing-Test); die Dartmouth-Konferenz 1956 prägt den Begriff *Artificial Intelligence*.
2. **1956–1974**: Frühe Euphorie — Programme lösen Logikrätsel und spielen Dame. Suche und Symbolverarbeitung dominieren („Good Old-Fashioned AI", GOFAI).
3. **1974–1980er**: Erster „AI Winter" (Erwartungen enttäuscht, Förderung gestrichen), danach Boom der **Expertensysteme** (regelbasiertes Fachwissen) — und deren Ernüchterung (zweiter AI Winter).
4. **1990er–2010**: Statistische Wende: Wahrscheinlichkeit und maschinelles Lernen statt handgeschriebener Regeln. 1997 schlägt Deep Blue Kasparow im Schach.
5. **ab 2012**: **Deep Learning**-Revolution (AlexNet gewinnt ImageNet), 2016 AlphaGo, ab 2017 die Transformer-Architektur, ab ~2020 große Sprachmodelle (GPT, Claude & Co.).

Merke: Die klassischen Inhalte dieses Moduls (Suche, Logik, Wahrscheinlichkeit) sind nicht „veraltet" — sie sind das begriffliche Skelett, auf dem auch moderne Systeme beschrieben und kombiniert werden (z. B. nutzt AlphaGo Baumsuche *plus* neuronale Netze).

### 1.2 Agenten und Umgebungen

Ein **Agent** ist alles, was seine Umgebung über **Sensoren** wahrnimmt und über **Aktuatoren** auf sie einwirkt.

```
        Wahrnehmungen (percepts)
   Umgebung ────────────────────▶ Agent
   Umgebung ◀──────────────────── Agent
              Aktionen (actions)
```

Eine Aufgabe beschreibt man mit dem **PEAS-Schema**:

| Buchstabe | Bedeutung | Beispiel: selbstfahrendes Taxi |
|--|--|--|
| **P**erformance | Erfolgsmaß | sicher, schnell, legal, komfortabel |
| **E**nvironment | Umgebung | Straßen, Verkehr, Fußgänger, Wetter |
| **A**ctuators | Aktuatoren | Lenkung, Gas, Bremse, Blinker |
| **S**ensors | Sensoren | Kameras, Lidar, GPS, Tacho |

**Eigenschaften von Umgebungen** (wichtig, weil sie bestimmen, welche Methode passt):

- **vollständig vs. teilweise beobachtbar** — sieht der Agent den ganzen relevanten Zustand? (Schach: ja. Poker: nein.)
- **deterministisch vs. stochastisch** — ist die Folge einer Aktion sicher vorhersagbar?
- **episodisch vs. sequenziell** — hängt die nächste Entscheidung von früheren ab?
- **statisch vs. dynamisch** — verändert sich die Welt, während der Agent nachdenkt?
- **diskret vs. stetig** — endlich viele Zustände/Aktionen oder kontinuierlich?
- **Einzelagent vs. Multiagent** — gibt es Mitspieler/Gegner?

> **Faustregel:** Je weiter rechts (teilweise beobachtbar, stochastisch, dynamisch, stetig, multiagent), desto schwieriger — und desto eher braucht man Wahrscheinlichkeit und Lernen statt reiner Suche.

**Agentenarchitekturen**, von einfach nach mächtig:

1. **Einfacher Reflexagent**: `wenn Wahrnehmung X, dann Aktion Y` (Thermostat).
2. **Modellbasierter Reflexagent**: hält einen internen Zustand über die Welt (Staubsaugerroboter mit Karte).
3. **Zielbasierter Agent**: plant Aktionsfolgen, um ein Ziel zu erreichen (Navi) → führt direkt zur **Suche** (Abschnitt 1.3).
4. **Nutzenbasierter Agent**: bewertet Zustände graduell über eine **Utility-Funktion** (nicht nur Ziel/kein Ziel, sondern „wie gut?").
5. **Lernender Agent**: verbessert alle obigen Komponenten aus Erfahrung → **maschinelles Lernen** (Abschnitt 2.5).

### 1.3 Problemlösen durch Suche

Die zentrale Idee der klassischen KI: Viele Probleme lassen sich als **Suche in einem Zustandsraum** formulieren. Dazu braucht man fünf Zutaten:

1. **Anfangszustand** — wo starte ich?
2. **Aktionen** — was kann ich in einem Zustand tun?
3. **Übergangsmodell** — welcher Zustand folgt auf Aktion $a$ in Zustand $s$?
4. **Zieltest** — bin ich fertig?
5. **Pfadkosten** — was kostet eine Aktionsfolge (Schritte, Kilometer, Zeit …)?

**Durchgerechnetes Mini-Beispiel — das 8-Puzzle:** Ein 3×3-Schiebepuzzle mit Steinen 1–8 und einem Loch. Zustand = Anordnung der Steine (es gibt $9!/2 = 181\,440$ erreichbare), Aktionen = Loch nach oben/unten/links/rechts schieben, Ziel = sortierte Anordnung, Kosten = Anzahl Züge. Damit ist das Puzzle vollständig als Suchproblem formalisiert — der Algorithmus muss nichts über „Puzzles" wissen.

Die Suche baut vom Anfangszustand aus einen **Suchbaum**: Knoten = Zustände, Kanten = Aktionen. Die noch nicht untersuchten Randknoten heißen **Frontier** (Grenze). Alle Suchalgorithmen unterscheiden sich nur darin, **welchen Frontier-Knoten sie als Nächstes expandieren**.

#### Uninformierte Suche (kennt nur das Problem, keine Zusatzhinweise)

| Algorithmus | Frontier-Strategie | vollständig? | optimal? | Merksatz |
|--|--|--|--|--|
| **Breitensuche (BFS)** | Warteschlange (FIFO) | ja | ja (bei gleichen Schrittkosten) | erst alle Nachbarn, dann deren Nachbarn |
| **Tiefensuche (DFS)** | Stapel (LIFO) | nein (Endlospfade!) | nein | immer weiter in die Tiefe, dann Backtracking |
| **Uniform-Cost (Dijkstra)** | Prioritätswarteschlange nach Pfadkosten $g(n)$ | ja | ja | billigsten bekannten Pfad zuerst |
| **Iterative Vertiefung (IDS)** | DFS mit wachsendem Tiefenlimit | ja | ja (wie BFS) | BFS-Garantien mit DFS-Speicherbedarf |

- *Vollständig* = findet eine Lösung, wenn eine existiert. *Optimal* = findet die billigste.
- BFS braucht **exponentiell viel Speicher** ($O(b^d)$ bei Verzweigungsfaktor $b$ und Tiefe $d$) — das ist in der Praxis sein Todesurteil bei tiefen Problemen; IDS ist dann der Trick.

#### Informierte (heuristische) Suche

Eine **Heuristik** $h(n)$ schätzt die Restkosten vom Knoten $n$ zum Ziel. Beispiel Routenplanung: Luftlinie zum Ziel. Beispiel 8-Puzzle: Anzahl falsch liegender Steine, oder besser die **Manhattan-Distanz** (Summe der horizontalen+vertikalen Abstände jedes Steins zu seinem Zielfeld).

- **Greedy Best-First**: expandiere den Knoten mit kleinstem $h(n)$. Schnell, aber weder vollständig noch optimal — rennt gierig Richtung Ziel und übersieht bessere Wege.
- **A\***: expandiere den Knoten mit kleinstem
$$f(n) = g(n) + h(n)$$
wobei $g(n)$ = bisherige Pfadkosten, $h(n)$ = geschätzte Restkosten. A\* kombiniert also „was hat es gekostet" mit „was wird es noch kosten".

**Der zentrale Satz:** Ist $h$ **zulässig** (admissible), d. h. überschätzt die echten Restkosten nie ($h(n) \le h^*(n)$), dann ist A\* **optimal**.

> **Intuition, warum das stimmt:** A\* nimmt sich immer den Knoten mit dem optimistischsten Gesamtschätzwert vor. Wenn A\* ein Ziel expandiert, haben alle anderen offenen Knoten $f$-Werte ≥ den Zielkosten — und da $h$ nie überschätzt, kann über sie kein billigerer Weg mehr führen.

Für Graphensuche (mit Duplikat-Erkennung) braucht man die etwas stärkere **Konsistenz**: $h(n) \le c(n, n') + h(n')$ — die Heuristik darf entlang einer Kante nie stärker fallen als die Kantenkosten (Dreiecksungleichung). Konsistent ⇒ zulässig; die Manhattan-Distanz und die Luftlinie sind beides.

**Qualitätsvergleich von Heuristiken:** $h_2$ *dominiert* $h_1$, wenn $h_2(n) \ge h_1(n)$ für alle $n$ (bei Zulässigkeit beider). Dominante Heuristiken expandieren nie mehr Knoten — beim 8-Puzzle schlägt Manhattan-Distanz die „falsch liegenden Steine" deutlich. Ideal ist $h$ so groß wie möglich, aber noch zulässig.

---

## 2. Aufbau (Intermediate)

### 2.1 Spiele: Adversariale Suche

Bei Zwei-Personen-Nullsummenspielen (Schach, Tic-Tac-Toe) plant ein Gegner aktiv *gegen* uns. Lösung: **Minimax**.

**Idee:** Baue den Spielbaum auf. Blätter bekommen einen Wert aus Sicht von Spieler MAX (+1 Gewinn, 0 Remis, −1 Verlust). Dann propagiere nach oben: MAX-Knoten nehmen das **Maximum** ihrer Kinder (ich wähle meinen besten Zug), MIN-Knoten das **Minimum** (der Gegner wählt den für mich schlimmsten).

$$\text{Minimax}(s) = \begin{cases} \text{Utility}(s) & s \text{ terminal} \\ \max_{a} \text{Minimax}(\text{Result}(s,a)) & \text{MAX am Zug} \\ \min_{a} \text{Minimax}(\text{Result}(s,a)) & \text{MIN am Zug} \end{cases}$$

Minimax spielt **perfekt gegen einen perfekten Gegner**. Problem: Der Spielbaum explodiert ($b^m$ Knoten; Schach: $b \approx 35$, $m \approx 80$ → hoffnungslos). Zwei Standard-Auswege:

1. **Alpha-Beta-Pruning**: schneide Teilbäume ab, die das Ergebnis beweisbar nicht mehr ändern können. $\alpha$ = beste bereits garantierte Option für MAX auf dem Pfad, $\beta$ = beste für MIN. Sobald an einem MIN-Knoten ein Wert ≤ $\alpha$ auftaucht (oder an einem MAX-Knoten ≥ $\beta$), kann man den Rest der Kinder ignorieren — der Elternknoten würde diesen Zweig nie wählen. Bei guter Zugsortierung sinkt der Aufwand von $O(b^m)$ auf $O(b^{m/2})$ — **doppelte Suchtiefe zum gleichen Preis**, und das Ergebnis ist *exakt* dasselbe wie bei Minimax.
2. **Tiefenlimit + Bewertungsfunktion**: statt bis zu den Blättern zu rechnen, brich bei Tiefe $d$ ab und schätze die Stellung mit einer **Evaluationsfunktion** (z. B. Materialbilanz im Schach). Ab hier ist das Spiel nicht mehr perfekt, aber praktikabel.

Moderne Spielprogramme (AlphaGo/AlphaZero) ersetzen die handgebaute Bewertungsfunktion durch ein **gelerntes neuronales Netz** und die vollständige Expansion durch **Monte-Carlo Tree Search (MCTS)** — die Grundidee „Baumsuche + Stellungsbewertung" bleibt.

### 2.2 Constraint Satisfaction Problems (CSPs)

Viele Probleme sind keine Pfadsuche, sondern eine **Belegungssuche**: Finde Werte für Variablen, sodass alle Nebenbedingungen erfüllt sind.

- **Variablen** $X_1, \dots, X_n$, jede mit **Domäne** (Wertemenge) $D_i$
- **Constraints**: erlaubte Wertkombinationen

**Beispiele:** Sudoku (81 Variablen, Domäne 1–9, Zeilen/Spalten/Box-Constraints), Kartenfärbung (Nachbarländer verschieden färben), Stundenplanung (kein Dozent doppelt belegt).

**Lösungsverfahren: Backtracking-Suche** — belege Variablen der Reihe nach; führt eine Belegung zum Widerspruch, nimm sie zurück und probiere den nächsten Wert. Das allein ist rohe Gewalt; klug wird es durch drei Standard-Verbesserungen:

1. **MRV (Minimum Remaining Values)**: wähle als nächstes die Variable mit den *wenigsten* noch möglichen Werten („fail first" — Sackgassen früh erkennen).
2. **Forward Checking**: nach jeder Belegung streiche unvereinbare Werte aus den Domänen der Nachbarvariablen; wird eine Domäne leer → sofort backtracken.
3. **Constraint Propagation (AC-3, Kantenkonsistenz)**: propagiere Einschränkungen durchs ganze Netz, bevor überhaupt gesucht wird. Bei leichten Sudokus reicht Propagation oft ganz ohne Suche — genau das macht ein Mensch, der „Kandidaten streicht".

> **Warum CSPs ein eigenes Kapitel sind:** Die Constraints geben der Suche *Struktur*, die generische Zustandsraumsuche ignorieren würde. Die Kombination „Suche + Propagation" ist ein wiederkehrendes KI-Muster.

### 2.3 Wissensrepräsentation und Logik

Ein **wissensbasierter Agent** speichert Wissen als Sätze in einer formalen Sprache (**Knowledge Base, KB**) und leitet daraus neue Schlüsse ab (**Inferenz**).

**Aussagenlogik (propositional logic)** — die einfachste Logik:

- Atome: $P, Q, R$ (wahr/falsch), Junktoren: $\neg, \land, \lor, \Rightarrow, \Leftrightarrow$
- Zentral ist der Begriff der **Folgerung (entailment)**: $KB \models \alpha$ heißt: *In jeder Welt, in der KB wahr ist, ist auch $\alpha$ wahr.*
- Prüfbar z. B. per **Wahrheitstabelle** (alle Modelle durchgehen — korrekt, aber exponentiell) oder per **Resolution** / **Forward Chaining** (Regeln anwenden).

**Durchgerechnetes Mini-Beispiel:** KB = { „Wenn es regnet, ist die Straße nass" ($R \Rightarrow N$), „Es regnet" ($R$) }. Mit **Modus Ponens** folgt $N$. Umgekehrt gilt aber $N \not\Rightarrow R$ — aus „Straße nass" folgt nicht „Regen" (vielleicht war es der Sprengwagen). Dieser Fehler (Abduktion mit Deduktion verwechseln) ist auch im Alltag häufig.

**Prädikatenlogik (first-order logic, FOL)** erweitert das um Objekte, Relationen und Quantoren:
$$\forall x\, (\text{Student}(x) \Rightarrow \text{Lernt}(x)) \qquad \exists x\, \text{Besteht}(x)$$
Damit kann man Welten kompakt beschreiben, die in Aussagenlogik unendlich viele Atome bräuchten. Preis: Inferenz in FOL ist nur noch **semi-entscheidbar** (wenn etwas folgt, findet man es; wenn nicht, terminiert die Suche evtl. nie).

**Einordnung:** Reine Logik-KI scheiterte historisch an zwei Dingen: der Mühsal, Alltagswissen von Hand zu formalisieren (**Knowledge Acquisition Bottleneck**), und ihrer Unfähigkeit, mit *Unsicherheit* umzugehen. Ersteres motiviert maschinelles Lernen, letzteres den nächsten Abschnitt. Logik lebt heute u. a. in Datenbanken, Verifikation und logischer Programmierung weiter (→ Module *Deduktive Datenbanken*, *Logische Programmierung*).

### 2.4 Schließen unter Unsicherheit

Die echte Welt ist stochastisch und teilweise beobachtbar. Das Werkzeug dafür ist die **Wahrscheinlichkeitstheorie**.

Kernbegriffe: $P(A)$ (Wahrscheinlichkeit), $P(A \mid B)$ (bedingte W'keit: Wahrscheinlichkeit von $A$, *gegeben* dass $B$ gilt), Produktregel $P(A \land B) = P(A\mid B)\,P(B)$. Daraus folgt die wichtigste Formel des Moduls:

$$P(H \mid E) = \frac{P(E \mid H)\; P(H)}{P(E)} \qquad \text{(Bayes-Regel)}$$

Lies sie so: **Posterior** (Glaube an Hypothese $H$ nach Beobachtung $E$) = **Likelihood** (wie gut erklärt $H$ die Beobachtung?) × **Prior** (Glaube vorher), normiert durch $P(E)$.

**Durchgerechnetes Beispiel (der Klassiker, den fast jeder falsch schätzt):** Ein Test erkennt eine Krankheit zu 99 % ($P(+\mid K) = 0{,}99$), Falsch-Positiv-Rate 5 % ($P(+\mid \neg K) = 0{,}05$), die Krankheit betrifft 1 % der Bevölkerung ($P(K) = 0{,}01$). Wie wahrscheinlich ist Krankheit bei positivem Test?

$$P(K \mid +) = \frac{0{,}99 \cdot 0{,}01}{0{,}99 \cdot 0{,}01 + 0{,}05 \cdot 0{,}99} = \frac{0{,}0099}{0{,}0099 + 0{,}0495} \approx 0{,}167$$

Nur **~17 %**! Der niedrige Prior (seltene Krankheit) drückt den Posterior — die vielen Gesunden produzieren mehr Falsch-Positive als die wenigen Kranken echte Positive. Diese „Base-Rate-Vernachlässigung" zu durchschauen ist der halbe Wert der Bayes-Regel.

**Naive Bayes — die Bayes-Regel als Klassifikator:** Um ein Objekt mit Merkmalen $x_1, \dots, x_n$ (z. B. die Wörter einer E-Mail) in eine Klasse $c$ (Spam/Ham) einzuordnen, nimmt man „naiv" an, die Merkmale seien **gegeben die Klasse unabhängig**:

$$P(c \mid x_1, \dots, x_n) \;\propto\; P(c) \prod_{i=1}^{n} P(x_i \mid c)$$

Die Annahme ist fast immer falsch (Wörter hängen voneinander ab!) — aber der Klassifikator funktioniert trotzdem erstaunlich gut, ist in Sekunden trainiert und war jahrelang das Rückgrat von Spamfiltern. Du baust im Final-Projekt selbst einen.

**Bayes-Netze** (Ausblick): gerichtete Graphen, die Abhängigkeiten zwischen Zufallsvariablen kompakt kodieren — statt einer Riesentabelle über alle Variablenkombinationen nur lokale bedingte Verteilungen pro Knoten. Sie sind das Bindeglied zwischen Logik-KI („Struktur") und Statistik („Unsicherheit"); Vertiefung folgt in *Machine Learning 1* und *Theorie der KI*.

### 2.5 Maschinelles Lernen — der Überblick

Statt Verhalten zu programmieren, lässt man den Agenten es **aus Daten lernen**. Drei Grundarten:

| Art | Gegeben | Gelernt wird | Typische Beispiele |
|--|--|--|--|
| **Supervised Learning** | Eingaben *mit* richtigen Ausgaben (Labels) | Abbildung Eingabe → Ausgabe | Spamfilter, Bilderkennung, Preisvorhersage |
| **Unsupervised Learning** | nur Eingaben, keine Labels | Struktur in den Daten | Clustering von Kunden, Dimensionsreduktion |
| **Reinforcement Learning** | Belohnungssignal nach Aktionen | Verhaltensstrategie (Policy) | Spiele, Robotik, Regelung |

Beim Supervised Learning unterscheidet man **Klassifikation** (diskrete Ausgabe: Spam/Ham) und **Regression** (stetige Ausgabe: Hauspreis).

Zwei Begriffe, die du von Anfang an richtig lernen solltest:

- **Generalisierung**: Es zählt die Leistung auf *neuen* Daten, nicht auf den Trainingsdaten. Deshalb teilt man Daten immer in **Trainings- und Testmenge**.
- **Overfitting**: Ein zu flexibles Modell lernt die Trainingsdaten auswendig (inklusive Rauschen) und versagt auf neuen Daten. Symptom: Trainingsfehler klein, Testfehler groß.

Mehr dazu ausführlich in *Machine Learning 1* — hier reicht die Landkarte.

---

## 3. Advanced-Themen

### 3.1 Lokale Suche und Optimierung

Manchmal interessiert nicht der *Pfad*, sondern nur der *Zielzustand* (z. B. eine gültige Stundenplan-Belegung, ein gutes Platinenlayout). Dann kann man auf den Suchbaum verzichten und direkt im Zustandsraum „wandern":

- **Hill Climbing**: gehe immer zum besten Nachbarzustand. Schnell und speicherfrei ($O(1)$), bleibt aber in **lokalen Maxima**, auf Plateaus und Graten hängen.
- **Simulated Annealing**: akzeptiere Verschlechterungen mit Wahrscheinlichkeit $e^{\Delta E / T}$, wobei die „Temperatur" $T$ langsam sinkt. Anfangs viel Zufall (Entkommen aus lokalen Maxima), am Ende fast reines Hill Climbing. Bei hinreichend langsamer Abkühlung findet es beweisbar das globale Optimum — praktisch nutzt man es als robusten Kompromiss.
- **Genetische Algorithmen**: eine *Population* von Lösungen; die besten werden selektiert, per **Crossover** kombiniert und per **Mutation** variiert. Nützlich, wenn Lösungen sich sinnvoll kombinieren lassen; oft aber weniger effizient als problemspezifische Verfahren.
- **Gradient Descent**: in *stetigen* Räumen folgt man dem negativen Gradienten $\theta \leftarrow \theta - \eta \nabla f(\theta)$. Das ist derselbe „bergab"-Gedanke wie Hill Climbing — und **der** Algorithmus, mit dem neuronale Netze trainiert werden. Lokale Suche ist also kein Nischenthema, sondern der Kern des Deep Learning.

### 3.2 Klassische Planung (kurz)

**Planung** = Suche mit *strukturierten* Zuständen: Zustände sind Mengen logischer Fakten, Aktionen haben **Vorbedingungen** und **Effekte** (STRIPS/PDDL-Formalismus):

```
Aktion: Aufheben(x)
  Vorbedingung: Greifer frei, x liegt oben
  Effekt:       Greifer hält x, ¬(Greifer frei)
```

Der Gewinn gegenüber roher Zustandsraumsuche: Aus der Aktionsstruktur lassen sich **automatisch Heuristiken** ableiten (z. B. „ignoriere alle negativen Effekte" → Relaxation → zulässige Schätzung). Planung ist das Bindeglied zwischen Suche und Logik und Grundlage der Robotik-Module.

### 3.3 Von neuronalen Netzen zu LLMs — die moderne KI-Landschaft

Ein **künstliches Neuron** berechnet $y = \sigma(\sum_i w_i x_i + b)$ — gewichtete Summe plus nichtlineare Aktivierung $\sigma$. Schichtet man viele davon (**Deep Learning**), kann das Netz hierarchische Merkmale lernen: Kanten → Formen → Objekte. Trainiert wird per Gradient Descent mit **Backpropagation** (effiziente Gradientenberechnung durch Kettenregel).

Meilensteine und was sie konzeptuell zeigen:

- **AlexNet (2012)**: Deep Learning + GPUs + große Datenmengen schlagen handgebaute Bildmerkmale → *Repräsentationen lernen statt konstruieren*.
- **AlphaGo (2016)**: gelernte Bewertungs-/Zugnetze + Monte-Carlo-Baumsuche → *Lernen und Suche kombiniert*, klassische KI und Deep Learning sind keine Gegensätze.
- **Transformer (2017)** und **LLMs (ab ~2020)**: Modelle, die auf riesigen Textmengen lernen, das nächste Token vorherzusagen, entwickeln breite sprachliche und faktische Fähigkeiten. Mit Instruction-Tuning und RLHF (Reinforcement Learning from Human Feedback) werden daraus Assistenten.

**Einordnung für dieses Modul:** Ein LLM ist im Agenten-Vokabular ein gelerntes Modell, das man zum Kern eines Agenten machen kann (Wahrnehmung: Text/Bilder; Aktionen: Text, Tool-Aufrufe). Die klassischen Fragen — Was ist das Erfolgsmaß? Wie geht der Agent mit Unsicherheit um? Plant er, oder reagiert er nur? — bleiben exakt dieselben. Deshalb lohnt sich das Fundament dieses Moduls auch im LLM-Zeitalter.

### 3.4 Grenzen, Ethik, Verantwortung

Gehört heute zwingend zur Einführung:

- **Bias & Fairness**: Modelle lernen Verzerrungen aus ihren Trainingsdaten (z. B. diskriminierende Muster in historischen Einstellungsdaten). „Die Daten sind objektiv" ist ein Trugschluss.
- **Erklärbarkeit (XAI)**: Tiefe Modelle sind schwer zu interpretieren — problematisch bei Kredit-, Medizin- oder Justizentscheidungen.
- **Robustheit**: Adversariale Beispiele (minimal veränderte Eingaben kippen die Vorhersage), Verteilungsverschiebung (Modell trifft auf Daten, die anders aussehen als das Training).
- **Alignment**: Wie stellt man sicher, dass ein optimierender Agent das tut, was wir *meinen*, nicht was wir *messen*? (Klassisches Beispiel: ein Agent, der ein Belohnungsmaß maximiert, findet oft Schlupflöcher — „reward hacking".)
- **Regulierung**: Die EU hat mit dem **AI Act** (in Kraft seit 2024, gestaffelte Anwendung) einen risikobasierten Rechtsrahmen geschaffen — von verbotenen Praktiken über Hochrisiko-Anforderungen bis zu Transparenzpflichten für generative Modelle.

---

## 4. Zusammenfassung / Cheat-Sheet

**Agenten**
- Rationaler Agent: maximiert erwartetes Erfolgsmaß gegeben Wahrnehmung + Wissen
- PEAS: Performance, Environment, Actuators, Sensors
- Umgebungsachsen: beobachtbar · deterministisch · episodisch · statisch · diskret · Einzelagent

**Suche**
- Problem = (Anfangszustand, Aktionen, Übergänge, Zieltest, Kosten)
- BFS: FIFO, optimal bei Einheitskosten, Speicher $O(b^d)$ · DFS: LIFO, speicherarm, nicht optimal · UCS: nach $g(n)$, optimal · IDS: BFS-Garantien, DFS-Speicher
- Greedy: nur $h(n)$ · **A\***: $f(n) = g(n) + h(n)$
- $h$ zulässig ($h \le h^*$) ⇒ A\* optimal; konsistent: $h(n) \le c(n,n') + h(n')$
- Bessere (dominante) Heuristik ⇒ weniger expandierte Knoten

**Spiele**
- Minimax: MAX maximiert, MIN minimiert; perfekt gegen perfekten Gegner
- Alpha-Beta: gleiches Ergebnis, bis zu $O(b^{m/2})$; Zugsortierung entscheidend
- Praxis: Tiefenlimit + Evaluationsfunktion (oder gelerntes Netz + MCTS)

**CSP**
- Variablen + Domänen + Constraints; Backtracking + MRV + Forward Checking + AC-3

**Logik**
- $KB \models \alpha$: $\alpha$ gilt in allen Modellen der KB
- Modus Ponens: aus $P \Rightarrow Q$ und $P$ folgt $Q$ (Umkehrung gilt nicht!)
- FOL: Objekte, Relationen, $\forall$, $\exists$ — mächtiger, Inferenz nur semi-entscheidbar

**Unsicherheit**
- Bayes: $P(H\mid E) = P(E\mid H)P(H)/P(E)$ — Posterior ∝ Likelihood × Prior
- Basisraten nicht vergessen (17-%-Beispiel!)
- Naive Bayes: $P(c\mid x) \propto P(c)\prod_i P(x_i\mid c)$ — „naiv" = bedingte Unabhängigkeit

**ML-Landkarte**
- supervised (Labels) · unsupervised (Struktur) · reinforcement (Belohnung)
- Generalisierung > Trainingsleistung; Overfitting = auswendig gelernt
- Deep Learning = geschichtete Neuronen + Gradient Descent + Backpropagation

---

## 5. Selbsttest

Versuche erst selbst zu antworten, dann aufklappen.

<details><summary><b>1. Warum definiert die moderne KI Intelligenz über „rationales Handeln" statt über „menschliches Denken"?</b></summary>

Rationales Handeln ist ein *objektives, messbares* Kriterium: Maximiere das erwartete Erfolgsmaß gegeben Wahrnehmung und Wissen. Menschliches Denken ist dagegen schwer zu spezifizieren, teils fehlerhaft (kognitive Verzerrungen) und als Bauvorgabe weder nötig noch hinreichend — so wie Flugzeuge nicht mit Federschlag fliegen. Außerdem lässt sich Rationalität mathematisch analysieren (Optimalität, Garantien).
</details>

<details><summary><b>2. Formuliere „Staubsaugerroboter reinigt eine Wohnung" als PEAS-Beschreibung.</b></summary>

**P**: gereinigte Fläche pro Zeit, Akkuverbrauch, keine Schäden/Abstürze (Treppen!). **E**: Wohnung mit Räumen, Möbeln, Teppichen, Menschen/Haustieren (dynamisch, teilweise beobachtbar). **A**: Räder/Motoren, Saugeinheit, Bürsten. **S**: Stoßsensoren, Abgrundsensoren, ggf. Lidar/Kamera, Ladezustand.
</details>

<details><summary><b>3. Warum ist DFS nicht vollständig, und mit welchem Trick bekommt man DFS-Speicherbedarf und BFS-Garantien zugleich?</b></summary>

DFS kann in unendlich tiefe (oder bei Zyklen: endlose) Pfade laufen und das Ziel nie erreichen, obwohl es in geringer Tiefe liegt. Der Trick ist **Iterative Deepening (IDS)**: DFS mit Tiefenlimit 0, 1, 2, … Die flachen Ebenen werden zwar mehrfach expandiert, aber weil die unterste Ebene exponentiell dominiert, kostet das nur einen konstanten Faktor.
</details>

<details><summary><b>4. Eine Heuristik überschätzt die Restkosten an genau einem Knoten. Was kann bei A* schiefgehen?</b></summary>

A\* kann **suboptimal** werden: Liegt der überschätzte Knoten auf dem (echten) optimalen Pfad, bekommt er einen zu hohen $f$-Wert und wird evtl. hinter einem schlechteren Ziel zurückgestellt — A\* gibt dann den teureren Pfad aus. Vollständig bleibt A\* (bei endlichen Graphen) trotzdem; nur die Optimalitätsgarantie hängt an der Zulässigkeit.
</details>

<details><summary><b>5. Warum liefert Alpha-Beta-Pruning exakt dasselbe Ergebnis wie Minimax, obwohl es Teilbäume überspringt?</b></summary>

Es überspringt nur Teilbäume, die das Endergebnis *beweisbar* nicht beeinflussen können: Wenn an einem MIN-Knoten schon ein Wert ≤ $\alpha$ gefunden ist, weiß man, dass MAX diesen Knoten nie wählen wird (MAX hat anderswo bereits $\alpha$ sicher) — der genaue Wert des Knotens ist dann irrelevant. Es wird also nichts „geschätzt", sondern nur Unnötiges nicht ausgerechnet.
</details>

<details><summary><b>6. Modelliere Sudoku als CSP und erkläre, was AC-3 dort leistet.</b></summary>

81 Variablen (Felder), Domänen {1,…,9} (vorgegebene Felder: einelementig), Constraints: alle Werte in jeder Zeile, Spalte und 3×3-Box paarweise verschieden. AC-3 macht die Kanten konsistent: Steht in einem Feld eine 5 fest, wird die 5 aus den Domänen aller Zeilen-/Spalten-/Box-Nachbarn gestrichen; Streichungen stoßen weitere Prüfungen an. Das entspricht dem menschlichen „Kandidaten streichen" und löst leichte Sudokus ganz ohne Suche.
</details>

<details><summary><b>7. Aus „Wenn es regnet, ist die Straße nass" und „Die Straße ist nass" schließt jemand „Es regnet". Wie heißt der Fehler?</b></summary>

**Affirming the consequent** (Bejahung des Konsequens): Aus $R \Rightarrow N$ und $N$ folgt logisch *nichts* über $R$ — die Straße kann aus anderen Gründen nass sein. Gültig wären Modus Ponens ($R \Rightarrow N$, $R$ ⊢ $N$) oder Modus Tollens ($R \Rightarrow N$, $\neg N$ ⊢ $\neg R$). Als *probabilistischer* Schluss (Abduktion) kann „Regen ist wahrscheinlicher geworden" aber durchaus vernünftig sein — dafür braucht man die Bayes-Regel statt Logik.
</details>

<details><summary><b>8. Test: 99 % Sensitivität, 5 % Falsch-Positiv-Rate, Krankheit bei 0,1 % (statt 1 %). Posterior bei positivem Test?</b></summary>

$P(K\mid+) = \frac{0{,}99 \cdot 0{,}001}{0{,}99 \cdot 0{,}001 + 0{,}05 \cdot 0{,}999} = \frac{0{,}00099}{0{,}00099 + 0{,}04995} \approx 0{,}019$ — knapp **2 %**. Je seltener die Krankheit, desto stärker dominiert der Prior: Fast alle positiven Tests sind Falsch-Positive.
</details>

<details><summary><b>9. Warum funktioniert Naive Bayes oft gut, obwohl seine Unabhängigkeitsannahme fast immer verletzt ist?</b></summary>

Für die *Klassifikation* muss nur die **Rangfolge** der Klassen-Posteriors stimmen, nicht ihr exakter Wert. Die Unabhängigkeitsannahme verzerrt die Wahrscheinlichkeiten (macht sie oft übertrieben sicher), kippt aber selten die Reihenfolge. Dazu kommt: Wenige Parameter ⇒ wenig Overfitting, gerade bei kleinen Datenmengen und vielen Merkmalen (Text!).
</details>

<details><summary><b>10. Ordne ein: Ist ein Schachprogramm mit Alpha-Beta „lernende KI"? Ist ein LLM ein „Agent"?</b></summary>

Alpha-Beta-Schach ist KI (rationales Handeln durch Suche), aber *kein* Lernen — alles Verhalten steckt in Algorithmus + Bewertungsfunktion. Ein LLM allein ist erst mal ein gelerntes *Modell* (Text rein, Text raus). Zum **Agenten** wird es, wenn man es in eine Wahrnehmungs-Handlungs-Schleife einbettet: Es bekommt Beobachtungen (Nutzereingaben, Tool-Ergebnisse), wählt Aktionen (Antworten, Tool-Aufrufe) und verfolgt ein Ziel. Agent ist eine *Rolle*, kein Modelltyp.
</details>

---

## 6. Literatur & Quellen

**Lehrbücher**

- **Russell & Norvig — „Artificial Intelligence: A Modern Approach" (AIMA), 4. Aufl.** — *das* Standardwerk, deckt dieses Modul fast 1:1 ab. Kapitel 1–2 (Einführung, Agenten), 3 (Suche), 5 (Spiele), 6 (CSPs), 7–9 (Logik), 12–13 (Unsicherheit, Bayes). *(vertiefend, aber gut lesbar; deutsche Übersetzung existiert)*
- **Poole & Mackworth — „Artificial Intelligence: Foundations of Computational Agents", 3. Aufl.** — komplett **kostenlos** online: https://artint.info *(einsteigerfreundlich)*

**Onlinekurse (kostenlos)**

- **UC Berkeley CS188 — Intro to AI**: Vorlesungsvideos, Folien und die berühmten Pac-Man-Projekte frei verfügbar: https://inst.eecs.berkeley.edu/~cs188/ *(einsteigerfreundlich, deckt exakt Suche/Spiele/CSP/Bayes ab — beste Ergänzung zu diesem Modul)*
- **Harvard CS50's Introduction to AI with Python** (edX/YouTube, kostenlos) *(sehr einsteigerfreundlich, praktisch orientiert)*
- **MIT 6.034 Artificial Intelligence** (OpenCourseWare, Patrick Winston) *(klassisch, hervorragende Vorlesungen)*

**Interaktive Visualisierungen & Blogposts (kostenlos)**

- *Red Blob Games — Introduction to A\**: https://www.redblobgames.com/pathfinding/a-star/introduction.html — die beste interaktive A\*-Erklärung im Netz *(einsteigerfreundlich, Pflichtlektüre vor Projekt 1)*
- *Setosa — Conditional probability visualized*: https://setosa.io/ev/conditional-probability/ *(einsteigerfreundlich)*
- 3Blue1Brown: *Bayes theorem* (YouTube) — geometrische Intuition für die Bayes-Regel *(einsteigerfreundlich)*

**Historisches / Vertiefendes**

- Turing (1950): *Computing Machinery and Intelligence* — das Original zum Turing-Test, gut lesbar. *(kostenlos online, vertiefend)*
- Silver et al. (2016): *Mastering the game of Go with deep neural networks and tree search* (Nature) — AlphaGo: Suche + Lernen kombiniert. *(vertiefend)*

---

**Nächster Schritt:** Ab in die Projekte → `projects/01-basic/` (A\*-Wegsuche), dann `projects/02-medium/` (Tic-Tac-Toe mit Minimax), dann `projects/03-final/` (Spamfilter mit Naive Bayes auf echten Daten).
