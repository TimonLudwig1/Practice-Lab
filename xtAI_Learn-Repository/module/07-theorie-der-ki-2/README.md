# Modul 07 — Theorie der Künstlichen Intelligenz 2

**Worum geht es?** Modul 06 hat gezeigt, wie ein Agent in einer *bekannten,
deterministischen* Welt durch Suche und Logik handelt. Dieses Modul macht drei
Schritte weiter: (1) **Planung** — Handeln in strukturierten, aber weiterhin
deterministischen Welten mit einer kompakten, logik-nahen Repräsentation von
Aktionen; (2) **Schließen unter Unsicherheit** — wenn die Welt nicht mehr
sicher bekannt ist, ersetzen wir „wahr/falsch" durch **Wahrscheinlichkeiten**
und rechnen mit **Bayes-Netzen**; (3) **rationales Entscheiden** — wenn Aktionen
unsichere Ausgänge haben, maximiert der Agent den **erwarteten Nutzen**, im
sequenziellen Fall über **Markov-Entscheidungsprozesse (MDPs)**. Das ist der
Übergang von der *symbolischen* zur *probabilistischen* und
*entscheidungstheoretischen* KI — und die theoretische Brücke zu Reinforcement
Learning (Module 13/14).

**Hilfreiche Vorkenntnisse.** Modul 06 (Suche, Logik — Planung baut direkt
darauf auf). Grundlagen der Wahrscheinlichkeitsrechnung und lineare Algebra
(Module 02/03 genügen). Etwas Analysis für die Konvergenzbeweise (Kontraktion).

**Empfohlene Vormodule.** Modul 06 „Theorie der KI 1" (zwingend fürs Planungs­kapitel),
Data Science 1/2 für die Wahrscheinlichkeitsbasics.

**Folgemodule.** Reinforcement Learning (13) und Deep RL (14) bauen unmittelbar
auf dem MDP-Teil auf. Bayes-Netze tauchen in ML, NLP und Bioinformatik wieder auf.

---

## Lernziele

Nach diesem Modul solltest du in der Lage sein,

- ein Planungsproblem in **STRIPS/PDDL** zu formalisieren und den Unterschied
  zwischen **Vorwärts-(Progression)-, Rückwärts-(Regression)- und Plan-Raum-Suche**
  zu erklären;
- **domänenunabhängige Heuristiken** aus der **Relaxation** (Ignorieren der
  Delete-Listen, $h_{\text{add}}$, $h_{\max}$, $h_{\text{FF}}$) herzuleiten;
- **GraphPlan** (Planungsgraph, Mutex-Relationen) und **SATPlan** als alternative
  Planungsparadigmen einzuordnen;
- die **Axiome der Wahrscheinlichkeit**, bedingte Wahrscheinlichkeit, den **Satz
  von Bayes** und **(bedingte) Unabhängigkeit** sicher anzuwenden;
- ein **Bayes-Netz** zu konstruieren, seine **Faktorisierung der Verbundverteilung**
  zu begründen und mit **d-Separation** Unabhängigkeiten abzulesen;
- **exakte Inferenz** (Aufzählung, **Variable Elimination**) und **approximative
  Inferenz** (Likelihood Weighting, **Gibbs-Sampling/MCMC**) durchzuführen;
- **temporale Modelle** (Markov-Kette, **HMM**) zu verstehen und **Filtering,
  Smoothing** und den **Viterbi-Algorithmus** herzuleiten;
- **rationale Entscheidungen** über den **Maximum-Expected-Utility (MEU)**-Grundsatz
  zu treffen, den **Wert von Information (VPI)** zu berechnen;
- **MDPs** zu definieren, die **Bellman-Gleichungen** aufzustellen und **Value
  Iteration** und **Policy Iteration** samt **Konvergenzbeweis (Kontraktion)** zu
  verstehen;
- den Zusammenhang zwischen Planung, probabilistischem Schließen und
  sequenzieller Entscheidung zu erklären.

---

## Teil 1 — Grundlagen: Klassische Planung

### 1.1 Warum Planung statt reiner Suche?

Man *könnte* jedes Planungsproblem als Suchproblem (Modul 06) auffassen. Das
Problem: Bei einer Welt mit vielen Objekten und Prädikaten ist der Zustandsraum
astronomisch, und eine reine Zustandssuche „sieht" die **Struktur** der Aktionen
nicht. **Planung** nutzt eine **faktorisierte, logik-nahe Repräsentation** von
Zuständen und Aktionen. Dadurch kann man (a) **domänenunabhängige Heuristiken**
automatisch aus der Aktionsbeschreibung ableiten und (b) Aktionen ausnutzen, die
unabhängig voneinander sind.

### 1.2 STRIPS und PDDL

Im **STRIPS**-Formalismus (Stanford Research Institute Problem Solver) ist ein
**Zustand** eine Menge von **Grundfluenten** (variablenfreie, wahre atomare
Aussagen) unter der **Closed-World-Assumption**: Was nicht in der Menge steht,
gilt als falsch. Ein **Planungsproblem** ist $(\mathcal{F}, s_0, g, \mathcal{A})$ mit

- $\mathcal{F}$: Menge aller Fluenten,
- $s_0 \subseteq \mathcal{F}$: **Startzustand**,
- $g \subseteq \mathcal{F}$: **Zielbedingung** (ein Zustand $s$ erfüllt das Ziel, wenn $g \subseteq s$),
- $\mathcal{A}$: Menge der **Aktionsschemata**. Eine Grundaktion $a$ besteht aus
  - $\mathrm{PRE}(a)$ — **Vorbedingungen** (Fluenten, die gelten müssen),
  - $\mathrm{ADD}(a)$ — **Add-Liste** (Fluenten, die $a$ wahr macht),
  - $\mathrm{DEL}(a)$ — **Delete-Liste** (Fluenten, die $a$ falsch macht).

Eine Aktion $a$ ist in $s$ **anwendbar**, wenn $\mathrm{PRE}(a) \subseteq s$. Das
**Übergangsmodell** (Progression) ist rein mengentheoretisch:
$$
\mathrm{Result}(s, a) = (s \setminus \mathrm{DEL}(a)) \cup \mathrm{ADD}(a).
$$

**PDDL** (Planning Domain Definition Language) ist die heute übliche Syntax dafür;
sie trennt eine **Domänendatei** (Prädikate, Aktionsschemata mit Variablen) von
einer **Problemdatei** (Objekte, Start, Ziel). Aktionsschemata mit Variablen
werden vor der Suche zu Grundaktionen instanziiert (*grounding*).

> **Beispiel (Blocksworld).** Fluenten: `On(x,y)`, `OnTable(x)`, `Clear(x)`,
> `Holding(x)`, `ArmEmpty`. Aktion `PickUp(x)`:
> $\mathrm{PRE} = \{\mathrm{Clear}(x), \mathrm{OnTable}(x), \mathrm{ArmEmpty}\}$,
> $\mathrm{ADD} = \{\mathrm{Holding}(x)\}$,
> $\mathrm{DEL} = \{\mathrm{Clear}(x), \mathrm{OnTable}(x), \mathrm{ArmEmpty}\}$.

Der Charme: Die **Frame-Problematik** (was ändert sich *nicht*?) ist elegant
gelöst — alles, was nicht in ADD/DEL steht, bleibt unverändert.

### 1.3 Vorwärts-, Rückwärts- und Plan-Raum-Suche

**Progression (Vorwärtssuche)** durchsucht den Zustandsraum von $s_0$ Richtung
Ziel mit den Verfahren aus Modul 06 (meist A\* mit einer der Heuristiken unten).
Vorteil: einfache Zustände; Nachteil: hoher Verzweigungsfaktor (viele anwendbare
Aktionen).

**Regression (Rückwärtssuche)** startet beim Ziel $g$ und arbeitet rückwärts.
Der Regressionsschritt: Um Zustandsbeschreibung $g'$ nach Anwendung von $a$ zu
erreichen, muss vorher gelten
$$
\mathrm{Regress}(g', a) = (g' \setminus \mathrm{ADD}(a)) \cup \mathrm{PRE}(a),
\quad \text{sofern } a \text{ relevant ist } (\mathrm{ADD}(a)\cap g' \neq \emptyset)
\text{ und konsistent } (\mathrm{DEL}(a) \cap g' = \emptyset).
$$
Vorteil: nur **relevante** Aktionen werden betrachtet (kleiner Verzweigungsfaktor);
Nachteil: Zustände sind *Teilbeschreibungen* (Mengen von Bedingungen), gutes
Heuristik-Design ist schwerer.

**Plan-Raum-Suche / Partial-Order Planning (POP).** Statt im Zustandsraum sucht
man im Raum *partieller Pläne*. Ein partieller Plan besteht aus (i) einer Menge
von Aktionen, (ii) **Ordnungsbeschränkungen** $a \prec b$ („$a$ vor $b$"),
(iii) **kausalen Links** $a \xrightarrow{p} b$ („$a$ stellt Vorbedingung $p$ von
$b$ bereit") und (iv) offenen Vorbedingungen. Man verfeinert den Plan, bis keine
offene Vorbedingung mehr existiert und alle **Bedrohungen (threats)** aufgelöst
sind. Eine Aktion $c$ **bedroht** einen Link $a \xrightarrow{p} b$, wenn $c$ das
$p$ löscht und zwischen $a$ und $b$ liegen könnte; man löst das durch
**Promotion** ($c \prec a$) oder **Demotion** ($b \prec c$). POP erzeugt **partiell
geordnete Pläne** — es committet sich nicht vorschnell auf eine Reihenfolge
unabhängiger Aktionen (**least commitment**).

### 1.4 Planungsheuristiken aus Relaxation

Der Durchbruch der modernen Planung: **domänenunabhängige** Heuristiken, die
automatisch aus der STRIPS-Beschreibung entstehen — analog zur Relaxation in
Modul 06, nur systematisch.

**Delete-Relaxation.** Man streicht *alle* Delete-Listen: $\mathrm{DEL}(a) =
\emptyset$. In der relaxierten Welt kann eine einmal erreichte Eigenschaft nie
wieder verloren gehen (monotoner Fluenten-Zuwachs), das relaxierte Problem ist in
**Polynomzeit** lösbar. Aus ihm gewinnt man:

- $h_{\text{add}}(s) = \sum_{p \in g}\, \Delta(p)$ — Summe der geschätzten Kosten,
  jedes Ziel-Fluent einzeln zu erreichen (überschätzt bei Interaktion → **nicht
  zulässig**, aber informativ).
- $h_{\max}(s) = \max_{p \in g}\, \Delta(p)$ — teuerstes einzelnes Ziel-Fluent
  (**zulässig**, aber oft schwach).
- $h_{\text{FF}}$ (Fast Forward): extrahiert einen konkreten **relaxierten Plan**
  aus dem relaxierten Planungsgraphen und nimmt dessen Länge — meist die beste
  praktische Heuristik (nicht zulässig, aber sehr treffsicher).

Dabei ist $\Delta(p)$ rekursiv definiert: $\Delta(p)=0$, wenn $p\in s$, sonst
$\Delta(p) = \min_{a:\, p\in \mathrm{ADD}(a)} \big(\mathrm{cost}(a) + \text{Kombi}_{q\in \mathrm{PRE}(a)}\Delta(q)\big)$,
wobei $\text{Kombi}=\sum$ für $h_{\text{add}}$ und $\text{Kombi}=\max$ für $h_{\max}$.

### 1.5 GraphPlan und SATPlan

**GraphPlan** baut einen geschichteten **Planungsgraphen** aus abwechselnden
**Zustandsebenen** ($S_0, S_1, \dots$) und **Aktionsebenen** ($A_0, A_1, \dots$)
auf. Jede Ebene enthält *alle* Fluenten/Aktionen, die *möglicherweise* erreichbar
sind, plus **Mutex-Relationen** (mutual exclusion):
- Zwei **Aktionen** sind mutex, wenn eine die Vorbedingung/Wirkung der anderen
  löscht (*inconsistent effects*, *interference*) oder ihre Vorbedingungen mutex
  sind (*competing needs*).
- Zwei **Fluenten** sind mutex, wenn jede sie erzeugende Aktion mutex ist
  (*inconsistent support*).

Der Graph wächst, bis alle Ziel-Fluenten nicht-mutex in einer Ebene erscheinen
(*level off*); dann extrahiert eine Rückwärts-Suche einen Plan. Der Graph liefert
zugleich die zulässige Heuristik „erste Ebene, in der ein Ziel-Fluent auftaucht".

**SATPlan.** Kodiere „existiert ein Plan der Länge $\le k$?" als **aussagenlogische
Formel** (Fluenten und Aktionen mit Zeitindex, Vorbedingungs-/Wirkungs- und
Frame-Axiome) und wirf einen **SAT-Solver** (Modul 06, DPLL/CDCL) darauf; erhöhe
$k$ iterativ. Zeigt eindrucksvoll, wie Modul-06-Technik hier wiederverwendet wird.

---

## Teil 2 — Aufbau: Schließen unter Unsicherheit

### 2.1 Warum Wahrscheinlichkeit?

Reale Agenten kennen die Welt nicht sicher: Sensoren rauschen, Aktionen misslingen,
Wissen ist unvollständig. Rein logisches Schließen scheitert, weil man alle
Ausnahmen aufzählen müsste („qualification problem"). **Wahrscheinlichkeit** fasst
diese Unsicherheit in *einer Zahl* pro Aussage — als **Grad der Überzeugung**
(bayesianische Sicht), nicht notwendig als Häufigkeit. Der Satz von de Finetti
zeigt: Wer Wetten abschließt, deren Quoten *nicht* den Wahrscheinlichkeitsaxiomen
gehorchen, ist durch ein **Dutch Book** garantiert ausbeutbar — Rationalität
*erzwingt* die Wahrscheinlichkeitsrechnung.

### 2.2 Grundlagen: Axiome, Bedingtheit, Bayes

Für Ereignisse gelten die **Kolmogorov-Axiome**:
$$
0 \le P(a) \le 1, \quad P(\text{wahr}) = 1, \quad
P(a \lor b) = P(a) + P(b) - P(a \land b).
$$
Die **Verbundverteilung (joint distribution)** $P(X_1, \dots, X_n)$ über alle
Variablen legt *alles* fest. Aus ihr gewinnt man jede Frage durch:
- **Marginalisierung (summing out):** $P(\mathbf{Y}) = \sum_{\mathbf{z}} P(\mathbf{Y}, \mathbf{z})$,
- **Konditionierung:** $\displaystyle P(a \mid b) = \frac{P(a \land b)}{P(b)}$ (für $P(b) > 0$).

Umstellen der Definition liefert die **Produktregel** $P(a\land b) = P(a\mid b)\,P(b)$
und, durch Gleichsetzen, den **Satz von Bayes**:
$$
\boxed{\,P(h \mid e) = \frac{P(e \mid h)\,P(h)}{P(e)}\,}
$$
das Arbeitspferd: aus dem **Likelihood** $P(e\mid h)$ (wie wahrscheinlich ist die
Evidenz unter der Hypothese) und dem **Prior** $P(h)$ wird der **Posterior**
$P(h\mid e)$. Der Nenner $P(e) = \sum_{h'} P(e\mid h')P(h')$ ist die
**Normierungskonstante**; oft schreibt man $P(h\mid e) = \alpha\, P(e\mid h)P(h)$.

> **Durchgerechnet (medizinischer Test).** Krankheit mit Prävalenz $P(D)=0{,}01$;
> Test mit Sensitivität $P(+\mid D)=0{,}9$ und Falsch-Positiv-Rate $P(+\mid \lnot D)=0{,}09$.
> Positiver Test — wie wahrscheinlich krank?
> $$P(D\mid +) = \frac{0{,}9 \cdot 0{,}01}{0{,}9\cdot 0{,}01 + 0{,}09\cdot 0{,}99}
> = \frac{0{,}009}{0{,}009 + 0{,}0891} \approx 0{,}092.$$
> Nur ~9\,%! Die niedrige **Basisrate** dominiert — der klassische *base rate
> fallacy*. Ein Musterbeispiel, warum man Priors nicht ignorieren darf.

**(Bedingte) Unabhängigkeit.** $X$ und $Y$ sind **unabhängig**, wenn
$P(X,Y)=P(X)P(Y)$. Wichtiger noch ist **bedingte Unabhängigkeit**: $X \perp Y \mid Z$
gdw. $P(X,Y\mid Z) = P(X\mid Z)\,P(Y\mid Z)$. Sie ist der Schlüssel, der die
exponentiell große Verbundverteilung **kompakt faktorisierbar** macht.

### 2.3 Bayes-Netze: Struktur und Semantik

Eine volle Verbundverteilung über $n$ booleschen Variablen braucht $2^n - 1$
Zahlen — unhandhabbar. Ein **Bayes-Netz (Bayesian Network)** nutzt bedingte
Unabhängigkeiten, um dieselbe Verteilung *kompakt* darzustellen:

- ein **gerichteter azyklischer Graph (DAG)**, Knoten = Zufallsvariablen, Kante
  $X \to Y$ = „$X$ beeinflusst $Y$ direkt",
- pro Knoten eine **bedingte Wahrscheinlichkeitstabelle (CPT)** $P(X_i \mid \mathrm{Parents}(X_i))$.

**Die zentrale Semantik (Kettenregel für Bayes-Netze):** Das Netz repräsentiert
die Verbundverteilung als **Produkt der lokalen CPTs**:
$$
\boxed{\,P(x_1, \dots, x_n) = \prod_{i=1}^{n} P\big(x_i \mid \mathrm{parents}(x_i)\big)\,}
$$
Das gilt genau dann, wenn jede Variable **bedingt unabhängig von ihren
Nicht-Nachkommen gegeben ihre Eltern** ist (die *lokale Markov-Bedingung*). Bei
begrenzter Elternzahl $k$ schrumpft der Speicher von $2^n$ auf $n\cdot 2^k$ —
oft von astronomisch auf handhabbar.

> **Klassiker (Alarm-Netz, Pearl).** Ein Einbruch ($B$) oder ein Erdbeben ($E$)
> kann einen Alarm ($A$) auslösen; John ($J$) und Mary ($M$) rufen an, je nachdem,
> ob sie den Alarm hören. Struktur: $B\to A \leftarrow E$, $A\to J$, $A\to M$.
> Verbund: $P(B,E,A,J,M) = P(B)P(E)P(A\mid B,E)P(J\mid A)P(M\mid A)$ — fünf kleine
> CPTs statt einer 32-Zeilen-Tabelle.

### 2.4 d-Separation — Unabhängigkeit aus der Graphstruktur ablesen

Wann gilt $X \perp Y \mid Z$ *allein aufgrund der Struktur*? Das beantwortet
**d-Separation**. Betrachte jeden ungerichteten Pfad zwischen $X$ und $Y$; der
Pfad ist **blockiert**, wenn er einen Knoten $n$ folgender Art enthält:

1. **Kette** $\to n \to$ oder **Gabel** $\leftarrow n \to$, und $n \in Z$
   (beobachtet) → blockiert.
2. **Kollider** $\to n \leftarrow$ („v-Struktur"), und **weder $n$ noch ein
   Nachkomme von $n$** ist in $Z$ → blockiert.

Sind *alle* Pfade blockiert, gilt $X \perp Y \mid Z$. Die Kollider-Regel ist der
subtile Punkt: Ein unbeobachteter Kollider *blockiert*, aber **Beobachtung des
Kolliders (oder eines Nachkommen) öffnet** ihn — das ist das **„explaining away"**:
Erdbeben und Einbruch sind a priori unabhängig, aber *gegeben den Alarm* werden
sie abhängig (hört man vom Erdbeben, sinkt die Einbruchswahrscheinlichkeit).

### 2.5 Exakte Inferenz

Die **Inferenzaufgabe**: Berechne $P(\mathbf{X}_{\text{query}} \mid \mathbf{e})$
für Anfragevariablen gegeben Evidenz $\mathbf{e}$.

**Inferenz durch Aufzählung.** Direkt aus der Verbund-Faktorisierung:
$$
P(X \mid \mathbf{e}) = \alpha \sum_{\mathbf{y}} P(X, \mathbf{e}, \mathbf{y}),
$$
wobei $\mathbf{y}$ die *versteckten* Variablen sind und das Produkt der CPTs
eingesetzt wird. Korrekt, aber $O(2^n)$ — die naive Summe wiederholt Teilprodukte.

**Variable Elimination (VE).** Beschleunigt die Aufzählung durch **Ausklammern**
(Distributivgesetz) und **Zwischenspeichern**. Man arbeitet mit **Faktoren**
(mehrdimensionale Tabellen). Zwei Operationen:
- **punktweises Produkt** zweier Faktoren $f_1 \times f_2$,
- **Ausmarginalisieren (summing out)** einer Variablen: $\sum_x f(\dots, x, \dots)$.

Algorithmus: Wähle eine **Eliminationsreihenfolge** der versteckten Variablen; für
jede: multipliziere alle Faktoren, die sie enthalten, und summiere sie heraus.
$$
P(B\mid j,m) = \alpha\, P(B) \sum_e P(e) \sum_a P(a\mid B,e)\,P(j\mid a)\,P(m\mid a).
$$
VE ist dramatisch schneller als Aufzählung, aber die Kosten hängen stark von der
Reihenfolge ab (der größte Zwischenfaktor bestimmt sie — die **Baumweite** des
Graphen). **Allgemeine Bayes-Netz-Inferenz ist NP-schwer**; für Netze mit kleiner
Baumweite (z. B. Polybäume) ist sie polynomiell.

### 2.6 Approximative Inferenz durch Sampling

Wenn exakte Inferenz zu teuer ist, **schätzt** man $P(\mathbf{X}\mid\mathbf{e})$ aus
Stichproben.

- **Direct/Prior Sampling:** Ziehe Werte topologisch geordnet gemäß den CPTs →
  Stichproben aus der Verbundverteilung.
- **Rejection Sampling:** Wie oben, aber verwirf alle Stichproben, die $\mathbf{e}$
  widersprechen. Korrekt, aber verschwenderisch bei seltener Evidenz.
- **Likelihood Weighting:** Fixiere die Evidenzvariablen auf ihre beobachteten
  Werte und **gewichte** jede Stichprobe mit dem Produkt der Evidenz-Likelihoods
  $\prod_{e_i} P(e_i \mid \mathrm{parents}(e_i))$. Verwirft nichts, effizienter.
- **Gibbs-Sampling (MCMC):** Ein **Markov-Chain-Monte-Carlo**-Verfahren. Fixiere
  Evidenz, initialisiere die übrigen Variablen beliebig; resample dann wiederholt
  *eine* Nicht-Evidenz-Variable aus ihrer Verteilung gegeben ihre **Markov-Decke**
  (Eltern, Kinder, Ko-Eltern). Die so erzeugte Kette hat die Posterior-Verteilung
  als **stationäre Verteilung**; Stichprobenmittel konvergieren gegen $P(\mathbf{X}\mid\mathbf{e})$.

Alle Sampling-Verfahren sind **konsistent** (Fehler $\to 0$ mit $N\to\infty$, Rate
$O(1/\sqrt N)$), tauschen also Exaktheit gegen Rechenzeit — der übliche Deal bei
schwierigen Netzen.

### 2.7 Temporale Modelle: Markov-Ketten und HMMs

Die Welt ändert sich über die Zeit. Ein **zeitdiskretes** Modell hat pro Schritt
$t$ Zustandsvariablen $\mathbf{X}_t$ und Evidenzvariablen $\mathbf{E}_t$. Zwei
Annahmen machen es handhabbar:

- **Markov-Annahme (erster Ordnung):** $P(\mathbf{X}_t \mid \mathbf{X}_{0:t-1}) =
  P(\mathbf{X}_t \mid \mathbf{X}_{t-1})$ — die Zukunft hängt nur über den
  **aktuellen** Zustand von der Vergangenheit ab. → **Übergangsmodell**.
- **Sensor-Markov-Annahme:** $P(\mathbf{E}_t \mid \mathbf{X}_{0:t}, \mathbf{E}_{0:t-1})
  = P(\mathbf{E}_t \mid \mathbf{X}_t)$. → **Sensormodell**.

Ein **Hidden Markov Model (HMM)** hat eine einzelne diskrete Zustandsvariable, die
man nicht direkt sieht (*hidden*), nur über Evidenz. Vier Standardaufgaben:

**Filtering** — $P(\mathbf{X}_t \mid \mathbf{e}_{1:t})$ (aktueller Zustand gegeben
alle bisherigen Beobachtungen). Rekursiv (**Forward-Algorithmus**):
$$
P(\mathbf{X}_{t+1}\mid \mathbf{e}_{1:t+1}) = \alpha\, \underbrace{P(\mathbf{e}_{t+1}\mid \mathbf{X}_{t+1})}_{\text{Update (Sensor)}}
\sum_{\mathbf{x}_t} \underbrace{P(\mathbf{X}_{t+1}\mid \mathbf{x}_t)}_{\text{Predict (Transition)}} P(\mathbf{x}_t\mid \mathbf{e}_{1:t}).
$$
„Predict-Update"-Schleife — dieselbe Idee steckt im **Kalman-Filter** (der
stetig-gaußsche Spezialfall) und in der Lokalisierung von Robotern (Modul 21).

**Prediction** — $P(\mathbf{X}_{t+k}\mid\mathbf{e}_{1:t})$ (Zukunft ohne neue Evidenz).

**Smoothing** — $P(\mathbf{X}_k\mid\mathbf{e}_{1:t})$ für $k<t$ (Vergangenheit mit
Rückschau). Der **Forward-Backward-Algorithmus** kombiniert die Vorwärtsnachricht
mit einer rückwärts laufenden Nachricht $b_{k+1:t} = P(\mathbf{e}_{k+1:t}\mid\mathbf{X}_k)$.

**Wahrscheinlichste Erklärung** — $\arg\max_{\mathbf{x}_{1:t}} P(\mathbf{x}_{1:t}\mid\mathbf{e}_{1:t})$.
Der **Viterbi-Algorithmus** ist dynamische Programmierung: Er ersetzt in der
Vorwärtsrekursion die Summe durch ein **Maximum** und merkt sich Rückzeiger, um den
besten Pfad zu rekonstruieren. Basis von Spracherkennung, POS-Tagging (Modul 08),
Bioinformatik-Sequenzanalyse (Modul 28).

---

## Teil 3 — Advanced: Rationale Entscheidungen und MDPs

### 3.1 Nutzentheorie und Maximum Expected Utility

Bisher haben wir *geschlossen*, jetzt *handeln* wir. Die **Nutzentheorie** (von
Neumann & Morgenstern) zeigt: Erfüllt eine Präferenzrelation über unsichere
Ausgänge („Lotterien") sechs **Rationalitätsaxiome** (Vollständigkeit,
Transitivität, Stetigkeit, Substituierbarkeit, Monotonie, Zerlegbarkeit), dann
**existiert eine Nutzenfunktion** $U$, sodass der Agent Lotterie $L_1$ genau dann
$L_2$ vorzieht, wenn $\mathrm{EU}(L_1) > \mathrm{EU}(L_2)$, mit dem **erwarteten
Nutzen**
$$
\mathrm{EU}(a\mid \mathbf{e}) = \sum_{s'} P(\mathrm{Result}(a) = s' \mid a, \mathbf{e})\; U(s').
$$
Das **MEU-Prinzip (Maximum Expected Utility):** Ein rationaler Agent wählt die
Aktion, die den erwarteten Nutzen maximiert:
$a^\ast = \arg\max_a \mathrm{EU}(a\mid\mathbf{e})$. Wichtig: Nutzen ist **nicht**
gleich Geld — die typischerweise **konkave** Nutzenfunktion für Geld erklärt
**Risikoaversion** (eine sichere 100 € kann mehr Nutzen haben als eine 50/50-Chance
auf 0/220 €).

### 3.2 Entscheidungsnetze und der Wert von Information

**Entscheidungsnetze (Influence Diagrams)** erweitern Bayes-Netze um
**Entscheidungsknoten** (Aktionen, die der Agent wählt) und einen **Nutzenknoten**.
Die Auswertung wählt die Aktionen mit maximalem erwarteten Nutzen.

**Value of Perfect Information (VPI).** Lohnt es sich, *vor* der Entscheidung eine
Variable $E_j$ zu messen? Der Informationswert ist die erwartete Nutzensteigerung:
$$
\mathrm{VPI}_{\mathbf{e}}(E_j) = \Big(\sum_{e_{j}} P(e_{j}\mid\mathbf{e})\;
\mathrm{EU}(a^\ast_{e_{j}} \mid \mathbf{e}, e_{j})\Big) - \mathrm{EU}(a^\ast \mid \mathbf{e}).
$$
VPI ist **nie negativ** (mehr Wissen kann im Erwartungswert nicht schaden) und
**nicht additiv**. Es liefert die theoretische Grundlage für *rationale
Informationsbeschaffung* — welchen Sensor/Test man ansteuert.

### 3.3 Markov-Entscheidungsprozesse (MDPs)

Jetzt der **sequenzielle** Fall: Entscheidungen über viele Schritte, mit
unsicheren Ausgängen. Ein **MDP** ist $(S, A, P, R, \gamma)$:

- $S$ Zustände, $A$ Aktionen,
- $P(s' \mid s, a)$ **Übergangsmodell** (stochastisch! — hier der Bruch mit der
  klassischen Planung),
- $R(s)$ (oder $R(s,a,s')$) **Belohnung**,
- $\gamma \in [0,1)$ **Diskontfaktor** (spätere Belohnungen zählen weniger; sichert
  auch Konvergenz bei unendlichem Horizont).

Gesucht ist eine **Policy** $\pi: S \to A$, die den erwarteten **diskontierten
Return** $\mathbb{E}\big[\sum_{t=0}^{\infty}\gamma^t R(s_t)\big]$ maximiert. Der
**Wert** eines Zustands unter $\pi$ ist $V^\pi(s) = \mathbb{E}\big[\sum_t \gamma^t
R(s_t) \mid s_0=s, \pi\big]$.

**Die Bellman-Gleichung** charakterisiert $V^\pi$ selbstkonsistent:
$$
V^\pi(s) = R(s) + \gamma \sum_{s'} P(s'\mid s, \pi(s))\, V^\pi(s').
$$
Für die **optimale** Policy $\pi^\ast$ gilt die **Bellman-Optimalitätsgleichung**:
$$
\boxed{\,V^\ast(s) = R(s) + \gamma \max_{a} \sum_{s'} P(s'\mid s, a)\, V^\ast(s')\,}
$$
und die optimale Policy liest man greedy ab:
$\pi^\ast(s) = \arg\max_a \sum_{s'} P(s'\mid s,a)\,V^\ast(s')$.

### 3.4 Value Iteration und Policy Iteration

**Value Iteration.** Fasse die Bellman-Optimalitätsgleichung als **Update** auf und
iteriere bis zur Konvergenz:
$$
V_{k+1}(s) \leftarrow R(s) + \gamma \max_a \sum_{s'} P(s'\mid s,a)\, V_k(s).
$$

> **Warum konvergiert das? (Kontraktionsbeweis.)** Der **Bellman-Optimalitäts­operator**
> $B$, definiert durch $(BV)(s) = R(s) + \gamma\max_a\sum_{s'}P(s'\mid s,a)V(s')$,
> ist eine **Kontraktion** bzgl. der Maximumsnorm $\lVert V\rVert_\infty = \max_s|V(s)|$
> mit Faktor $\gamma$: für beliebige $V, V'$ gilt
> $$\lVert BV - BV'\rVert_\infty \le \gamma\,\lVert V - V'\rVert_\infty.$$
> *Beweisskizze:* $|\max_a f(a) - \max_a g(a)| \le \max_a|f(a)-g(a)|$, und
> $\gamma\sum_{s'}P(s'\mid s,a)|V(s')-V'(s')| \le \gamma\lVert V-V'\rVert_\infty$, da
> $\sum_{s'}P=1$. Nach dem **Banachschen Fixpunktsatz** hat $B$ einen **eindeutigen
> Fixpunkt** $V^\ast$, und $V_k \to V^\ast$ **geometrisch**: $\lVert V_k - V^\ast\rVert_\infty
> \le \gamma^k \lVert V_0 - V^\ast\rVert_\infty$. Für $\gamma<1$ garantiert das
> Konvergenz — je näher $\gamma$ an 1, desto langsamer. $\quad\blacksquare$

**Policy Iteration.** Alterniert zwei Schritte bis zur Stabilität der Policy:
1. **Policy Evaluation:** Löse $V^{\pi}(s) = R(s) + \gamma\sum_{s'}P(s'\mid s,\pi(s))V^\pi(s')$
   — ein **lineares Gleichungssystem** in $|S|$ Unbekannten (exakt lösbar oder
   iterativ genähert).
2. **Policy Improvement:** Setze
   $\pi'(s) \leftarrow \arg\max_a \sum_{s'}P(s'\mid s,a)V^\pi(s')$.

Ändert sich die Policy nicht mehr, ist sie optimal. Policy Iteration konvergiert in
**endlich vielen** Schritten (es gibt nur endlich viele Policies, und jeder Schritt
verbessert strikt oder terminiert) — oft in *sehr wenigen* Iterationen, dafür ist
jede teurer als ein Value-Iteration-Schritt. Beide sind Spezialfälle der
**generalisierten Policy-Iteration**, dem konzeptionellen Kern des
Reinforcement Learning (Modul 13): Der Unterschied dort ist nur, dass $P$ und $R$
**unbekannt** sind und aus Erfahrung gelernt werden müssen.

### 3.5 Partielle Beobachtbarkeit (POMDPs) — Ausblick

Kann der Agent den Zustand nicht direkt beobachten (nur verrauschte Sensoren),
wird das MDP zum **POMDP**. Der Trick: Der Agent hält einen **belief state** $b(s)$
(eine Wahrscheinlichkeitsverteilung über Zustände, per Filtering aus Abschnitt 2.7
aktualisiert) und löst ein MDP im *kontinuierlichen* Belief-Raum. POMDPs sind
theoretisch elegant, aber exakte Lösung ist **PSPACE-hart** — praktisch nutzt man
Approximationen. Sie verbinden Filtering (Teil 2) mit MDPs (Teil 3) zum
vollständigen Bild des rationalen Agenten unter Unsicherheit.

### 3.6 Ausblick: Nichtmonotones Schließen und Beschreibungslogiken

Zwei weitere Antworten auf Unsicherheit — nicht probabilistisch, sondern
**qualitativ**:

- **Nichtmonotones Schließen.** Klassische Logik ist *monoton*: Mehr Prämissen →
  nie weniger Schlüsse. Alltagsschlüsse sind aber **defeasible** („Vögel fliegen —
  aber nicht Pinguine"). **Default-Logik**, **Circumscription** und
  **Answer-Set-Programming** formalisieren solche Standardannahmen, die neue
  Information *zurücknehmen* kann. Grundlage der Wissensrepräsentation und von
  Logikprogrammierung mit Negation (Modul 33).
- **Beschreibungslogiken (Description Logics, DL).** Das entscheidbare
  FOL-Fragment hinter **Ontologien** und dem **Semantic Web** (OWL). Eine
  **TBox** definiert Konzepte/Rollen ($\text{Vater} \equiv \text{Mann} \sqcap
  \exists\text{hatKind}.\top$), eine **ABox** enthält Instanzfakten. Kern-Inferenzen
  (Subsumption, Instanz-Check) sind entscheidbar — der bewusste Tausch
  Ausdrucksstärke gegen Entscheidbarkeit, den Modul 06 (Abschnitt 4.5) schon
  angekündigt hat. DL-Reasoner (z. B. via Tableau-Verfahren) sind die praktische
  Fortsetzung des Theorembeweisens aus Modul 06.

---

## Zusammenfassung / Cheat-Sheet

**Planung**

| Begriff | Kern |
|---|---|
| STRIPS-Aktion | $\langle\mathrm{PRE},\mathrm{ADD},\mathrm{DEL}\rangle$; anwendbar wenn $\mathrm{PRE}\subseteq s$ |
| Progression | $\mathrm{Result}(s,a) = (s\setminus\mathrm{DEL})\cup\mathrm{ADD}$ |
| Regression | $(g'\setminus\mathrm{ADD})\cup\mathrm{PRE}$, relevant + konsistent |
| POP | partielle Ordnung + kausale Links; Bedrohungen via Promotion/Demotion |
| Relaxation | Delete-Listen streichen → $h_{\max}$ (zulässig), $h_{\text{add}}$, $h_{\text{FF}}$ |
| GraphPlan | Planungsgraph + Mutex; SATPlan: Plan als SAT kodieren |

**Wahrscheinlichkeit & Bayes-Netze**

| Begriff | Kern |
|---|---|
| Bayes | $P(h\mid e) = \dfrac{P(e\mid h)P(h)}{P(e)} = \alpha\,P(e\mid h)P(h)$ |
| bedingte Unabh. | $X\perp Y\mid Z \iff P(X,Y\mid Z)=P(X\mid Z)P(Y\mid Z)$ |
| BN-Faktorisierung | $P(x_1..x_n) = \prod_i P(x_i\mid\mathrm{parents}(x_i))$ |
| d-Separation | Kette/Gabel blockiert wenn beobachtet; Kollider blockiert wenn **un**beobachtet |
| explaining away | Kollider öffnet bei Beobachtung → Eltern werden abhängig |
| Enumeration | $P(X\mid\mathbf e)=\alpha\sum_{\mathbf y}\prod_i P(x_i\mid\mathrm{parents})$ |
| Variable Elim. | Faktoren: punktweises Produkt + summing out; Kosten ~ Baumweite; BN-Inferenz NP-schwer |
| Sampling | prior/rejection/**likelihood weighting**/**Gibbs (MCMC)**; konsistent, $O(1/\sqrt N)$ |

**Temporale Modelle**

| Begriff | Kern |
|---|---|
| Markov 1. Ordn. | $P(\mathbf X_t\mid\mathbf X_{0:t-1})=P(\mathbf X_t\mid\mathbf X_{t-1})$ |
| Filtering (Forward) | $\alpha\,P(\mathbf e_{t+1}\mid\mathbf X_{t+1})\sum_{\mathbf x_t}P(\mathbf X_{t+1}\mid\mathbf x_t)f_t$ |
| Smoothing | Forward-Backward (Vorwärts- × Rückwärtsnachricht) |
| Viterbi | wie Filtering, aber $\max$ statt $\sum$ + Rückzeiger; beste Zustandsfolge |

**Entscheidungen & MDPs**

| Begriff | Kern |
|---|---|
| MEU | $a^\ast=\arg\max_a\sum_{s'}P(s'\mid a,\mathbf e)U(s')$ |
| VPI | erwartete Nutzensteigerung durch Messung; $\ge 0$, nicht additiv |
| MDP | $(S,A,P,R,\gamma)$; maximiere $\mathbb E[\sum_t\gamma^t R]$ |
| Bellman opt. | $V^\ast(s)=R(s)+\gamma\max_a\sum_{s'}P(s'\mid s,a)V^\ast(s')$ |
| Value Iteration | Bellman-Update iterieren; $B$ ist $\gamma$-Kontraktion → $V_k\to V^\ast$ geometrisch |
| Policy Iteration | Eval (lin. System) + Improvement; terminiert in endlich vielen Schritten |

---

## Selbsttest

<details><summary><b>1. Was ist der Unterschied zwischen Progression und Regression in STRIPS, und wann bevorzugt man welche?</b></summary>

*Progression* sucht vorwärts von $s_0$: Zustände sind vollständige Fluentenmengen,
$\mathrm{Result}(s,a)=(s\setminus\mathrm{DEL})\cup\mathrm{ADD}$. Verzweigungsfaktor
hoch (alle anwendbaren Aktionen), aber Zustände konkret und gut heuristisch
bewertbar — deshalb in der Praxis (mit $h_{\text{FF}}$ o. Ä.) meist der Sieger.
*Regression* sucht rückwärts vom Ziel: Zustände sind Teilbeschreibungen,
$\mathrm{Regress}(g',a)=(g'\setminus\mathrm{ADD})\cup\mathrm{PRE}$ für relevante,
konsistente $a$. Kleiner Verzweigungsfaktor (nur relevante Aktionen), aber
Heuristiken schwerer. Regression lohnt bei wenigen zielrelevanten Aktionen.
</details>

<details><summary><b>2. Warum ist die Delete-Relaxation nützlich, obwohl $h_{\text{add}}$ nicht zulässig ist?</b></summary>

Streicht man alle Delete-Listen, wächst die Fluentenmenge monoton, und das
relaxierte Problem ist in Polynomzeit lösbar — man bekommt also *billig* eine
Schätzung. $h_{\max}$ (Maximum) ist sogar zulässig, aber schwach. $h_{\text{add}}$
(Summe) überschätzt, weil es Ziel-Fluenten als unabhängig behandelt und geteilte
Teilpläne doppelt zählt — dafür ist es viel informativer und lenkt die Suche gut.
$h_{\text{FF}}$ extrahiert einen echten relaxierten Plan und ist meist am besten.
In der Praxis zählt Informativität oft mehr als strikte Zulässigkeit (solange man
nicht Optimalität garantieren muss).
</details>

<details><summary><b>3. Ein Test ist zu 90 % sensitiv und hat 9 % Falsch-Positive; die Krankheit hat 1 % Prävalenz. Warum ist $P(\text{krank}\mid+)$ nur ~9 %?</b></summary>

Bayes: $P(D\mid+)=\frac{0{,}9\cdot0{,}01}{0{,}9\cdot0{,}01+0{,}09\cdot0{,}99}\approx0{,}092$.
Der Grund ist die niedrige **Basisrate**: Es gibt 99-mal so viele Gesunde wie
Kranke. Selbst bei nur 9 % Falsch-Positiven erzeugen die vielen Gesunden
($0{,}09\cdot0{,}99\approx0{,}089$) fast zehnmal so viele positive Tests wie die
wenigen echten Kranken ($0{,}9\cdot0{,}01=0{,}009$). Der Prior darf nie ignoriert
werden (*base rate fallacy*).
</details>

<details><summary><b>4. Erkläre die Bayes-Netz-Faktorisierung und warum sie Speicher spart.</b></summary>

Ein BN behauptet $P(x_1,\dots,x_n)=\prod_i P(x_i\mid\mathrm{parents}(x_i))$. Das
folgt aus der lokalen Markov-Bedingung (jede Variable ist bedingt unabhängig von
ihren Nicht-Nachkommen gegeben ihre Eltern). Statt der vollen Verbundtabelle mit
$2^n-1$ Einträgen speichert man pro Knoten nur $2^k$ Zahlen ($k$ = Elternzahl),
insgesamt $n\cdot2^k$. Bei begrenztem $k$ ist das linear statt exponentiell in $n$
— der ganze Sinn von Bayes-Netzen.
</details>

<details><summary><b>5. Was ist „explaining away"? Formuliere es mit d-Separation.</b></summary>

An einem Kollider $A\to C\leftarrow B$ sind $A$ und $B$ **a priori unabhängig**
(der unbeobachtete Kollider blockiert den Pfad). Beobachtet man $C$ (oder einen
Nachkommen), **öffnet** sich der Pfad: $A$ und $B$ werden *bedingt abhängig*. Wenn
$C$ eingetreten ist und man erfährt, dass $A$ es erklärt, sinkt die
Wahrscheinlichkeit von $B$ — die eine Ursache „erklärt die andere weg". Beispiel:
Alarm ($C$) durch Einbruch ($A$) oder Erdbeben ($B$); Erdbebenmeldung senkt
$P(\text{Einbruch}\mid\text{Alarm})$.
</details>

<details><summary><b>6. Wieso ist Variable Elimination schneller als Inferenz durch Aufzählung, und wovon hängen die Kosten ab?</b></summary>

Aufzählung berechnet dieselben Teilprodukte immer wieder (die naive Doppelsumme
hat exponentiell viele wiederholte Faktoren). VE klammert per Distributivgesetz
aus und **speichert Zwischenfaktoren**, sodass jedes Teilprodukt nur einmal
berechnet wird. Die Kosten werden vom **größten Zwischenfaktor** dominiert, dessen
Größe von der **Eliminationsreihenfolge** und letztlich der **Baumweite** des
Graphen abhängt. Bei kleiner Baumweite (Polybäume) polynomiell; allgemein ist
BN-Inferenz NP-schwer.
</details>

<details><summary><b>7. Wann nutzt man Likelihood Weighting statt Rejection Sampling?</b></summary>

Rejection Sampling verwirft alle Stichproben, die der Evidenz widersprechen — bei
*seltener* Evidenz landet fast alles im Müll (exponentiell ineffizient).
Likelihood Weighting fixiert stattdessen die Evidenzvariablen auf ihre Werte und
**gewichtet** jede Stichprobe mit $\prod_{e_i}P(e_i\mid\mathrm{parents}(e_i))$;
keine Stichprobe wird verworfen. Es ist konsistent und deutlich effizienter, kann
aber bei Evidenz „weit unten" im Netz ebenfalls hohe Varianz haben — dann Gibbs/MCMC.
</details>

<details><summary><b>8. Leite die Filtering-Rekursion des HMM her (Predict/Update).</b></summary>

Gesucht $f_{1:t+1}=P(\mathbf X_{t+1}\mid\mathbf e_{1:t+1})$. Bayes bzgl. der neuen
Evidenz: $\propto P(\mathbf e_{t+1}\mid\mathbf X_{t+1},\mathbf e_{1:t})\,P(\mathbf X_{t+1}\mid\mathbf e_{1:t})$.
Sensor-Markov: erster Faktor $=P(\mathbf e_{t+1}\mid\mathbf X_{t+1})$ (**Update**).
Der zweite ist die **Prediction**: marginalisieren über $\mathbf X_t$,
$P(\mathbf X_{t+1}\mid\mathbf e_{1:t})=\sum_{\mathbf x_t}P(\mathbf X_{t+1}\mid\mathbf x_t)P(\mathbf x_t\mid\mathbf e_{1:t})$
(Markov-Übergang × vorherige Filter-Nachricht). Zusammen:
$f_{1:t+1}=\alpha\,P(\mathbf e_{t+1}\mid\mathbf X_{t+1})\sum_{\mathbf x_t}P(\mathbf X_{t+1}\mid\mathbf x_t)f_{1:t}$.
</details>

<details><summary><b>9. Beweise, dass Value Iteration konvergiert.</b></summary>

Der Bellman-Optimalitätsoperator $B$ mit $(BV)(s)=R(s)+\gamma\max_a\sum_{s'}P(s'\mid s,a)V(s')$
ist eine $\gamma$-Kontraktion in der Maximumsnorm: Für beliebige $V,V'$ gilt
$\lVert BV-BV'\rVert_\infty\le\gamma\lVert V-V'\rVert_\infty$ (nutze
$|\max_a f-\max_a g|\le\max_a|f-g|$ und $\sum_{s'}P=1$). Nach dem Banachschen
Fixpunktsatz hat $B$ einen eindeutigen Fixpunkt $V^\ast$, und die Iteration
$V_{k+1}=BV_k$ konvergiert geometrisch: $\lVert V_k-V^\ast\rVert_\infty\le\gamma^k\lVert V_0-V^\ast\rVert_\infty$.
Für $\gamma<1$ folgt Konvergenz; $\gamma\to1$ macht sie beliebig langsam.
</details>

<details><summary><b>10. Value Iteration vs. Policy Iteration — Vor- und Nachteile?</b></summary>

*Value Iteration*: pro Schritt ein billiges Bellman-Update über alle Zustände,
aber viele Schritte bis zur Konvergenz (geometrisch mit $\gamma$), und die Policy
stabilisiert sich oft *bevor* die Werte konvergiert sind. *Policy Iteration*: pro
Schritt teurer (Policy Evaluation löst ein $|S|\times|S|$-Gleichungssystem), dafür
sehr wenige Schritte — terminiert exakt in endlich vielen Iterationen, da es nur
endlich viele Policies gibt und jede Iteration strikt verbessert. Kompromiss:
*modifizierte* Policy Iteration (Evaluation nur näherungsweise). Beide sind
Instanzen der generalisierten Policy-Iteration — dem Kern des RL (Modul 13).
</details>

---

## Literatur & Quellen

**Lehrbücher**
- **Russell & Norvig, *AIMA*, 4. Aufl.** — Kap. 11 (klassische Planung), 12 (Planung
  in der realen Welt), 13 (Quantifizierung von Unsicherheit), 14 (probabilistisches
  Schließen/Bayes-Netze), 15 (temporale Modelle), 16 (einfache Entscheidungen),
  17 (komplexe Entscheidungen/MDPs). *Die primäre Quelle für dieses Modul.*
- **Koller & Friedman, *Probabilistic Graphical Models*, MIT Press** — die
  erschöpfende Referenz zu Bayes-Netzen, Inferenz und Lernen. *Vertiefend, anspruchsvoll.*
- **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2. Aufl.** — Kap. 3–4
  (MDPs, dynamische Programmierung) als perfekte Vertiefung des MDP-Teils und Brücke
  zu Modul 13. **Kostenlos** unter `incompleteideas.net/book/the-book.html`. *Sehr empfohlen.*
- **Ghallab, Nau & Traverso, *Automated Planning and Acting***, für den Planungsteil. *Vertiefend.*

**Frei verfügbare Kurse & Materialien** (kostenlos)
- **UC Berkeley CS188** — die Einheiten zu Bayes-Netzen, HMMs und MDPs mit den
  Pac-Man-Projekten (`inst.eecs.berkeley.edu/~cs188`). *Einsteigerfreundlich, praktisch.*
- **Stanford CS228 „Probabilistic Graphical Models"** — Notes online, `ermongroup.github.io/cs228-notes`. *Vertiefend.*
- **David Silver, *RL Course* (DeepMind/UCL)** — Vorlesungsvideos; Lecture 2–3 zu
  MDPs und dynamischer Programmierung. *Sehr gut für den MDP-Teil.*
- **Fast Downward / PDDL-Editor** (`editor.planning.domains`) — PDDL im Browser
  schreiben und einen echten Planer laufen lassen. *Praktisch.*

**Interaktiv / Visualisierungen** (kostenlos)
- **„Seeing Theory"** (`seeing-theory.brown.edu`) — interaktive Wahrscheinlichkeit & Bayes. *Einsteigerfreundlich.*
- **Bayes-Netz-Demos** (z. B. `github.com/mbilalzonjy/BayesNetVisualization`) und der **SamIam**-Reasoner zum Herumspielen.
- **Gridworld-MDP-Visualisierungen** (Value/Policy Iteration Schritt für Schritt), z. B. Andrej Karpathys `reinforcejs`.

**Klassische Papers** (kostenlos, vertiefend)
- Pearl (1988): *Probabilistic Reasoning in Intelligent Systems* — die Geburt der Bayes-Netze.
- Blum & Furst (1997): *Fast Planning Through Planning Graph Analysis* — GraphPlan.
- Hoffmann & Nebel (2001): *The FF Planning System* — die $h_{\text{FF}}$-Heuristik.

---

## Die drei Projekte

Die drei Projekte spiegeln die drei Modulteile — Planung, probabilistisches
Schließen, sequenzielle Entscheidung — und steigern sich in Schwierigkeit und
Eigenleistung:

- **01 – basic** (`projekte/01-basic/`): **Ein STRIPS-Vorwärtsplaner.** Geführtes
  Notebook: STRIPS-Zustände/Aktionen, Vorwärtssuche mit BFS *und* mit einer
  Relaxations-Heuristik ($h_{\text{add}}$) via A\*; angewandt auf Blocksworld. Viel
  Anleitung, knüpft direkt an Modul 06 an.
- **02 – medium** (`projekte/02-medium/`): **Bayes-Netz mit exakter und
  approximativer Inferenz.** Python-Projekt: Netzstruktur + CPTs, Inferenz durch
  Aufzählung *und* Variable Elimination *und* Likelihood Weighting; validiert am
  Alarm-Netz und einem Diagnose-Szenario. Wenig Anleitung.
- **03 – final** (`projekte/03-final/`): **Ein entscheidungstheoretischer
  MDP-Agent.** Keine Code-Vorgabe: Value Iteration *und* Policy Iteration auf einer
  Gridworld mit stochastischer Bewegung, Konvergenz empirisch prüfen, optimale
  Policy visualisieren, $\gamma$-Studie. Master-Niveau, Brücke zu RL.

Details, Setup und Musterlösungen jeweils in der `README.md` des Projektordners.
