# Modul 16 — Machine Learning for Networks 2

> **Worum geht es?** Modul 15 behandelte jedes Netzwerkereignis für sich: ein **Flow**, ein
> Feature-Vektor, eine Vorhersage. Aber ein Netz ist kein Haufen unabhängiger Zeilen — es ist
> ein **Graph**, und Verkehr ist eine **Zeitreihe**. Dieses Modul nimmt genau die zwei
> Strukturen ernst, die Modul 15 ignoriert hat: **Struktur im Raum** (Topologie → *Graph Neural
> Networks*) und **Struktur in der Zeit** (Saisonalität → *Forecasting*) — und am Ende beides
> zusammen (**Spatio-Temporal Learning**). Dazu kommt der Blick nach vorn: verschlüsselte
> Verkehrsanalyse und **selbstfahrende Netze**.

**Hilfreiche Vorkenntnisse:** Lineare Algebra (Matrixmultiplikation, Eigenwerte),
Wahrscheinlichkeitsrechnung, PyTorch-Grundlagen.

**Diese Module solltest du vorher gemacht haben:**
- **Modul 15 (ML for Networks 1)** — *zwingend*. Flows, Messung, **Klassenungleichgewicht**,
  **Base-Rate-Fallacy**, PR-statt-ROC, **Concept Drift**, Leakage. Das alles gilt hier
  unverändert weiter und wird **nicht** wiederholt.
- **Modul 05 (ML 2)** und **Modul 09 (NLP 2)** — neuronale Netze, PyTorch, und die
  **Attention**-Idee (GAT ist Attention auf Graphen; ein Transformer *ist* ein GNN auf dem
  vollständigen Graphen).
- **Modul 13/14 (RL)** — nur für Abschnitt 4.3 (Routing/Traffic Engineering als
  Entscheidungsproblem).

> **Hinweis zur Ausgestaltung.** Wie bei Modul 15 lag keine offizielle Modulbeschreibung vor.
> Ich löse hier ein, was das Modul-15-Skript als Fortsetzung angekündigt hat: **Graph-basiertes
> Lernen, Netzwerk-Zeitreihen, verschlüsselte Verkehrsanalyse, selbstlernende Netze**.
>
> **Werkzeug-Entscheidung:** `torch_geometric` (die Standard-GNN-Bibliothek) ist hier **nicht
> installiert**, `statsmodels` (ARIMA) ebenfalls nicht. Das ist didaktisch ein Glücksfall: wir
> bauen das **GCN from scratch** in reinem PyTorch (es sind ~15 Zeilen — und danach ist die
> Bibliothek entzaubert) und das Forecasting ebenso. ARIMA/SARIMA erkläre ich vollständig
> **formal**, ohne Paket-Demo.

---

## Lernziele

Nach diesem Modul kannst du …

- ein Netz als **Graph** modellieren und seine Struktur charakterisieren: **Gradverteilung**,
  **Power-Law/Scale-Free**, **Small-World**, **Zentralitäten**, Clustering;
- erklären, **warum das Internet robust gegen Zufallsausfälle und fragil gegen gezielte
  Hub-Angriffe** ist — und es an echten Daten zeigen;
- **Power-Law-Exponenten korrekt schätzen** (MLE statt naivem log-log-Fit) und begründen,
  warum der naive Weg systematisch falsch liegt;
- **Graph Representation Learning** einordnen: handgebaute Graph-Features → **node2vec** →
  **GNNs**;
- das **Message-Passing-Framework** formulieren und **GCN** vollständig herleiten
  (inkl. der Frage, warum $\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ *so* aussieht);
  **GraphSAGE** (induktiv) und **GAT** (Attention) abgrenzen; **Over-Smoothing** erklären;
- **Link Prediction** als Aufgabe formulieren und Heuristiken (Common Neighbors, Jaccard,
  **Adamic-Adar**, Preferential Attachment) gegen gelernte Verfahren stellen;
- **Netzwerk-Zeitreihen** vorhersagen: Saisonalität, die **saisonal-naive Baseline**,
  Lag-Features, ARIMA/SARIMA (formal), und **Spatio-Temporal GNNs**;
- **verschlüsselte Verkehrsanalyse** (Website Fingerprinting) und **Self-Driving Networks**
  (Intent-Based Networking, Digital Twin, RL-Routing) einordnen.

---

## 1 · Das Netz als Graph

### 1.1 Welcher Graph eigentlich?

„Das Netzwerk" ist mehrdeutig — je nach Abstraktionsebene bekommt man einen anderen Graphen:

| Ebene | Knoten | Kanten | Größenordnung |
|---|---|---|---|
| **AS-Level** | Autonome Systeme (Provider, Firmen) | BGP-Peerings | ~75 000 heute |
| **Router-Level** | einzelne Router | physische Links | Millionen |
| **PoP/Backbone** | Standorte eines Betreibers | Glasfaserstrecken | ~10–100 |
| **Overlay/Verkehr** | Hosts | beobachtete Flows | riesig, dynamisch |

Die Projekte nutzen die **AS-Level-Topologie** (echte Daten, siehe unten). Wichtig: Der
**Verkehrsgraph** (wer redet mit wem) ist *nicht* der **Topologiegraph** (was ist physisch
verbunden) — eine Verwechslung, die viele Papers unbemerkt begehen.

Formal: $G=(V,E)$, **Adjazenzmatrix** $A\in\{0,1\}^{n\times n}$ mit $A_{ij}=1$ ⟺ $(i,j)\in E$,
**Gradmatrix** $D=\operatorname{diag}(d_1,\dots,d_n)$, $d_i=\sum_j A_{ij}$. Ungerichtet ⟹
$A=A^\top$. Bei $n=10\,670$ Knoten hätte ein *dichtes* $A$ schon 114 Mio. Einträge (~455 MB) —
bei 22 000 Kanten sind aber **99,96 %** davon Null (nur 0,04 % besetzt). **Netzgraphen sind
extrem dünn besetzt; man rechnet grundsätzlich sparse.**

### 1.2 Die Gradverteilung: Scale-Free

Die berühmteste Eigenschaft der Internet-Topologie (Faloutsos³, 1999): die Gradverteilung folgt
näherungsweise einem **Potenzgesetz**
$$P(d) \propto d^{-\alpha},\qquad \alpha \approx 2{,}1 \text{ (Internet-AS-Level)}.$$

Das bedeutet: **die meisten AS haben 1–2 Nachbarn, einige wenige „Hubs" haben Tausende.** Kein
typischer Grad, keine sinnvolle „mittlere" Skala — daher *scale-free*. Auf den echten Daten der
Projekte: Median-Grad **2**, Mittelwert **4,12**, Maximum **2312**. Wie bei den Flow-Größen aus
Modul 15 gilt: **der Mittelwert ist bedeutungslos**, wenn die Verteilung schwere Ränder hat.

Warum entsteht das? **Preferential Attachment** (Barabási–Albert): Das Netz *wächst*, und neue
Knoten hängen sich bevorzugt an bereits gut verbundene Knoten (ein neuer ISP kauft Transit bei
einem großen Provider). „Reich wird reicher" ⟹ Potenzgesetz mit $\alpha=3$ im
BA-Grundmodell.

> ### ⚠️ Wie man den Exponenten **nicht** schätzt
> Der naheliegende Weg — Histogramm der Grade, log-log auftragen, Gerade fitten — ist
> **systematisch falsch**. Grund: im Schwanz sitzen pro Bin nur 0–3 Beobachtungen; deren
> logarithmiertes Rauschen ist massiv verzerrt (und alle leeren Bins fallen ganz raus). Die
> Kleinste-Quadrate-Gerade wird dadurch verbogen.
> **Clauset, Shalizi & Newman (2009)** zeigen: man braucht den **Maximum-Likelihood-Schätzer**
> $$\hat\alpha = 1 + n\Big[\sum_{i=1}^{n}\ln\frac{d_i}{d_{\min}-\tfrac12}\Big]^{-1},$$
> angewandt nur auf den Schwanz $d\ge d_{\min}$. Auf unseren echten Daten:
>
> | Methode | Ergebnis |
> |---|---|
> | naiver log-log-Fit (PMF) | **−1,14** ❌ |
> | MLE, $d_{\min}=5$ | 2,12 |
> | **MLE, $d_{\min}=10$** | **2,08** ✅ (Literatur: ≈ 2,1) |
>
> Ein Faktor **2** Unterschied — und die naive Methode steht in erschreckend vielen Papers.
> (Projekt 01 rechnet beides nach.)

### 1.3 Small-World, Clustering, Zentralitäten

- **Small-World**: trotz 10 000+ Knoten ist der mittlere kürzeste Pfad winzig (~3–4 Hops im
  AS-Graph). „Sechs Handschläge" — im Internet eher drei. Ursache: die Hubs.
- **Clustering-Koeffizient** $C_i = \frac{2|\{(j,k)\in E: j,k\in N(i)\}|}{d_i(d_i-1)}$ — „wie
  viele meiner Nachbarn sind untereinander verbunden?" Real gemessen: **0,297** — weit über dem
  Zufallsgraphen, typisch für reale Netze.
- **Zentralitäten** — verschiedene Antworten auf „wer ist wichtig?":
  - **Grad**: viele direkte Nachbarn (lokal).
  - **Betweenness** $g(v)=\sum_{s\neq v\neq t}\frac{\sigma_{st}(v)}{\sigma_{st}}$: auf wie vielen
    kürzesten Pfaden liegt $v$? ⟹ **der Flaschenhals/Abhörpunkt schlechthin.** Teuer:
    $O(|V||E|)$ (Brandes).
  - **Closeness**: mittlere Distanz zu allen anderen ⟹ gut für Cache-/Server-Platzierung.
  - **Eigenvector/PageRank**: wichtig ist, wer *wichtige* Nachbarn hat (rekursiv).

### 1.4 Robustheit: der berühmteste Netzwerk-Befund

**Albert, Jeong & Barabási (2000, Nature):** Scale-free-Netze sind **extrem robust gegen
zufällige Ausfälle** und **extrem fragil gegen gezielte Angriffe auf Hubs**. Der Grund ist die
Gradverteilung selbst: Ein *zufällig* getroffener Knoten ist mit hoher Wahrscheinlichkeit ein
Blatt mit Grad 1–2 — sein Ausfall stört niemanden. Ein *gezielt* getroffener Hub reißt Tausende
Verbindungen mit.

Auf der **echten** AS-Topologie (Projekt 01 reproduziert das):

| entfernte Knoten | zufälliger Ausfall | gezielt (nach Grad) |
|---|---|---|
| 1 % | 0,988 | **0,413** |
| 5 % | 0,923 | **0,003** |
| 10 % | 0,836 | 0,001 |

*(Anteil der Knoten in der größten zusammenhängenden Komponente.)*

**5 % der Hubs entfernt — und das Internet zerfällt in Staub** (0,3 %), während 10 % zufälliger
Ausfälle es kaum jucken. Das ist zugleich eine Sicherheits-, eine Resilienz- und eine
Regulierungsaussage.

---

## 2 · Graph Representation Learning

**Das Problem:** ML will Vektoren, ein Graph ist aber kombinatorisch (und Knoten haben keine
kanonische Reihenfolge). Wie bekommt man aus Struktur brauchbare Merkmale? Drei Generationen:

### 2.1 Generation 1: handgebaute Graph-Features

Für jeden Knoten: Grad, Clustering, Zentralitäten, Nachbarschaftsstatistiken → normaler
Feature-Vektor → normales sklearn-Modell (Modul 04). **Simpel, schnell, interpretierbar** — und
eine ernstzunehmende Baseline, die man erst mal schlagen muss (Projekt 01).

Für **Kanten**(-Vorhersage) gibt es die klassischen Ähnlichkeits-Heuristiken. Mit $N(u)$ =
Nachbarschaft von $u$:

| Heuristik | Formel | Idee |
|---|---|---|
| **Common Neighbors** | $\|N(u)\cap N(v)\|$ | gemeinsame Bekannte |
| **Jaccard** | $\frac{\|N(u)\cap N(v)\|}{\|N(u)\cup N(v)\|}$ | normiert auf Nachbarschaftsgröße |
| **Adamic-Adar** | $\sum_{w\in N(u)\cap N(v)}\frac{1}{\log d_w}$ | seltene gemeinsame Nachbarn zählen mehr |
| **Preferential Attachment** | $d_u\cdot d_v$ | Hubs verbinden sich mit Hubs |

**Adamic-Adar** ist die stärkste dieser Heuristiken — die Gewichtung $1/\log d_w$ sagt: ein
gemeinsamer Nachbar mit Grad 3 ist ein starkes Indiz, ein gemeinsamer Nachbar mit Grad 2000
(ein Tier-1-Hub, mit dem *alle* verbunden sind) sagt fast nichts. Genau die richtige Intuition
für scale-free-Netze.

### 2.2 Generation 2: node2vec / DeepWalk

**Idee:** „Word2Vec für Graphen" (Modul 08!). Führe **Random Walks** auf dem Graphen aus,
behandle jeden Walk als *Satz* und Knoten als *Wörter*, und trainiere Skip-Gram. Knoten, die in
ähnlichen Kontexten auftauchen, bekommen ähnliche Embeddings.

**node2vec** steuert die Walks mit zwei Parametern zwischen zwei Extremen:
- **BFS-artig** (lokal) ⟹ Embeddings kodieren **strukturelle Rolle** (Hub? Brücke? Blatt?)
- **DFS-artig** (weit weg) ⟹ Embeddings kodieren **Community-Zugehörigkeit**

**Grenze:** rein **transduktiv** und **merkmalsblind**. Jeder Knoten bekommt einen fest
gelernten Vektor; kommt ein neuer Knoten dazu, muss man **neu trainieren**. Knoten-Features
(z. B. Verkehrsstatistiken) kann node2vec gar nicht nutzen. Genau das lösen GNNs.

### 2.3 Generation 3: Graph Neural Networks

**Das Message-Passing-Framework** — der gemeinsame Nenner *aller* GNNs. In Schicht $k$ gilt für
jeden Knoten $v$:
$$\mathbf h_v^{(k)} = \underbrace{\text{UPDATE}}_{\text{was mache ich damit?}}\Big(\mathbf h_v^{(k-1)},\ \underbrace{\text{AGGREGATE}}_{\text{was sagen die Nachbarn?}}\big(\{\mathbf h_u^{(k-1)}: u\in N(v)\}\big)\Big)$$

Jede Schicht = **ein Hop** weiter. Nach $k$ Schichten „sieht" ein Knoten seine
$k$-Hop-Nachbarschaft. Da AGGREGATE über eine **Menge** läuft, muss es
**permutationsinvariant** sein (Summe, Mittel, Max) — Knoten haben keine Reihenfolge.

#### GCN (Kipf & Welling 2017) — vollständig hergeleitet

Der naive Ansatz „mittle die Nachbarn und multipliziere mit einer Gewichtsmatrix" ist
$H' = \sigma(A H W)$. Drei Probleme, drei Reparaturen:

1. **Der Knoten vergisst sich selbst.** $A$ hat Nulldiagonale ⟹ $\mathbf h_v$ geht nicht in
   die Summe ein. Reparatur: **Self-Loops**, $\tilde A = A + I$.
2. **Die Skalen explodieren.** $\tilde A H$ *summiert* die Nachbarn — ein Hub mit Grad 2312
   bekommt Aktivierungen ~1000× größer als ein Blatt mit Grad 2. Nach ein paar Schichten
   divergiert das. Reparatur: **normieren**. Zeilenweise $\tilde D^{-1}\tilde A$ = Mittelwert
   der Nachbarn.
3. **Warum dann $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$?** Die *symmetrische* Normierung
   teilt eine Kante $(u,v)$ zwischen **beiden** Endpunkten auf:
   $$\hat A_{uv}=\frac{1}{\sqrt{d_u}\sqrt{d_v}}.$$
   Das hat drei Vorteile: (a) $\hat A$ bleibt **symmetrisch** (bei $\tilde D^{-1}\tilde A$
   nicht!), (b) die Botschaft eines Hubs an ein Blatt wird durch $\sqrt{d_{\text{Hub}}}$
   gedämpft — ein Hub, der mit *allen* verbunden ist, trägt eben wenig Information (dieselbe
   Intuition wie **Adamic-Adar** und wie **IDF** aus Modul 08!), (c) die Eigenwerte von $\hat A$
   liegen in $[-1,1]$ ⟹ **stabile Tiefe**. Formal ist das eine Spektralgraph-Theorie-Approximation
   (1. Ordnung Chebyshev auf dem normalisierten Laplace $L=I-\hat A$) — daher der Name
   *spectral* GNN.

Damit lautet die **GCN-Schicht**:
$$\boxed{\;H^{(k)}=\sigma\big(\hat A\,H^{(k-1)}\,W^{(k)}\big),\qquad \hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}\;}$$

Das ist **alles**. Zwei Zeilen Mathematik, ~15 Zeilen PyTorch — und $\hat A$ berechnet man
**einmal** vorab (sparse). Projekt 02 baut genau das.

#### GraphSAGE (Hamilton et al. 2017) — induktiv

GCN ist **transduktiv**: $\hat A$ enthält den ganzen Graphen, ein neuer Knoten erzwingt
Neuberechnung. **GraphSAGE** *sampelt* stattdessen eine feste Zahl Nachbarn und lernt eine
**Aggregator-Funktion**:
$$\mathbf h_v^{(k)}=\sigma\Big(W\cdot\text{CONCAT}\big(\mathbf h_v^{(k-1)},\ \text{AGG}(\{\mathbf h_u^{(k-1)}\})\big)\Big)$$
Weil die *Funktion* gelernt wird (nicht ein Vektor pro Knoten), lässt sie sich auf **nie
gesehene** Knoten/Graphen anwenden — **induktiv**. Das Nachbar-Sampling macht es zudem auf
riesigen Graphen skalierbar (Hubs würden sonst jeden Batch sprengen). Für Netze mit ständig
neuen Knoten ist das der praxisrelevante Ansatz.

#### GAT (Veličković et al. 2018) — Attention

GCN gewichtet Nachbarn **fest** nach Grad. **GAT** *lernt* die Gewichte:
$$\alpha_{vu}=\frac{\exp\big(\text{LeakyReLU}(\mathbf a^\top[W\mathbf h_v\,\|\,W\mathbf h_u])\big)}{\sum_{w\in N(v)}\exp\big(\text{LeakyReLU}(\mathbf a^\top[W\mathbf h_v\,\|\,W\mathbf h_w])\big)},\qquad
\mathbf h_v'=\sigma\Big(\sum_{u\in N(v)}\alpha_{vu}W\mathbf h_u\Big)$$
Das ist **exakt** die Attention aus Modul 09 — nur über die Nachbarschaft statt über alle
Tokens. Umgekehrt gilt: **ein Transformer ist ein GAT auf dem vollständigen Graphen** (jedes
Token ist mit jedem verbunden, Positional Encoding ersetzt die Kanten). Das ist keine Analogie,
sondern dieselbe Rechnung.

### 2.4 Over-Smoothing: warum GNNs flach bleiben

Tiefe Netze sind in Vision/NLP normal (50+ Schichten). GNNs haben meist **2–3**. Grund:
Jede Schicht mittelt über Nachbarn. Nach $k$ Schichten ist $H^{(k)}\approx\hat A^k H W\dots$ —
und $\hat A^k$ konvergiert für großes $k$ gegen die stationäre Verteilung eines Random Walks.
**Alle Knotenvektoren werden einander ähnlich**, die Information kollabiert (formal: die
Eigenwerte $<1$ sterben aus, nur der zum Eigenwert 1 überlebt). Das heißt **Over-Smoothing**.
Verschärft wird es im **Small-World**-Netz: bei mittlerem Pfad ~4 sieht ein 4-Schicht-GNN schon
*den ganzen Graphen* — jeder Knoten aggregiert dann praktisch dasselbe. Gegenmittel:
Residual-/Skip-Connections, **Jumping Knowledge**, PairNorm — oder schlicht: flach bleiben.

### 2.5 Link Prediction

**Aufgabe:** Gegeben den Graphen ohne einige Kanten — welche fehlen? **Anwendungen im Netz:**
Topologie-Inferenz (die gemessene AS-Topologie ist notorisch **unvollständig** — viele Peerings
sieht kein Route-Collector), Vorhersage künftiger Peerings, Anomalieerkennung (eine Kante, die
„nicht sein dürfte").

**Der große Vorteil:** Link Prediction braucht **keine Labels** — der Graph labelt sich selbst.
Man entfernt einen Teil der Kanten (**positive** Beispiele), zieht gleich viele nicht-existente
Knotenpaare (**negative**), und prüft, ob das Modell sie trennt.

> ### ⚠️ Zwei Fallen, die Link-Prediction-Ergebnisse regelmäßig ruinieren
> 1. **Leakage über den Graphen selbst.** Die Test-Kanten müssen **aus dem Graphen entfernt
>    sein**, bevor man Features/Embeddings berechnet. Sonst berechnet man „Common Neighbors"
>    auf einem Graphen, der die gesuchte Kante schon enthält — und misst Auswendiglernen.
>    Das ist die Graph-Variante des Leakage-Problems aus Modul 15 (3.5).
> 2. **Die Basisrate — schon wieder.** Ein Graph mit $n=10\,670$ hat ~57 Mio. mögliche Paare,
>    aber nur 22 000 Kanten ⟹ die echte Basisrate ist $\pi\approx 4\cdot10^{-4}$. Evaluiert man
>    (wie üblich) mit **1:1** positiv:negativ, misst man bei $\pi=0{,}5$ — und der **Base-Rate-
>    Fallacy aus Modul 15 (3.1)** schlägt in voller Härte zu: eine AUC von 0,95 auf balancierten
>    Daten heißt im echten Einsatz (alle Paare durchprobieren) fast nur Fehlalarme. Balanciertes
>    Sampling ist bequem — aber man muss dazusagen, dass es die Realität **nicht** abbildet.

---

## 3 · Netzwerk-Zeitreihen & Forecasting

### 3.1 Die Aufgabe und die Struktur

Verkehr $y_t$ auf einem Link/Knoten vorhersagen — für Kapazitätsplanung, Autoscaling,
Energiesparen (Links nachts abschalten) und als **Baseline für Anomalieerkennung** („mehr
Verkehr als vorhergesagt = verdächtig"). Netzverkehr hat eine sehr freundliche Eigenschaft:
**starke, stabile Saisonalität**.
- **Tagesrhythmus** (Periode 24 h): Nachts wenig, abends Peak („Prime Time").
- **Wochenrhythmus** (Periode 7 d): Werktag ≠ Wochenende.
- **Trend**: langfristiges Wachstum.
- **Rauschen/Bursts**: Netzverkehr ist bekanntermaßen **selbstähnlich/langzeitkorreliert**
  (Leland et al. 1994) — Bursts auf *allen* Zeitskalen, nicht Poisson-glatt.

### 3.2 Baselines — und warum sie die halbe Miete sind

> **Die wichtigste Regel des Forecastings:** Schlage erst die triviale Baseline, dann rede über
> Modelle.

- **Naiv**: $\hat y_{t+1}=y_t$ („morgen wie heute").
- **Saisonal-naiv**: $\hat y_{t+h}=y_{t+h-m}$ mit Saisonlänge $m$ („nächsten Dienstag 20 Uhr wie
  letzten Dienstag 20 Uhr"). **Bei Netzverkehr ist das erschreckend stark** — jedes teure Modell
  muss sich zuerst hieran messen. Wer diese Baseline nicht berichtet, berichtet nichts.
- **Mittelwert/Drift**: Referenz nach unten.

Metriken: **MAE**, **RMSE** (bestraft Ausreißer stärker — bei burstigem Verkehr relevant),
**MAPE** (prozentual, aber explodiert bei $y_t\approx0$ ⟹ nachts unbrauchbar), und
**MASE** = MAE relativ zur naiven Baseline (**<1 = besser als naiv**; skalenfrei, deshalb der
sauberste Vergleich über verschiedene Links).

### 3.3 Klassisch: ARIMA / SARIMA (formal)

Mit dem **Backshift-Operator** $B y_t = y_{t-1}$:

- **AR(p)** — der Wert hängt von seinen $p$ Vorgängern ab:
  $\phi(B)y_t=\varepsilon_t$ mit $\phi(B)=1-\phi_1B-\dots-\phi_pB^p$.
- **MA(q)** — der Wert hängt von den letzten $q$ **Schocks** ab:
  $y_t=\theta(B)\varepsilon_t$ mit $\theta(B)=1+\theta_1B+\dots+\theta_qB^q$.
- **I(d)** — **Differenzieren** $\nabla^d=(1-B)^d$ macht die Reihe **stationär** (entfernt Trend).

**ARIMA(p,d,q)**: $\phi(B)\,(1-B)^d y_t=\theta(B)\,\varepsilon_t$.

**SARIMA(p,d,q)(P,D,Q)$_m$** ergänzt die Saison mit Periode $m$:
$$\Phi(B^m)\,\phi(B)\,(1-B^m)^D(1-B)^d\,y_t=\Theta(B^m)\,\theta(B)\,\varepsilon_t.$$
Der Term $(1-B^m)^D$ ist die **saisonale Differenz** — im Kern die saisonal-naive Baseline, in
das Modell eingebaut. Ordnungswahl klassisch über ACF/PACF oder AIC/BIC (Box-Jenkins).

*(Hier keine Code-Demo: `statsmodels` ist in dieser Umgebung nicht installiert. Die Projekte
nutzen stattdessen Lag-Features + ML — in der Praxis ohnehin oft stärker und flexibler.)*

### 3.4 ML auf Lag-Features

Der pragmatische, meist beste Weg: Zeitreihe → **Tabelle**. Features pro Zeitpunkt: Lags
($y_{t-1},y_{t-2},\dots,y_{t-m},y_{t-2m}$), gleitende Mittel/Std, und **Kalender-Features**
(Stunde, Wochentag) — Letztere am besten **zyklisch kodiert**:
$$\sin\!\Big(\frac{2\pi\,\text{Stunde}}{24}\Big),\ \cos\!\Big(\frac{2\pi\,\text{Stunde}}{24}\Big)$$
damit 23 Uhr und 0 Uhr benachbart sind (als rohe Zahl wären sie maximal weit auseinander!).
Darauf: Ridge oder Gradient Boosting (Modul 04/05). Für Sequenzen: LSTM (Modul 09) — bei
starker Saisonalität und wenig Daten aber oft **schlechter** als Ridge mit guten Lag-Features.

> **Evaluation: nur zeitbasiert.** Bei Zeitreihen ist ein zufälliger Split **fatal** — man
> trainiert auf der Zukunft und testet auf der Vergangenheit. Korrekt: **zeitlicher Split** bzw.
> **Rolling-Origin/Walk-Forward-Validation** (`TimeSeriesSplit`). Das ist dieselbe Leakage-Regel
> wie in Modul 15 (3.5) — hier nur noch offensichtlicher.

### 3.5 Die Synthese: Spatio-Temporal GNNs

Jetzt kommen beide Hälften zusammen. Verkehr auf einem Netz ist **gleichzeitig** zeitlich
korreliert (Saisonalität) **und räumlich** (benachbarte Links tragen denselben Verkehr — was
über Link A fließt, fließt gleich über B). Ein Modell pro Link ignoriert die Topologie; ein GNN
ohne Zeit ignoriert die Saisonalität.

**Spatio-Temporal GNNs** kombinieren beides. Die Datenstruktur ist ein Tensor
$X\in\mathbb R^{T\times n\times f}$ (Zeit × Knoten × Features), und die Architektur wechselt
zwischen:
- **räumlicher** Aggregation (GNN-Schicht über die Topologie) und
- **zeitlicher** Aggregation (1D-Faltung, GRU/LSTM oder Attention über $T$).

Bekannte Vertreter: **STGCN** (Graph-Conv + temporale Conv), **DCRNN** (Diffusion Convolution +
GRU; modelliert Verkehr explizit als **Diffusionsprozess** auf dem Graphen), **Graph WaveNet**
(lernt die Adjazenz gleich mit — nützlich, wenn die wahre Topologie unbekannt ist).

> **Ehrliche Einordnung:** Spatio-Temporal GNNs sind das Aushängeschild dieses Feldes — und
> gleichzeitig gilt in vielen veröffentlichten Vergleichen, dass eine **gut gemachte
> saisonal-naive oder Lag-Ridge-Baseline** erstaunlich nah herankommt. Die Topologie hilft nur,
> wenn die **räumliche Korrelation** wirklich stark ist und die zeitliche Struktur nicht ohnehin
> schon alles erklärt. **Projekt 03 misst genau das** — statt es zu behaupten.

---

## 4 · Advanced

### 4.1 Verschlüsselte Verkehrsanalyse

Modul 15 (2.1) hielt fest: Verschlüsselung schützt den **Inhalt**, nicht die **Metadaten**.
Zu Ende gedacht wird daraus **Website Fingerprinting**: Ein Beobachter (WLAN-Betreiber, ISP,
Tor-Exit-Beobachter) sieht nur **Paketgrößen, Richtungen und Timing** — und erkennt daraus,
*welche Seite* du besuchst. Denn jede Seite hat ein charakteristisches Ladeprofil (Anzahl und
Größe der Ressourcen). Merkmale sind Größen-/Richtungs-Sequenzen, Burst-Muster,
Inter-Arrival-Times; Modelle klassisch (kNN/RF) oder CNN/Transformer auf der Rohsequenz.

**Gegenmaßnahmen** kosten alle Bandbreite oder Latenz: **Padding** (auf feste Größen),
**Traffic Shaping** (konstante Rate), **Dummy-Traffic**, Multiplexing. Und: die
Laborergebnisse („95 % Genauigkeit!") sind meist **closed-world** (nur 100 Kandidatenseiten).
In der **open world** (Millionen Seiten, meist uninteressante) schlägt — natürlich — der
**Base-Rate-Fallacy** aus Modul 15 zu. Dieselbe Technik in Gut: QoE-Monitoring des Betreibers
(„läuft das Video gerade ruckelfrei?") ohne die Nutzdaten zu entschlüsseln.

### 4.2 Self-Driving Networks

Die Vision: Netze, die sich **selbst** beobachten, verstehen und optimieren — die
Konsequenz aus Modul 15+16.
- **Intent-Based Networking**: Der Operator sagt *was* er will („Videokonferenzen < 50 ms"),
  nicht *wie* (Router-Konfigurationen). Das System übersetzt, setzt um, verifiziert, korrigiert.
- **Digital Twin**: ein laufendes Simulationsmodell des echten Netzes — für „Was-wäre-wenn"
  ohne Risiko. Genau hier zahlt sich **modellbasiertes Lernen** aus (Modul 14, 4.4).
- **Closed Loop**: messen → verstehen → entscheiden → wirken → messen. Die Netz-Variante des
  Agent-Umgebung-Kreises aus Modul 13.
- **Realitäts-Check**: Netzbetrieb ist **sicherheitskritisch**. Ein Modell mit 99 % Genauigkeit,
  das 1 % der Zeit das Backbone fehlkonfiguriert, ist inakzeptabel. Deshalb sind
  **Erklärbarkeit**, **Verifikation**, konservative Fallbacks und begrenzte Autonomie hier keine
  Kür, sondern Voraussetzung. Das erklärt, warum die Vision seit ~2017 diskutiert und nur
  zögerlich eingesetzt wird.

### 4.3 Routing & Traffic Engineering als Entscheidungsproblem

Hier trifft dieses Modul den RL-Block (13/14). **Traffic Engineering**: Wie lege ich Pfade/
Gewichte, um Auslastung und Verzögerung zu minimieren? Klassisch: Optimierung (Multi-Commodity
Flow, LP) auf einer **geschätzten** Verkehrsmatrix. Mit RL: Zustand = Auslastung + Topologie
(→ **GNN als Encoder**!), Aktion = Routing-Gewichte, Belohnung = −Auslastung/Verzögerung.

**Warum GNN + RL hier zusammengehören:** Ein MLP auf einer festen Adjazenzmatrix müsste für
**jede** Topologie neu lernen. Ein GNN **generalisiert über Topologien** — genau das brauchen
Netze, die sich ständig ändern. (Vgl. **RouteNet**, das QoS-Metriken aus Topologie + Routing +
Verkehrsmatrix vorhersagt.)

**Aber:** Der Vergleich muss ehrlich gegen die **klassische Optimierung** laufen — die bei
bekanntem Modell oft **exakt optimal** ist. Das ist genau die Lektion aus dem Modul-14-Finale:
*Kennst du dein Modell, nimm Optimierung. RL ist für den Fall, dass du es nicht kennst.*

### 4.4 Skalierung und Privatsphäre

- **Skalierung:** Millionen Knoten passen nicht in einen Batch. Lösungen: Nachbar-Sampling
  (GraphSAGE), Cluster-GCN (Graph partitionieren), historische Embeddings.
- **Federated Learning:** Betreiber wollen/dürfen ihre Verkehrsdaten nicht teilen
  (Geschäftsgeheimnis, DSGVO). Lösung: Modelle statt Daten teilen. Am Rande relevant: schon die
  **Topologie** ist ein Geschäftsgeheimnis (Peering-Beziehungen).
- **Messverzerrung:** Die „echte" AS-Topologie kennt **niemand**. Route-Collectors sehen nur,
  was BGP ihnen zeigt; **Peer-to-Peer-Links zwischen kleinen AS fehlen systematisch**. Man
  trainiert also auf einem **unvollständigen, verzerrt gemessenen** Graphen — und die fehlenden
  Kanten sind nicht zufällig verteilt. Das ist bei *jeder* Aussage über Internet-Topologie
  mitzudenken (auch bei unserer Robustheits-Tabelle).

---

## 5 · Zusammenfassung / Cheat-Sheet

**Graph.** $G=(V,E)$, $A$ (sparse!), $D=\operatorname{diag}(d_i)$. AS-Level ≠ Router-Level ≠
Verkehrsgraph.

**Scale-Free.** $P(d)\propto d^{-\alpha}$, Internet-AS $\alpha\approx2{,}1$. Entsteht durch
**Preferential Attachment**. **Exponent per MLE schätzen**
($\hat\alpha=1+n[\sum\ln\frac{d_i}{d_{\min}-0{,}5}]^{-1}$), **nie** per log-log-Fit (Faktor 2 daneben!).

**Robustheit.** Zufallsausfall: harmlos (10 % → 0,84). **Hub-Angriff: fatal (5 % → 0,003).**

**Message Passing.** $\mathbf h_v^{(k)}=\text{UPDATE}(\mathbf h_v^{(k-1)},\text{AGGREGATE}(\{\mathbf h_u\}))$,
AGGREGATE **permutationsinvariant**.

**GCN.** $H^{(k)}=\sigma(\hat A H^{(k-1)}W^{(k)})$, $\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}$.
Self-Loops (sich selbst nicht vergessen) + symmetrische Normierung (Skalen zähmen, Hubs dämpfen,
Eigenwerte in $[-1,1]$).

**Familien.** GCN (transduktiv, feste Gewichte) · **GraphSAGE** (induktiv, Sampling) ·
**GAT** (Attention = Modul 09; Transformer = GAT auf vollständigem Graphen).

**Over-Smoothing.** $\hat A^k$ → stationär ⟹ alle Knoten gleich ⟹ GNNs bleiben **2–3 Schichten**.

**Link Prediction.** Common Neighbors · Jaccard · **Adamic-Adar** ($\sum 1/\log d_w$) ·
Pref. Attachment. **Test-Kanten VOR der Feature-Berechnung entfernen!** Echte Basisrate
$\approx4\cdot10^{-4}$, nicht 0,5.

**Forecasting.** Erst **saisonal-naiv** schlagen! · Lags + zyklische Kalender-Features
($\sin/\cos$) · **MASE < 1** = besser als naiv · SARIMA
$\Phi(B^m)\phi(B)(1-B^m)^D(1-B)^d y_t=\Theta(B^m)\theta(B)\varepsilon_t$ · **zeitlich splitten!**

**Spatio-Temporal.** $X\in\mathbb R^{T\times n\times f}$; STGCN/DCRNN/Graph WaveNet =
räumliche + zeitliche Aggregation im Wechsel.

---

## 6 · Selbsttest

<details>
<summary><b>1.</b> Was heißt „scale-free", und warum ist der mittlere Grad dabei wenig aussagekräftig?</summary>

$P(d)\propto d^{-\alpha}$ — es gibt **keine typische Skala**: die meisten Knoten haben Grad 1–2,
wenige Hubs Tausende. Der Mittelwert liegt (wie bei allen schweren Rändern, vgl. Flow-Größen in
Modul 15) irgendwo dazwischen und beschreibt **niemanden**: real Median 2, Mittel 4,12,
Maximum 2312. Ursache: **Preferential Attachment** beim Wachstum.
</details>

<details>
<summary><b>2.</b> Warum ist der naive log-log-Fit für den Power-Law-Exponenten falsch?</summary>

Im Schwanz liegen pro Bin nur 0–3 Beobachtungen; deren logarithmiertes Rauschen ist stark
verzerrt, leere Bins verschwinden ganz — die Kleinste-Quadrate-Gerade wird verbogen. Korrekt ist
der **MLE** (Clauset et al. 2009) auf dem Schwanz $d\ge d_{\min}$. Real gemessen: naiv **−1,14**
vs. MLE **2,08** — Faktor 2.
</details>

<details>
<summary><b>3.</b> Erkläre, warum das Internet robust gegen Zufallsausfälle und fragil gegen Hub-Angriffe ist.</summary>

Wegen der Gradverteilung. Ein **zufällig** getroffener Knoten ist fast sicher ein Blatt (Grad
1–2) — sein Ausfall stört nichts. Ein **gezielt** getroffener Hub reißt Tausende Kanten mit.
Real: 10 % zufällig → 84 % bleiben verbunden; **5 % Hubs → 0,3 %** (Albert/Jeong/Barabási 2000).
</details>

<details>
<summary><b>4.</b> Formuliere Message Passing. Warum muss AGGREGATE permutationsinvariant sein?</summary>

$\mathbf h_v^{(k)}=\text{UPDATE}(\mathbf h_v^{(k-1)},\ \text{AGGREGATE}(\{\mathbf h_u^{(k-1)}:u\in N(v)\}))$.
AGGREGATE läuft über eine **Menge** — Nachbarn haben **keine kanonische Reihenfolge**. Wäre die
Funktion reihenfolgeabhängig, hinge die Ausgabe von der willkürlichen Knotennummerierung ab.
Daher Summe/Mittel/Max.
</details>

<details>
<summary><b>5.</b> Warum $\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}$ — was leistet jeder Teil?</summary>

**$+I$ (Self-Loops):** sonst fällt $\mathbf h_v$ aus seiner eigenen Aktualisierung heraus — der
Knoten vergisst sich selbst. **$\tilde D^{-1/2}\cdot\tilde D^{-1/2}$:** normiert, sonst
explodieren die Skalen (Hub mit Grad 2312 vs. Blatt mit Grad 2). **Symmetrisch** statt
zeilenweise, weil (a) $\hat A$ symmetrisch bleibt, (b) die Botschaft eines Hubs gedämpft wird
(Hubs sind uninformativ — wie **Adamic-Adar**/IDF), (c) die Eigenwerte in $[-1,1]$ bleiben
⟹ stabile Tiefe.
</details>

<details>
<summary><b>6.</b> GCN vs. GraphSAGE vs. GAT — je ein Satz.</summary>

**GCN**: feste, gradbasierte Gewichte, braucht den **ganzen** Graphen ⟹ **transduktiv**.
**GraphSAGE**: **sampelt** Nachbarn und lernt eine **Aggregator-Funktion** ⟹ **induktiv**
(funktioniert auf neuen Knoten/Graphen) und skalierbar. **GAT**: **lernt** die
Nachbargewichte per **Attention** (= Modul 09, nur über die Nachbarschaft).
</details>

<details>
<summary><b>7.</b> Was ist Over-Smoothing, und warum sind GNNs deshalb flach?</summary>

Jede Schicht mittelt über Nachbarn; $\hat A^k$ konvergiert gegen die stationäre Verteilung eines
Random Walks ⟹ **alle Knotenvektoren werden gleich**, die Information kollabiert. Deshalb 2–3
Schichten. Im **Small-World**-Netz (mittlerer Pfad ~4) verschärft: ein 4-Schicht-GNN sieht schon
den ganzen Graphen.
</details>

<details>
<summary><b>8.</b> Nenne die zwei Fallen bei Link Prediction.</summary>

(1) **Leakage über den Graphen**: Test-Kanten müssen entfernt sein, **bevor** man Features/
Embeddings rechnet — sonst „sieht" Common Neighbors die gesuchte Kante schon.
(2) **Basisrate**: real $\pi\approx4\cdot10^{-4}$ (22 000 Kanten unter 57 Mio. Paaren), evaluiert
wird aber meist **1:1-balanciert** ⟹ Base-Rate-Fallacy (Modul 15): tolle AUC, im Echteinsatz
fast nur Fehlalarme.
</details>

<details>
<summary><b>9.</b> Was ist die saisonal-naive Baseline und was sagt MASE < 1?</summary>

$\hat y_{t+h}=y_{t+h-m}$ („wie letzte Woche zur selben Zeit"), $m$ = Saisonlänge. Bei Netzverkehr
**sehr stark**. **MASE** = MAE relativ zur naiven Baseline: **< 1 = besser als naiv**, ≥ 1 = das
Modell ist sein Geld nicht wert. Skalenfrei ⟹ über verschiedene Links vergleichbar.
</details>

<details>
<summary><b>10.</b> Warum Stunde/Wochentag zyklisch als sin/cos kodieren?</summary>

Als rohe Zahl wären **23 Uhr und 0 Uhr maximal weit entfernt** (Distanz 23), obwohl sie
benachbart sind. $\sin(2\pi h/24)$, $\cos(2\pi h/24)$ bilden die Stunde auf einen **Kreis** ab —
23 und 0 liegen dann direkt nebeneinander. Zwei Komponenten sind nötig, weil eine allein nicht
eindeutig ist.
</details>

<details>
<summary><b>11.</b> Was ist ein Spatio-Temporal GNN, und wann lohnt er sich?</summary>

Ein Modell auf $X\in\mathbb R^{T\times n\times f}$, das **räumliche** Aggregation (GNN über die
Topologie) und **zeitliche** Aggregation (Conv/GRU/Attention) abwechselt (STGCN, DCRNN, Graph
WaveNet). Er lohnt sich nur, wenn die **räumliche Korrelation** wirklich Zusatzinformation
trägt — sonst erklärt die Saisonalität allein schon alles und die saisonal-naive Baseline hält mit.
</details>

<details>
<summary><b>12.</b> Warum kennt niemand die echte AS-Topologie — und was heißt das für unsere Ergebnisse?</summary>

Route-Collectors sehen nur, was BGP ihnen zeigt; **Peer-to-Peer-Links zwischen kleinen AS fehlen
systematisch** (sie werden nirgends announciert). Der gemessene Graph ist also **unvollständig
und verzerrt**, und die fehlenden Kanten sind **nicht zufällig** verteilt. Jede Aussage — auch
unsere Robustheits-Tabelle — steht unter diesem Vorbehalt.
</details>

---

## 7 · Literatur & Quellen

**Graph/Netzwerk-Grundlagen:**
- 📗 **Barabási — *Network Science*** (networksciencebook.com) — **komplett frei online**,
  hervorragend illustriert; Kap. 4 (Scale-Free), 8 (Robustheit) decken Abschnitt 1 ab.
  *Einsteigerfreundlich.* **Beste Einzelquelle für Teil 1.**
- 📄 **Faloutsos, Faloutsos & Faloutsos (1999), *On Power-Law Relationships of the Internet
  Topology*** — das Paper, das alles startete.
- 📄 **Albert, Jeong & Barabási (2000), *Error and Attack Tolerance of Complex Networks***
  (Nature) — das Robustheits-Ergebnis aus 1.4.
- 📄 **Clauset, Shalizi & Newman (2009), *Power-Law Distributions in Empirical Data*** — warum
  der log-log-Fit falsch ist und wie man es richtig macht. *Vertiefend, sehr lehrreich.*

**Graph Learning:**
- 📗 **Hamilton — *Graph Representation Learning Book*** (frei, cs.mcgill.ca/~wlh/grl_book/) —
  das Standardwerk. *Einsteiger→vertiefend.* **Beste Einzelquelle für Teil 2.**
- 📄 **Kipf & Welling (2017), *Semi-Supervised Classification with GCNs*** — das GCN-Paper.
- 📄 **Hamilton et al. (2017), *Inductive Representation Learning on Large Graphs*** (GraphSAGE).
- 📄 **Veličković et al. (2018), *Graph Attention Networks*** (GAT).
- 📄 **Grover & Leskovec (2016), *node2vec***.
- 🎥 **Stanford CS224W — *Machine Learning with Graphs*** (Leskovec; Videos + Folien frei) —
  der beste Kurs zum Thema. *Vertiefend.*
- 🌐 **PyTorch Geometric Docs** (pyg.org) — die Standardbibliothek. Wir bauen im Modul bewusst
  from scratch; für die Praxis solltest du PyG kennen.

**Zeitreihen:**
- 📗 **Hyndman & Athanasopoulos — *Forecasting: Principles and Practice*** (otexts.com/fpp3) —
  **frei online**, das Standardwerk; Kap. 5 (Baselines, MASE), 9 (ARIMA/SARIMA).
  *Einsteigerfreundlich.* **Beste Einzelquelle für Teil 3.**
- 📄 **Leland et al. (1994), *On the Self-Similar Nature of Ethernet Traffic*** — der Klassiker:
  Netzverkehr ist **nicht** Poisson.
- 📄 **Yu et al. (2018), *STGCN***; **Li et al. (2018), *DCRNN***; **Wu et al. (2019),
  *Graph WaveNet*** — Spatio-Temporal GNNs. *Vertiefend.*

**Anwendung/Vision:**
- 📄 **Rusek et al. (2020), *RouteNet***— GNN sagt QoS aus Topologie+Routing+Verkehr vorher.
- 📄 **Feamster & Rexford (2018), *Why (and How) Networks Should Run Themselves*** —
  Self-Driving Networks. *Einsteigerfreundlich.*
- 📄 **Panchenko et al. (2016), *Website Fingerprinting at Internet Scale*** — inkl. der
  ernüchternden Open-World-Realität.

**Daten:**
- 🌐 **SNAP** (snap.stanford.edu/data) — echte Netzgraphen, u. a. die **Oregon-AS-Topologie**
  der Projekte. *Frei, kleine Downloads.*
- 🌐 **CAIDA** (caida.org) — die Referenz für Internet-Messdaten (teils Registrierung nötig).
- 🌐 **SNDlib** (sndlib.zib.de) — echte Backbone-Topologien **inkl. Verkehrsmatrizen**.

---

## Nächstes Modul

Damit ist der Netzwerk-Block (15+16) abgeschlossen. Es folgt **Modul 17 — Core XR: Principles
of Interactive Systems** und damit ein völlig neues Feld. Was du hier gelernt hast — Graphen als
Datenstruktur, GNNs, Zeitreihen, und vor allem die **Disziplin, gegen ehrliche Baselines zu
messen** — trägt weit über Netzwerke hinaus: 3D-Punktwolken (Modul 20) sind Graphen, Robotik
(21/22) ist Zeitreihen + Regelung, und Self-aware Computing (24) ist die Idee aus 4.2 in
allgemein.
