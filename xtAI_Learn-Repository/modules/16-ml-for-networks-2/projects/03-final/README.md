# Project 03 (final) — spatio-temporal traffic forecasting: is the topology worth it?

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Format: Python project, _without any given code_.** No scaffold — you build the generator, the
models, the evaluation and the tests yourself. It is the master-level examination piece of the
module and the **synthesis of its two halves**: structure in **space** (graph) and structure in
**time** (seasonality).

---

## The guiding question

The script (3.5) claims something uncomfortable about spatio-temporal GNNs:

> "They are the flagship of this field — and at the same time, in many published comparisons a
> **well-made seasonal-naive or lag-ridge baseline** comes astonishingly close. The topology only
> helps if the **spatial correlation** really carries additional information."

**This project does not claim that — it measures it.** And not only "does the graph help, yes or
no", but the actually interesting question:

> **What does it depend on whether a graph model is worth it?**

## The data — and an honest decision

**Topology: real.** The AS peering topology from projects 01/02 (SNAP `oregon1_010331`), reduced
to a connected **backbone subgraph** (~190 nodes around the largest hubs). A real backbone has
10–100 PoPs — simulating 10,670 AS hourly would be neither realistic nor necessary.

**Traffic: simulated — and that is a justified choice.** Public traffic matrices of real backbones
(Abilene, GÉANT via SNDlib) could not be loaded without hurdles at build time (Zenodo mirror:
**HTTP 403**). Instead of taking a poor substitute, we simulate traffic with the properties that
real network traffic **demonstrably** has — and we **disclose the generator**, so that every
assumption is visible and changeable. *(A simulated dataset whose assumptions you can read is more
honest than a "real" one whose provenance you do not know.)*

Modelled (script 3.1): base load $\propto\log(\text{degree})$ (the gravity idea) · a **daily
pattern** (24 h) with a node-individual phase · a **weekly pattern** (weekend −30 %) · **AR(1)
noise** (traffic is correlated, not white) · **propagating events** (flash crowd / congestion: an
outbreak at *one* node spreads across the edges with a delay).

The last element is the pivot: **only if traffic spreads spatially can the topology provide any
additional information at all.** Two parameters dose this:
- `decay` — how strongly an event persists at the node **itself**,
- `spread` — how strongly it carries over to the **neighbours**.

## Assignment (step by step)

1. **`traffic_sim.py`** — load the topology, build the backbone subgraph, the row-normalized
   adjacency $A_{\text{norm}}$, and the **traffic generator** with all five components above.
   Reproducible (seed!), with `decay`/`spread` freely choosable.
2. **`forecast.py`** — in this order:
   - **`seasonal_naive`**: $\hat y_t = y_{t-168}$ (last week, same hour). **The yardstick.**
   - **`mase`**: MAE relative to the naive baseline. **< 1 = better than naive.**
   - **`build_features`**: time series → table. Lags ($t-1,t-2,t-3,t-24,t-168$),
     **cyclically encoded** calendar features ($\sin/\cos$ of hour & weekday), and **optionally**
     the graph features (the neighbour mean at $t{-}1$, $t{-}2$, and the 2-hop mean).
   - **`time_split`**: **time-based** — the last week is the test set.
3. **`run.py`** — compare the three models, **then the study**: vary `spread` and measure the
   benefit of the graph features.
4. **`test_*.py`** — among others: the daily pattern exists (autocorrelation at lag 24), the
   weekend is weaker, large AS carry more traffic, events propagate **only** at `spread > 0`, the
   simulation is reproducible, the **time split has no future in the training set**, the cyclic
   encoding works (11 p.m. is closer to midnight than to 11 a.m.), ridge beats the baseline.

### Two rules that are non-negotiable

- **Split by time.** A random split would be self-deception here: you would train on the future and
  test on the past (script 3.4 / module 15, 3.5). **A test has to secure this.**
- **Beat the baseline first, then talk about models.** Whoever starts with the GNN never knows
  whether it contributes anything. **MASE ≥ 1 means: your model is not worth its money.**

## What should come out at the end

**The three models** (at `decay=0.2`, `spread=0.75`; ~23 % of the traffic are events):

| Model | MAE | MASE |
|---|---|---|
| seasonal-naive | 16.91 | **1.0000** ← the yardstick |
| Lag+Ridge | 10.63 | **0.6286** |
| **Lag+Ridge+GRAPH** | 9.52 | **0.5630** |

Both ML models clearly beat the naive baseline; the graph features add another **10.4 %**.

**The actual insight — the study:**

| decay | spread | MASE without graph | MASE with graph | **gain** |
|---|---|---|---|---|
| 0.90 | 0.05 | 0.1727 | 0.1676 | **2.9 %** |
| 0.45 | 0.50 | 0.4986 | 0.4548 | **8.8 %** |
| 0.20 | 0.75 | 0.6286 | 0.5630 | **10.4 %** |
| 0.00 | 0.95 | 0.7456 | 0.6369 | **14.6 %** |

**The benefit of the topology grows monotonically with the spread** — and for a very concrete
reason: if the traffic stays local (`spread ≈ 0`, high `decay`), the node's **own past is already
the entire information** — $y_{t-1}$ already knows everything the neighbour would know too, and the
graph brings ~nothing. If traffic does spread, by contrast, the **neighbour at $t{-}1$** says
something about **me at $t$** that I *cannot* take from my own history.

> ### The answer to the guiding question
> "Is a spatio-temporal GNN worth it?" is **not a model question, it is a data question.** It
> hinges on whether the traffic propagates spatially — not on how sophisticated the architecture
> is. This also explains why papers in this field report such contradictory results: **they measure
> on data with differently strong spatial coupling.** Whoever proposes a graph model should
> **measure and report** this coupling — otherwise nobody (themselves included) knows why it worked
> or did not.

A side finding: at `decay=0.9` the MASE is **0.17** — the prediction becomes much better in
*absolute* terms (persistent events are easy to predict), but the **relative** benefit of the graph
shrinks. **A good MASE and a useful graph are two different things.**

## Reference solution

Complete and tested in [`solution/`](solution/) (generator, models, `run.py`, **15 tests**).
**Try it yourself first!**

```bash
/.../xtAI_Learn-Repository/.venv/bin/python solution/test_forecast.py   # 15 tests, ~1 s
/.../xtAI_Learn-Repository/.venv/bin/python solution/run.py            # experiment + plots, ~3 s
```
Only `numpy`/`networkx`/`scikit-learn` (+ `matplotlib`). CPU, seconds. The graph is downloaded to
`datasets/` on the first run (69 KB, gitignored).

## Extensions (for the especially motivated)

- **A real GNN instead of graph features:** replace the neighbour mean with a GCN from project 02
  (the `A_norm` multiplication *is* already message passing — a GCN additionally learns the
  weighting). Does it beat the handcrafted version? If not: what does that tell you?
- **Several steps ahead** ($h = 1, 6, 24$): from which horizon on does Lag+Ridge break down, and
  does the seasonal-naive baseline then keep up better?
- **The Graph WaveNet idea** (script 3.5): learn the adjacency **along the way** instead of
  prescribing it. Does the model recover the true topology?
- **Rolling-origin validation** (script 3.4) instead of a single split — how stable are the MASE
  values across several test weeks?
- **Anomaly detection** (a bridge to module 15): use the forecast as a **baseline** — "considerably
  more traffic than predicted" = alarm. And then compute the **PPV** at a realistic base rate.
  Sobering?
- **Self-similarity** (Leland et al. 1994): our AR(1) noise is *short-range* correlated. Real
  traffic is **long-range dependent** (bursts on all time scales). Replace AR(1) with fractional
  noise — does forecasting get harder?

---

# Projekt 03 (final) — Spatio-temporales Traffic-Forecasting: lohnt sich die Topologie? (deutsche Fassung)

**Format: Python-Projekt, _ohne Code-Vorgabe_.** Kein Gerüst — du baust Generator, Modelle,
Auswertung und Tests selbst. Es ist die Master-Prüfungsleistung des Moduls und die **Synthese
seiner beiden Hälften**: Struktur im **Raum** (Graph) und Struktur in der **Zeit** (Saisonalität).

---

## Die Leitfrage

Das Skript (3.5) behauptet über Spatio-Temporal GNNs etwas Unbequemes:

> „Sie sind das Aushängeschild dieses Feldes — und gleichzeitig kommt in vielen veröffentlichten
> Vergleichen eine **gut gemachte saisonal-naive oder Lag-Ridge-Baseline** erstaunlich nah heran.
> Die Topologie hilft nur, wenn die **räumliche Korrelation** wirklich Zusatzinformation trägt."

**Dieses Projekt behauptet das nicht — es misst es.** Und zwar nicht nur „hilft der Graph, ja
oder nein", sondern die eigentlich interessante Frage:

> **Wovon hängt es ab, ob sich ein Graph-Modell lohnt?**

## Die Daten — und eine ehrliche Entscheidung

**Topologie: echt.** Die AS-Peering-Topologie aus Projekt 01/02 (SNAP `oregon1_010331`),
reduziert auf einen zusammenhängenden **Backbone-Teilgraphen** (~190 Knoten rund um die größten
Hubs). Ein realer Backbone hat 10–100 PoPs — 10 670 AS stündlich zu simulieren wäre weder
realistisch noch nötig.

**Verkehr: simuliert — und das ist eine begründete Wahl.** Öffentliche Verkehrsmatrizen echter
Backbones (Abilene, GÉANT via SNDlib) waren beim Bau nicht ohne Hürden ladbar (Zenodo-Mirror:
**HTTP 403**). Statt einen schlechten Ersatz zu nehmen, simulieren wir Verkehr mit den
Eigenschaften, die echter Netzverkehr **nachweislich** hat — und legen den **Generator offen**,
damit jede Annahme sichtbar und veränderbar ist. *(Ein simulierter Datensatz, dessen Annahmen
man lesen kann, ist ehrlicher als ein „echter", dessen Zustandekommen man nicht kennt.)*

Modelliert (Skript 3.1): Grundlast $\propto\log(\text{Grad})$ (Gravity-Idee) · **Tagesgang**
(24 h) mit knotenindividueller Phase · **Wochengang** (Wochenende −30 %) · **AR(1)-Rauschen**
(Verkehr ist korreliert, nicht weiß) · **propagierende Ereignisse** (Flash Crowd / Congestion:
ein Ausbruch an *einem* Knoten breitet sich mit Verzögerung über die Kanten aus).

Das letzte Element ist der Angelpunkt: **Nur wenn Verkehr sich räumlich ausbreitet, kann die
Topologie überhaupt Zusatzinformation liefern.** Zwei Parameter dosieren das:
- `decay` — wie stark ein Ereignis am Knoten **selbst** nachwirkt,
- `spread` — wie stark es auf die **Nachbarn** übergeht.

## Aufgabenstellung (Schritt für Schritt)

1. **`traffic_sim.py`** — Topologie laden, Backbone-Teilgraph, zeilennormierte Nachbarschaft
   $A_{\text{norm}}$, und den **Verkehrsgenerator** mit allen fünf Komponenten oben.
   Reproduzierbar (Seed!), Parameter `decay`/`spread` frei wählbar.
2. **`forecast.py`** — in dieser Reihenfolge:
   - **`seasonal_naive`**: $\hat y_t = y_{t-168}$ (letzte Woche, gleiche Stunde). **Die Messlatte.**
   - **`mase`**: MAE relativ zur naiven Baseline. **< 1 = besser als naiv.**
   - **`build_features`**: Zeitreihe → Tabelle. Lags ($t-1,t-2,t-3,t-24,t-168$),
     **zyklisch kodierte** Kalender-Features ($\sin/\cos$ von Stunde & Wochentag), und
     **optional** die Graph-Features (Nachbar-Mittel zu $t{-}1$, $t{-}2$, 2-Hop-Mittel).
   - **`time_split`**: **zeitbasiert** — die letzte Woche ist Test.
3. **`run.py`** — die drei Modelle vergleichen, **dann die Studie**: `spread` variieren und den
   Nutzen der Graph-Features messen.
4. **`test_*.py`** — u. a.: Tagesgang existiert (Autokorrelation bei Lag 24), Wochenende
   schwächer, große AS tragen mehr Verkehr, Ereignisse propagieren **nur** bei `spread > 0`,
   Simulation reproduzierbar, **Zeit-Split ohne Zukunft im Training**, zyklische Kodierung
   (23 Uhr näher an 0 Uhr als an 11 Uhr), Ridge schlägt die Baseline.

### Zwei Regeln, die nicht verhandelbar sind

- **Zeitbasiert splitten.** Ein zufälliger Split wäre hier Selbstbetrug: man trainierte auf der
  Zukunft und testete auf der Vergangenheit (Skript 3.4 / Modul 15, 3.5). **Ein Test muss das
  absichern.**
- **Erst die Baseline schlagen, dann über Modelle reden.** Wer mit dem GNN anfängt, weiß nie,
  ob es etwas beiträgt. **MASE ≥ 1 heißt: dein Modell ist sein Geld nicht wert.**

## Was am Ende herauskommen soll

**Die drei Modelle** (bei `decay=0.2`, `spread=0.75`; ~23 % des Verkehrs sind Ereignisse):

| Modell | MAE | MASE |
|---|---|---|
| saisonal-naiv | 16,91 | **1,0000** ← die Messlatte |
| Lag+Ridge | 10,63 | **0,6286** |
| **Lag+Ridge+GRAPH** | 9,52 | **0,5630** |

Beide ML-Modelle schlagen die naive Baseline klar; die Graph-Features bringen **10,4 %**
zusätzlich.

**Die eigentliche Erkenntnis — die Studie:**

| decay | spread | MASE ohne Graph | MASE mit Graph | **Gewinn** |
|---|---|---|---|---|
| 0,90 | 0,05 | 0,1727 | 0,1676 | **2,9 %** |
| 0,45 | 0,50 | 0,4986 | 0,4548 | **8,8 %** |
| 0,20 | 0,75 | 0,6286 | 0,5630 | **10,4 %** |
| 0,00 | 0,95 | 0,7456 | 0,6369 | **14,6 %** |

**Der Nutzen der Topologie wächst monoton mit der Ausbreitung** — und zwar aus einem sehr
konkreten Grund: Bleibt der Verkehr lokal (`spread ≈ 0`, `decay` hoch), ist die **eigene
Vergangenheit des Knotens bereits die ganze Information** — $y_{t-1}$ weiß schon alles, was der
Nachbar auch wüsste, und der Graph bringt ~nichts. Breitet sich Verkehr dagegen aus, sagt der
**Nachbar bei $t{-}1$** etwas über **mich bei $t$**, das ich meiner eigenen Historie *nicht*
entnehmen kann.

> ### Die Antwort auf die Leitfrage
> „Lohnt sich ein Spatio-Temporal-GNN?" ist **keine Modellfrage, sondern eine Datenfrage.**
> Sie hängt daran, ob der Verkehr räumlich propagiert — nicht daran, wie ausgefeilt die
> Architektur ist. Das erklärt zugleich, warum Papers auf diesem Gebiet so widersprüchliche
> Ergebnisse berichten: **sie messen auf Daten mit unterschiedlich starker räumlicher Kopplung.**
> Wer ein Graph-Modell vorschlägt, sollte diese Kopplung **messen und berichten** — sonst weiß
> niemand (er selbst eingeschlossen), warum es funktioniert hat oder nicht.

Nebenbefund: Bei `decay=0.9` ist MASE **0,17** — die Vorhersage wird *absolut* viel besser
(persistente Ereignisse sind leicht vorherzusagen), aber der **relative** Nutzen des Graphen
schrumpft. **Ein gutes MASE und ein nützlicher Graph sind zwei verschiedene Dinge.**

## Referenzlösung

Vollständig und getestet in [`solution/`](solution/) (Generator, Modelle, `run.py`, **15 Tests**).
**Erst selbst versuchen!**

```bash
/.../xtAI_Learn-Repository/.venv/bin/python solution/test_forecast.py   # 15 Tests, ~1 s
/.../xtAI_Learn-Repository/.venv/bin/python solution/run.py            # Experiment + Plots, ~3 s
```
Nur `numpy`/`networkx`/`scikit-learn` (+ `matplotlib`). CPU, Sekunden. Der Graph wird beim ersten
Lauf nach `datasets/` geladen (69 KB, gitignored).

## Erweiterungen (für die besonders Motivierten)

- **Echtes GNN statt Graph-Features:** Ersetze das Nachbar-Mittel durch ein GCN aus Projekt 02
  (die `A_norm`-Multiplikation *ist* schon Message Passing — ein GCN lernt die Gewichtung dazu).
  Schlägt es die Handarbeit? Wenn nicht: Was sagt das?
- **Mehrere Schritte voraus** ($h = 1, 6, 24$): Ab welchem Horizont bricht Lag+Ridge ein, und
  hält die saisonal-naive Baseline dann besser mit?
- **Graph WaveNet-Idee** (Skript 3.5): Lerne die Adjazenz **mit**, statt sie vorzugeben. Findet
  das Modell die echte Topologie wieder?
- **Rolling-Origin-Validation** (Skript 3.4) statt eines einzelnen Splits — wie stabil sind die
  MASE-Werte über mehrere Testwochen?
- **Anomalieerkennung** (Brücke zu Modul 15): Nutze den Forecast als **Baseline** — „deutlich
  mehr Verkehr als vorhergesagt" = Alarm. Und dann rechne den **PPV** bei realistischer
  Basisrate aus. Ernüchternd?
- **Selbstähnlichkeit** (Leland et al. 1994): Unser AR(1)-Rauschen ist *kurzzeit*korreliert.
  Echter Verkehr ist **langzeitkorreliert** (Bursts auf allen Zeitskalen). Ersetze AR(1) durch
  fraktionales Rauschen — wird Forecasting schwerer?
