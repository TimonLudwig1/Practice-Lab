# Projekt 03 (final) — Spatio-temporales Traffic-Forecasting: lohnt sich die Topologie?

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
   - **`saisonal_naiv`**: $\hat y_t = y_{t-168}$ (letzte Woche, gleiche Stunde). **Die Messlatte.**
   - **`mase`**: MAE relativ zur naiven Baseline. **< 1 = besser als naiv.**
   - **`baue_features`**: Zeitreihe → Tabelle. Lags ($t-1,t-2,t-3,t-24,t-168$),
     **zyklisch kodierte** Kalender-Features ($\sin/\cos$ von Stunde & Wochentag), und
     **optional** die Graph-Features (Nachbar-Mittel zu $t{-}1$, $t{-}2$, 2-Hop-Mittel).
   - **`zeit_split`**: **zeitbasiert** — die letzte Woche ist Test.
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

Vollständig und getestet in [`loesung/`](loesung/) (Generator, Modelle, `run.py`, **15 Tests**).
**Erst selbst versuchen!**

```bash
/.../xtAI_Learn-Repository/.venv/bin/python loesung/test_forecast.py   # 15 Tests, ~1 s
/.../xtAI_Learn-Repository/.venv/bin/python loesung/run.py            # Experiment + Plots, ~3 s
```
Nur `numpy`/`networkx`/`scikit-learn` (+ `matplotlib`). CPU, Sekunden. Der Graph wird beim ersten
Lauf nach `daten/` geladen (69 KB, gitignored).

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
