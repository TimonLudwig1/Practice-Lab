# Projekt 01 (basic) — Das Internet als Graph

**Format: Jupyter Notebook** (`internet_graph.ipynb`). **Warum?** Graphen versteht man durch
**Hinschauen**: Gradverteilungen plotten, Hubs finden, das Netz zerbrechen lassen. Explorative
Analyse mit vielen Plots — genau das Notebook-Format.

---

## Ziel

Du arbeitest mit der **echten Internet-Topologie** und reproduzierst zwei der berühmtesten
Ergebnisse der Netzwerkforschung — um dann zu lernen, wie leicht man sich bei genau solchen
Analysen selbst betrügt.

**Datensatz:** `oregon1_010331` (SNAP/Stanford) — die AS-Peering-Topologie aus **BGP**-Daten der
Oregon Route-Views, 31. März 2001. **10 670 autonome Systeme, 22 002 Peerings.** Echte
Messdaten, 69 KB, Download in ~1 s, danach lokal gecached (`daten/`, gitignored).

*Warum echte Daten und kein synthetischer Graph?* Weil die spannenden Eigenschaften
(scale-free, Hub-Fragilität) genau die sind, die man bei synthetischen Graphen **hineinsteckt**
statt zu **finden**. Hier misst man sie.

## Vorwissen

Skript **Modul 16, Abschnitt 1** (Graph, Scale-Free, Robustheit) und **2.1/2.5** (Heuristiken,
Fallen). `networkx` wird im Notebook eingeführt.

## Aufgabe

Das meiste ist vorgegeben (basic!). Du füllst **vier** `# TODO`-Blöcke:

1. **Grad-Statistiken** (`grade`-Array aus `G.degree()`),
2. **Power-Law-Exponent** — `alpha_naiv` (log-log-Fit) **und** `alpha_mle` (Clauset-MLE),
3. **Robustheits-Schleife** — Knoten zufällig vs. nach Grad entfernen,
4. **Die vier Link-Prediction-Heuristiken** (Common Neighbors, Jaccard, Adamic-Adar, Pref.
   Attachment).

## Was am Ende herauskommt

**1. Scale-free.** Grad: min 1, **Median 2**, Mittel 4,12, **Max 2312**. Dichte 0,039 %.
Clustering 0,297. Es gibt **keinen typischen Grad** — der Mittelwert beschreibt niemanden
(dasselbe Muster wie die Flow-Größen in Modul 15).

**2. Der Exponent — und die erste Falle** ⚠️

| Methode | α |
|---|---|
| naiver log-log-Fit (PMF) | **1,14** ❌ |
| MLE, $d_{\min}=5$ | 2,12 |
| **MLE, $d_{\min}=10$** | **2,08** ✅ |

Der Literaturwert ist **≈ 2,1** (Faloutsos³ 1999). Der MLE trifft ihn, der naive Fit liegt um
**Faktor 2** daneben — und steht trotzdem in erschreckend vielen Papers.

**3. Robustheit** — der Albert/Jeong/Barabási-Befund (2000), auf echten Daten:

| entfernt | zufälliger Ausfall | **gezielt (Hubs)** |
|---|---|---|
| 1 % | 0,989 | **0,413** |
| 5 % | 0,939 | **0,003** |
| 10 % | 0,800 | 0,001 |

**5 % der Hubs entfernt → das Internet zerfällt in Staub (0,3 %)**, während 10 % *zufällige*
Ausfälle es kaum jucken (0,80). Dieselbe Struktur, die es gegen Pannen immunisiert, macht es
gegen Angreifer fragil. (Die Top-Hubs *sind* auch die Betweenness-Flaschenhälse: 16/20
Überschneidung.)

**4. Link Prediction — und die entscheidende Falle** 🕵️

Erster Durchlauf (uniform gezogene Negativbeispiele, wie überall üblich):

| Heuristik | ROC-AUC | PR-AUC |
|---|---|---|
| Common Neighbors | 0,719 | 0,712 |
| Adamic-Adar | 0,724 | 0,732 |
| **Preferential Attachment** | **0,763** | **0,857** |

**Preferential Attachment gewinnt** — ausgerechnet die *dümmste* Heuristik, die nur Grade
anschaut und die Nachbarschaft komplett ignoriert. Das ist verdächtig. Und tatsächlich:

Median-Gradprodukt $d_u\cdot d_v$: **echte Kanten 500**, **Zufallspaare 4**. In einem
scale-free-Graphen ist ein zufälliges Paar fast immer **Blatt × Blatt** — PA muss also gar keine
Struktur erkennen, es unterscheidet nur *Hub-beteiligt* von *Blatt×Blatt*.

Mit **grad-gematchten** Negativbeispielen (gleiche Gradverteilung wie die echten Kanten):

| Heuristik | uniform | **grad-gematcht** |
|---|---|---|
| Common Neighbors | 0,719 | 0,559 |
| Jaccard | 0,698 | 0,565 |
| **Adamic-Adar** | 0,724 | **0,566** ← jetzt bester |
| **Preferential Attachment** | **0,763** | **0,406** ← Zufallsniveau |

**PA stürzt von 0,763 auf 0,406.** Sein ganzer Vorsprung war **kein Signal, sondern ein Artefakt
der Negativ-Auswahl**. Und alle Verfahren fallen auf ~0,56: die Aufgabe ist **viel schwerer** als
die erste Tabelle suggeriert.

> ### Die drei Fallen dieses Projekts — alle vorgeführt, nicht behauptet
> 1. **Schätz-Falle:** naiver log-log-Fit statt MLE → Faktor 2 falsch.
> 2. **Leakage:** Testkanten müssen **vor** der Feature-Berechnung entfernt sein — sonst „sieht"
>    Common Neighbors die gesuchte Kante schon.
> 3. **Grad-Confound:** uniform gezogene Negative machen jede gradbasierte Größe künstlich stark.
>
> Dazu die Basisraten-Falle aus Modul 15: real ist $\pi\approx3{,}9\cdot10^{-4}$ (22 002 Kanten
> unter 57 Mio. Paaren), evaluiert wird 1:1-balanciert. **Nicht nur das Modell kann täuschen —
> schon die Konstruktion der Testmenge entscheidet, was man misst.**

Laufzeit: **~6 s** komplett.

## Ausführen / Setup

Repo-`venv`. Notebook öffnen:
`/.../xtAI_Learn-Repository/.venv/bin/python -m jupyter lab` (oder `.venv`-Kernel in VS Code).
Nur `networkx`, `numpy`, `matplotlib`, `scikit-learn` — alles vorhanden. Internet nur beim
**ersten** Laden nötig (der Download braucht einen `User-Agent`-Header, sonst 403).

## Lösung

Vollständig gelöst und **ausgeführt** in
[`loesung/internet_graph_loesung.ipynb`](loesung/internet_graph_loesung.ipynb). Erst selbst
probieren! Enthält vier Erweiterungs-Aufgaben (echten PPV via Bayes aus Modul 15 abschätzen,
Leakage bewusst herbeiführen, 1 % statt 10 % Testkanten, Robustheit gegen einen **Zufallsgraphen**
gleicher Größe vergleichen — der eigentliche Beweis, dass die Asymmetrie an der
scale-free-Struktur liegt).

> **Vorbehalt** (Skript 4.4): Die „echte" AS-Topologie kennt **niemand**. Route-Collectors sehen
> nur, was BGP announciert; **Peerings zwischen kleinen AS fehlen systematisch**. Wir arbeiten
> auf einem *unvollständigen, verzerrt gemessenen* Graphen — das gilt für **alle** Zahlen oben.

## Transfer

Die Graph-Grundbegriffe und besonders die **saubere Link-Prediction-Evaluation** sind die Basis
für **Projekt 02** (GCN from scratch, das gegen genau diese Heuristiken antreten muss).
