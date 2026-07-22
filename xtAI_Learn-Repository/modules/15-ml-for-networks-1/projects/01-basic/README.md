# Project 01 (basic) — classifying network traffic: from flows to features

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format: Jupyter notebook** (`flow_classification.ipynb`). **Why?** This is about
**understanding data**: looking at real, dirty network data, building features, plotting
distributions, comparing models. Exactly the exploratory workflow a notebook is made for (as in
modules 02–04).

---

## Goal

You work with **real flow data** for the first time (KDD Cup 99, via `scikit-learn`) and:

1. understand the **feature structure** of a flow record (volume, time, protocol, context),
2. build a clean **preprocessing** step (decode bytes, cast types, one-hot encode categorical
   features, **log-transform** heavy tails),
3. classify **normal vs. attack** as well as the **attack types**,
4. walk right into the **accuracy trap** — and understand why 99.99 % means almost nothing.

## Prior knowledge

Module 15 script, **section 1** (flows, features, heavy tails) and **3.1 stage 1** (accuracy
trap). Module 04 (classification, pipelines, metrics), pandas basics.

## Task

Most of it is given (this is the basic project!). You fill in **two** `# TODO` blocks:

1. **EDA:** histogram of `src_bytes` raw vs. **log-transformed** — make the heavy tail visible.
2. **Features:** build the target `y` (attack yes/no) and the matrix `X` — log transform of the
   byte counters plus one-hot for `protocol_type`/`service`/`flag`.

After that: compare models and **interpret** the result table.

## What comes out at the end

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Dummy** ("always normal") | **0.96645** | 0.000 | 0.000 | 0.000 |
| Logistic regression | 0.99957 | 0.995 | 0.992 | 0.994 |
| Random forest | **0.99987** | 1.000 | 0.996 | 0.998 |

**The lesson:** the dummy finds **not a single attack** and still has **96.6 % accuracy**. The
random forest is only ~3 points better in accuracy — the metric **hides** the entire relevant
difference, because it is dominated by the majority class. Precision/recall show the truth
(0.0 vs. ~1.0).

Plus two more doses of reality:
- **Ultra-rare classes:** `pod`, `multihop`, `warezmaster` have **exactly one** example → a
  stratified split is impossible (`ValueError`). We remove them **in a documented way** — and
  discuss that this is precisely one popular way of prettifying results.
- **Tiny supports:** `teardrop` has 3 test examples → recall 0.67. A single misclassification
  costs 33 points. Percentages like these are **noise**.

Runtime: **~8 s** (the RF trains in 0.3 s). On the first run the dataset downloads ~2 MB and is
cached in `~/scikit_learn_data`.

> ### ⚠️ Two honest warnings
>
> **1. The 99.99 % are too good to be true.** (a) KDD99 is **redundant and too easy** — artifacts
> such as `src_bytes` separate the classes almost perfectly; (b) we split **randomly** → **data
> leakage** (script 3.5); (c) the base rate of 3.36 % attacks is **absurdly high** for a real
> network. Why all of this collapses in production is computed in **project 02**.
>
> **2. The dataset** (script 3.6): KDD99 is **outdated (1999), synthetic, redundant**. We use it
> because it delivers real flow features with a realistic imbalance and no download hurdle — ideal
> for learning **methodology**. It permits **no** statement about real IDS quality. For real work:
> **UNSW-NB15** or **CIC-IDS2017**.

## A trap that is demonstrated explicitly in the notebook

The values arrive as bytes (`b'normal.'`). The tempting "fix" is **wrong**:

```python
s.astype(str).str.strip("b'")   # WRONG
```

`str.strip` removes **all** characters from the set `{b, '}` at both ends — not the prefix `b'`.
So `b'back.'` becomes **`ack`**, and the service `b'bgp'` becomes **`gp`**. You silently destroy
data and never notice. The correct way is a real `decode()`; an `assert` in the notebook secures
this. *(I fell into exactly this trap while building — which is why it is in here.)*

## Running / setup

Repo `venv`. Open the notebook:
`/.../xtAI_Learn-Repository/.venv/bin/python -m jupyter lab` (or the `.venv` kernel in VS Code).
Only `scikit-learn`, `pandas`, `numpy`, `matplotlib` — all available. Internet is needed only for
the **first** load.

## Solution

Fully solved and **executed** in
[`solution/flow_classification_solution.ipynb`](solution/flow_classification_solution.ipynb).
Try it yourself first! It contains four extension tasks (drop the log, protocol features only,
remove `src_bytes` → what does that say about the dataset?, read the confusion matrix).

## Transfer

Flow features, one-hot encoding, the log transform and the accuracy trap are the basis for
**project 02** (base-rate fallacy: why a "perfect" detector drowns in false alarms in production)
and **project 03** (zero-day detection without attack labels).

---

# Projekt 01 (basic) — Netzwerkverkehr klassifizieren: von Flows zu Features (deutsche Fassung)

**Format: Jupyter Notebook** (`flow_classification.ipynb`). **Warum?** Hier geht es um
**Datenverständnis**: echte, schmutzige Netzwerkdaten anschauen, Features bauen, Verteilungen
plotten, Modelle vergleichen. Genau der explorative Ablauf, für den ein Notebook gemacht ist
(wie in Modul 02–04).

---

## Ziel

Du arbeitest zum ersten Mal mit **echten Flow-Daten** (KDD Cup 99, via `scikit-learn`) und:

1. verstehst die **Feature-Struktur** eines Flow-Records (Volumen, Zeit, Protokoll, Kontext),
2. baust eine saubere **Vorverarbeitung** (Bytes dekodieren, Typen casten, kategoriale Features
   One-Hot-kodieren, schwere Ränder **log-transformieren**),
3. klassifizierst **normal vs. Angriff** sowie die **Angriffstypen**,
4. läufst in die **Accuracy-Falle** — und verstehst, warum 99,99 % fast nichts bedeuten.

## Vorwissen

Skript **Modul 15, Abschnitt 1** (Flows, Features, schwere Ränder) und **3.1 Stufe 1**
(Accuracy-Falle). Modul 04 (Klassifikation, Pipelines, Metriken), pandas-Grundlagen.

## Aufgabe

Das meiste ist vorgegeben (basic!). Du füllst **zwei** `# TODO`-Blöcke:

1. **EDA:** Histogramm von `src_bytes` roh vs. **log-transformiert** — mach den schweren Rand
   sichtbar.
2. **Features:** Ziel `y` (Angriff ja/nein) und Matrix `X` bauen — Log-Transformation der
   Byte-Zähler + One-Hot für `protocol_type`/`service`/`flag`.

Danach: Modelle vergleichen und die Ergebnistabelle **interpretieren**.

## Was am Ende herauskommt

| Modell | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Dummy** („immer normal") | **0.96645** | 0.000 | 0.000 | 0.000 |
| Logistische Regression | 0.99957 | 0.995 | 0.992 | 0.994 |
| Random Forest | **0.99987** | 1.000 | 0.996 | 0.998 |

**Die Lektion:** Der Dummy findet **keinen einzigen Angriff** und hat trotzdem **96,6 %
Accuracy**. Der Random Forest ist in Accuracy nur ~3 Punkte besser — die Metrik **versteckt**
den gesamten relevanten Unterschied, weil sie von der Mehrheitsklasse dominiert wird.
Precision/Recall zeigen die Wahrheit (0,0 vs. ~1,0).

Dazu zwei weitere Realitätsdosen:
- **Ultra-seltene Klassen:** `pod`, `multihop`, `warezmaster` haben **genau ein** Beispiel →
  ein stratifizierter Split ist unmöglich (`ValueError`). Wir entfernen sie **dokumentiert** —
  und besprechen, dass genau das eine beliebte Art ist, Ergebnisse zu schönen.
- **Winzige Supports:** `teardrop` hat 3 Testbeispiele → Recall 0,67. Eine einzige
  Fehlklassifikation kostet 33 Punkte. Solche Prozentzahlen sind **Rauschen**.

Laufzeit: **~8 s** (RF trainiert in 0,3 s). Der Datensatz lädt beim ersten Mal ~2 MB und wird in
`~/scikit_learn_data` gecached.

> ### ⚠️ Zwei ehrliche Warnungen
>
> **1. Die 99,99 % sind zu gut, um wahr zu sein.** (a) KDD99 ist **redundant und zu leicht** —
> Artefakte wie `src_bytes` trennen die Klassen fast perfekt; (b) wir splitten **zufällig** →
> **Data Leakage** (Skript 3.5); (c) die Basisrate von 3,36 % Angriffen ist für ein reales Netz
> **absurd hoch**. Warum das im Betrieb alles zusammenbricht, rechnet **Projekt 02** durch.
>
> **2. Der Datensatz** (Skript 3.6): KDD99 ist **veraltet (1999), synthetisch, redundant**. Wir
> nutzen ihn, weil er ohne Download-Hürde echte Flow-Features mit realistischem Ungleichgewicht
> liefert — ideal, um **Methodik** zu lernen. Er erlaubt **keine** Aussage über reale IDS-Güte.
> Für echte Arbeit: **UNSW-NB15** oder **CIC-IDS2017**.

## Eine Falle, die im Notebook explizit vorgeführt wird

Die Werte kommen als Bytes (`b'normal.'`). Der verlockende „Fix" ist **falsch**:

```python
s.astype(str).str.strip("b'")   # FALSCH
```

`str.strip` entfernt **alle** Zeichen aus der Menge `{b, '}` an beiden Enden — nicht das Präfix
`b'`. Aus `b'back.'` wird damit **`ack`**, aus dem Dienst `b'bgp'` wird **`gp`**. Man zerstört
still Daten und merkt es nie. Richtig ist ein echtes `decode()`; ein `assert` im Notebook sichert
das ab. *(Ich bin beim Bauen selbst in genau diese Falle getappt — deshalb steht sie drin.)*

## Ausführen / Setup

Repo-`venv`. Notebook öffnen:
`/.../xtAI_Learn-Repository/.venv/bin/python -m jupyter lab` (oder `.venv`-Kernel in VS Code).
Nur `scikit-learn`, `pandas`, `numpy`, `matplotlib` — alles vorhanden. Internet nur beim
**ersten** Laden nötig.

## Lösung

Vollständig gelöst und **ausgeführt** in
[`solution/flow_classification_solution.ipynb`](solution/flow_classification_solution.ipynb).
Erst selbst probieren! Enthält vier Erweiterungs-Aufgaben (Log weglassen, nur Protokoll-Features,
`src_bytes` entfernen → was sagt das über den Datensatz?, Konfusionsmatrix lesen).

## Transfer

Flow-Features, One-Hot, Log-Transformation und die Accuracy-Falle sind die Basis für **Projekt
02** (Base-Rate-Fallacy: warum ein „perfekter" Detektor im Betrieb an Fehlalarmen erstickt) und
**Projekt 03** (Zero-Day-Erkennung ohne Angriffslabels).
