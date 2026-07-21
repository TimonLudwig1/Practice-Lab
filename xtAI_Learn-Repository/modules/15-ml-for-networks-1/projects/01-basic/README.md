# Projekt 01 (basic) — Netzwerkverkehr klassifizieren: von Flows zu Features

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
