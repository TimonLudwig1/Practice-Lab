# P03 (final) — "Put-that-there": a multimodal reference interpreter

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 18 — Multimodal Interfaces** · Format: **Python project (free implementation, no given code)**

> This is the final project. There is **no given code** — you build the system yourself out of what you have learned in module 18. The reference solution is in [`solution/`](solution/); look at it **only after your own attempt**. This README is the complete specification.

## What it is about

You rebuild the core of Richard Bolt's legendary system from 1980 (script, sections 2 & 12): an interpreter that merges a **spoken deictic command** ("… put **that** …") with a **simultaneous pointing gesture** in order to resolve *which object* of the scene is meant. That is the **synergistic** CASE case (parallel + fused) with **complementary** modalities (CARE) — the most demanding and most interesting case of multimodal interaction.

The punchline: neither the pointing alone (spatially ambiguous — the pointer comes close to several objects) nor the temporal nor the semantic context alone is sufficient. Only their **multiplicative fusion** (the Bayes product) resolves the reference reliably. You will **prove that quantitatively** — including **mutual disambiguation** and the role of the temporal offset.

## Learning objective

You apply all three fusion levels of the module to a realistic, asynchronous scenario:
- the **Bayes product rule** over three factors (script, section 12),
- **temporal fusion** with a time window and the empirical offset "the gesture leads the word" (section 11),
- **mutual disambiguation** and its measurement (section 10),
- sound **ablation evaluation** (section 16).

## Prior knowledge

The entire module 18 [README](../../README.md), especially **sections 7 (the Bayes product), 11 (the time window), 12 (reference resolution), 16 (evaluation)**. From P01/P02 the fusion mechanics. Numpy.

---

## Assignment (the specification)

### 1. The scene & asynchronous event streams (the data generator)

Build a **reproducible, disclosed** generator (a fixed seed) — the choice of data is deliberately **synthetic**, because only this way are the *ground truth* (which object was really meant) and the built-in ambiguities exactly controllable; real pointing/speech data would have neither labels nor the targeted traps. Model:

- **The scene**: $n$ objects, each with a 2D screen position $\mathbf{p}_o \in [0,1]^2$ and a **type** $\tau_o$ from e.g. `{button, slider, text, image}`.
- **The speech stream**: a deictic word "that" at the time $t_{\text{word}}$, plus a **noisy ASR distribution** $q$ over a type-noun slot ("… that **button**"). Model the ASR uncertainty: with probability `asr_correct_prob` the distribution lies on the *true* type, otherwise on a wrong one; sometimes no noun is spoken at all ($q$ uniform).
- **The gesture stream**: a **sampled, noisy 2D pointing position** $\mathbf{r}(t)$ over time (timestamp + position).

**Deliberately build in two ambiguities** (this is the didactic core):
1. **A decoy (the temporal trap)**: let the pointer begin its movement **on a different object** and wander from there to the target. This way the pointer comes spatially very close to the decoy — but at the **wrong time** (right at the beginning, far before $t_{\text{word}}$). Only the **temporal** factor can rule it out.
2. **A twin (the semantic trap)**: with a probability of ~50 % place a second object **directly next to the target**, but with a **different type**. Spatially and temporally almost identical to the target — only the **semantic** factor (the heard type) can separate it.

Important: model the offset "the gesture leads the word" — the pointer reaches the target somewhat *before* the deictic word ($t_{\text{arrive}} = t_{\text{word}} + \mu$ with $\mu < 0$, empirically ~$-0.15$ s). Afterwards the pointer moves on (to the "there" position), so that the target is a *sharp waypoint*, not a permanent dwelling.

### 2. The interpreter (the fusion)

For every object $o$ compute $\log P(o) = \log P_{\text{point}}(o) + \log P_{\text{temp}}(o) + \log P_{\text{sem}}(o)$, then a softmax over the objects. What is resolved is $\arg\max_o P(o)$. The three factors:

$$P_{\text{point}}(o) \propto \exp\!\Big(-\frac{d_{\min}(o)^2}{2\sigma^2}\Big), \qquad
  P_{\text{temp}}(o) \propto \exp\!\Big(-\frac{(\delta_o - \mu)^2}{2\tau^2}\Big), \qquad
  P_{\text{sem}}(o) = q[\tau_o]$$

where $d_{\min}(o)$ is the minimal pointer-object distance over the whole stream and $\delta_o = t_{\text{near}}(o) - t_{\text{word}}$ is the temporal distance of the moment of greatest proximity to the deictic word. $P_{\text{point}}$ asks *"did the pointer ever come close?"*, $P_{\text{temp}}$ *"did that happen at the right time?"*, $P_{\text{sem}}$ *"does the type fit what was heard?"*.

Build the factors so that they can be **switched off individually** (flags), so that you can run ablations.

### 3. Evaluation

Produce a large pool of commands (e.g. 3000 over changing scenes) and measure the **resolution accuracy** (resolved == true object) for:

- **Ablations**: pointing only · semantics only · pointing+semantics (without time) · pointing+time (without semantics) · **the full fusion**. Plus a **naive baseline** (the object that was closest to the pointer *exactly* at the moment of the word).
- **Mutual disambiguation**: count the commands for which **pointing alone is wrong**, and of those the share that the **full fusion rescues**.
- **A μ study**: vary the assumed offset $\hat\mu$ and show that the accuracy is highest at $\hat\mu \approx \mu_{\text{true}}$ and drops clearly at the naive assumption $\hat\mu = 0$ ("simultaneous").

Produce plots (into `results/`, gitignored) and a test suite (a `__main__` runner, since there is no pytest).

---

## What should come out at the end (reference orders of magnitude)

Your numbers may deviate (different parameters/seeds), but the **ranking and the story** have to hold:

| Configuration | accuracy (reference) |
|---|---|
| pointing only ($P_{\text{point}}$) | ~0.48 — **ambiguous** (the decoy ≈ the target spatially) |
| semantics only ($P_{\text{sem}}$) | ~0.36 — weak on its own |
| pointing + semantics (without time) | ~0.69 |
| pointing + time (without semantics) | ~0.83 — **time eliminates the decoy** |
| **the full fusion** | **~0.89 — the maximum** |
| naive (the nearest object @ $t_{\text{word}}$) | ~0.48 — it fails (the pointer has already moved on) |

- **Mutual disambiguation**: in the reference, pointing alone is wrong for ~1560/3000 commands — of those the full fusion rescues **~86 %**. That is the quantitative proof that the temporal + semantic context resolves the spatial ambiguity.
- **The μ study**: the best $\hat\mu \approx -0.2$ s (close to the true $-0.15$); $\hat\mu = 0$ loses ~9 percentage points.

> **The big lesson of the module, shown on one system:** the three modality factors are **complementary, not redundant** — each resolves a *different* ambiguity (time → the decoy, semantics → the twin, space → the coarse position). That is why the full fusion beats every partial combination, and that is why "Put-that-there" is a *synergistic* interface: no modality is dispensable.

## Setup & running

```bash
cd modules/18-multimodal-interfaces/projects/03-final
# write your own implementation, then:
/Users/.../.venv/bin/python test_putthatthere.py   # the test suite
/Users/.../.venv/bin/python run.py                  # evaluation + plots
```

Only `numpy` + `matplotlib` are needed. Runtime a few seconds (pure CPU, no training).

## Solution

A complete reference is in [`solution/`](solution/): `putthatthere.py` (the generator + interpreter), `run.py` (the ablation, mutual disambiguation, the μ study, the plots), `test_putthatthere.py` (8 tests). **Build it yourself first!**

## Looking back & ahead

With this module 18 closes: you have worked through multimodal fusion from **inverse-variance perception** (P01) via **early/late fusion & mutual disambiguation** (P02) to the **complete synergistic interpreter** (P03). The pointing and reference mathematics here is direct preparatory work for **module 19 "3D User Interfaces"** (selection/manipulation in 3D space).

---

# P03 (final) — „Put-that-there": ein multimodaler Referenz-Interpreter (deutsche Fassung)

**Modul 18 — Multimodal Interfaces** · Format: **Python-Projekt (freie Umsetzung, keine Code-Vorgabe)**

> Dies ist das Abschlussprojekt. Es gibt **keinen vorgegebenen Code** — du baust das System selbst aus dem, was du in Modul 18 gelernt hast. Die Referenzlösung liegt in [`solution/`](solution/); sieh sie dir **erst nach eigenem Versuch** an. Diese README ist die vollständige Spezifikation.

## Worum es geht

Du baust den Kern von Richard Bolts legendärem System von 1980 (Skript, Abschnitt 2 & 12) nach: einen Interpreter, der einen **gesprochenen deiktischen Befehl** („… put **that** …") mit einer **gleichzeitigen Zeigegeste** verschmilzt, um aufzulösen, *welches Objekt* der Szene gemeint ist. Das ist der **synergistische** CASE-Fall (parallel + fusioniert) mit **komplementären** Modalitäten (CARE) — der anspruchsvollste und interessanteste Fall multimodaler Interaktion.

Der Clou: Weder das Zeigen allein (räumlich mehrdeutig — der Zeiger kommt mehreren Objekten nahe) noch der zeitliche noch der semantische Kontext allein genügen. Erst ihre **multiplikative Fusion** (Bayes-Produkt) löst die Referenz zuverlässig auf. Du wirst das **quantitativ nachweisen** — inklusive **Mutual Disambiguation** und der Rolle des zeitlichen Versatzes.

## Lernziel

Du wendest alle drei Fusionsebenen des Moduls an einem realistischen, asynchronen Szenario an:
- die **Bayes-Produktregel** über drei Faktoren (Skript, Abschnitt 12),
- **zeitliche Fusion** mit einem Zeitfenster und dem empirischen Versatz „Geste führt Wort" (Abschnitt 11),
- **Mutual Disambiguation** und deren Messung (Abschnitt 10),
- saubere **Ablations-Evaluation** (Abschnitt 16).

## Vorwissen

Die gesamte Modul-18-[README](../../README.md), besonders **Abschnitte 7 (Bayes-Produkt), 11 (Zeitfenster), 12 (Referenzauflösung), 16 (Evaluation)**. Aus P01/P02 die Fusionsmechanik. Numpy.

---

## Aufgabenstellung (Spezifikation)

### 1. Szene & asynchrone Ereignisströme (Datengenerator)

Baue einen **reproduzierbaren, offengelegten** Generator (fester Seed) — die Datenwahl ist bewusst **synthetisch**, weil nur so die *ground truth* (welches Objekt wirklich gemeint war) und die eingebauten Mehrdeutigkeiten exakt kontrollierbar sind; echte Zeige-/Sprachdaten hätten weder Labels noch die gezielten Fallen. Modelliere:

- **Szene**: $n$ Objekte, jedes mit einer 2D-Bildschirmposition $\mathbf{p}_o \in [0,1]^2$ und einem **Typ** $\tau_o$ aus z. B. `{button, slider, text, image}`.
- **Sprach-Strom**: ein deiktisches Wort „that" zum Zeitpunkt $t_{\text{word}}$, plus eine **verrauschte ASR-Verteilung** $q$ über einen Typ-Nomen-Slot („… that **button**"). Modelliere die ASR-Unsicherheit: mit Wahrscheinlichkeit `asr_correct_prob` liegt die Verteilung auf dem *wahren* Typ, sonst auf einem falschen; manchmal wird gar kein Nomen gesprochen ($q$ uniform).
- **Gesten-Strom**: eine über die Zeit **abgetastete, verrauschte 2D-Zeigeposition** $\mathbf{r}(t)$ (Zeitstempel + Position).

**Baue gezielt zwei Mehrdeutigkeiten ein** (das ist der didaktische Kern):
1. **Decoy (zeitliche Falle)**: Lass den Zeiger seine Bewegung **auf einem anderen Objekt** beginnen und von dort zum Ziel wandern. So kommt der Zeiger dem Decoy räumlich sehr nahe — aber zur **falschen Zeit** (ganz am Anfang, weit vor $t_{\text{word}}$). Nur der **zeitliche** Faktor kann ihn ausschließen.
2. **Zwilling (semantische Falle)**: Platziere mit ~50 % Wahrscheinlichkeit ein zweites Objekt **direkt neben dem Ziel**, aber mit **anderem Typ**. Räumlich und zeitlich fast identisch zum Ziel — nur der **semantische** Faktor (der gehörte Typ) kann ihn trennen.

Wichtig: Modelliere den **Versatz „Geste führt Wort"** — der Zeiger erreicht das Ziel etwas *vor* dem deiktischen Wort ($t_{\text{arrive}} = t_{\text{word}} + \mu$ mit $\mu < 0$, empirisch ~$-0.15$ s). Danach wandert der Zeiger weiter (zur „there"-Position), sodass das Ziel ein *scharfer Wegpunkt* ist, nicht ein dauerndes Verweilen.

### 2. Der Interpreter (die Fusion)

Für jedes Objekt $o$ berechne $\log P(o) = \log P_{\text{point}}(o) + \log P_{\text{temp}}(o) + \log P_{\text{sem}}(o)$, dann Softmax über die Objekte. Aufgelöst wird $\arg\max_o P(o)$. Die drei Faktoren:

$$P_{\text{point}}(o) \propto \exp\!\Big(-\frac{d_{\min}(o)^2}{2\sigma^2}\Big), \qquad
  P_{\text{temp}}(o) \propto \exp\!\Big(-\frac{(\delta_o - \mu)^2}{2\tau^2}\Big), \qquad
  P_{\text{sem}}(o) = q[\tau_o]$$

wobei $d_{\min}(o)$ die minimale Zeiger-Objekt-Distanz über den ganzen Strom ist und $\delta_o = t_{\text{near}}(o) - t_{\text{word}}$ der zeitliche Abstand des Moments größter Nähe zum deiktischen Wort. $P_{\text{point}}$ fragt *„kam der Zeiger je nahe?"*, $P_{\text{temp}}$ *„geschah das zur richtigen Zeit?"*, $P_{\text{sem}}$ *„passt der Typ zum Gehörten?"*.

Baue die Faktoren **einzeln abschaltbar** (Flags), damit du Ablationen fahren kannst.

### 3. Evaluation

Erzeuge einen großen Pool von Kommandos (z. B. 3000 über wechselnde Szenen) und miss die **Auflösungs-Genauigkeit** (aufgelöstes == wahres Objekt) für:

- **Ablationen**: nur Zeigen · nur Semantik · Zeigen+Semantik (ohne Zeit) · Zeigen+Zeit (ohne Semantik) · **volle Fusion**. Plus eine **naive Baseline** (das Objekt, das dem Zeiger *exakt* zum Wortzeitpunkt am nächsten war).
- **Mutual Disambiguation**: Zähle die Kommandos, bei denen **Zeigen allein danebenliegt**, und davon den Anteil, den die **volle Fusion rettet**.
- **μ-Studie**: Variiere den angenommenen Versatz $\hat\mu$ und zeige, dass die Genauigkeit bei $\hat\mu \approx \mu_{\text{wahr}}$ am höchsten ist und bei der naiven Annahme $\hat\mu = 0$ („gleichzeitig") deutlich einbricht.

Erzeuge Plots (nach `results/`, gitignored) und eine Testsuite (`__main__`-Runner, da kein pytest).

---

## Was am Ende herauskommen soll (Referenz-Größenordnungen)

Deine Zahlen dürfen abweichen (andere Parameter/Seeds), aber die **Rangfolge und die Geschichte** müssen stimmen:

| Konfiguration | Genauigkeit (Referenz) |
|---|---|
| nur Zeigen ($P_{\text{point}}$) | ~0.48 — **mehrdeutig** (Decoy ≈ Ziel räumlich) |
| nur Semantik ($P_{\text{sem}}$) | ~0.36 — allein schwach |
| Zeigen + Semantik (ohne Zeit) | ~0.69 |
| Zeigen + Zeit (ohne Semantik) | ~0.83 — **Zeit eliminiert den Decoy** |
| **volle Fusion** | **~0.89 — Maximum** |
| naiv (nächstes Objekt @ $t_{\text{word}}$) | ~0.48 — versagt (Zeiger schon weitergewandert) |

- **Mutual Disambiguation**: In der Referenz liegt Zeigen-allein bei ~1560/3000 Kommandos falsch — davon rettet die volle Fusion **~86 %**. Das ist der quantitative Beweis, dass der zeitliche + semantische Kontext die räumliche Mehrdeutigkeit auflöst.
- **μ-Studie**: bestes $\hat\mu \approx -0.2$ s (nahe am wahren $-0.15$); $\hat\mu = 0$ verliert ~9 Prozentpunkte.

> **Die große Lehre des Moduls, an einem System gezeigt:** Die drei Modalitäts-Faktoren sind **komplementär, nicht redundant** — jeder löst eine *andere* Mehrdeutigkeit (Zeit → Decoy, Semantik → Zwilling, Raum → grobe Position). Deshalb schlägt die volle Fusion jede Teilkombination, und deshalb ist „Put-that-there" ein *synergistisches* Interface: keine Modalität ist entbehrlich.

## Setup & Ausführen

```bash
cd modules/18-multimodal-interfaces/projects/03-final
# eigene Umsetzung schreiben, dann:
/Users/.../.venv/bin/python test_putthatthere.py   # Testsuite
/Users/.../.venv/bin/python run.py                  # Evaluation + Plots
```

Nur `numpy` + `matplotlib` nötig. Laufzeit wenige Sekunden (reine CPU, kein Training).

## Lösung

Vollständige Referenz in [`solution/`](solution/): `putthatthere.py` (Generator + Interpreter), `run.py` (Ablation, Mutual Disambiguation, μ-Studie, Plots), `test_putthatthere.py` (8 Tests). **Erst selbst bauen!**

## Rückblick & Ausblick

Damit schließt Modul 18: Du hast die multimodale Fusion von der **inverse-Varianz-Wahrnehmung** (P01) über **early/late Fusion & Mutual Disambiguation** (P02) bis zum **vollständigen synergistischen Interpreter** (P03) durchgearbeitet. Die Zeige- und Referenzmathematik hier ist direkte Vorarbeit für **Modul 19 „3D User Interfaces"** (Selektion/Manipulation im 3D-Raum).
