# Project 02 (medium) — the base-rate fallacy: why "99 % detection" is worth nothing

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Format: Python project** (`.py` + tests). **Why?** The core is a **formula** that has to be
exactly right — and unit tests are the right tool for that (a notebook could not be secured this
way). It also separates things cleanly: loading data ↔ computational core ↔ analysis.

---

## Goal

Project 01 ended with a random forest at **99.99 %** and the suspicion that this is too good to be
true. Here comes the reckoning. You show — **empirically and analytically** — why a detector that
looks excellent on paper drowns in false alarms in **production**.

This is **the** core topic of intrusion detection (Axelsson 2000) and the reason why so many IDS
papers are practically worthless.

## Prior knowledge

Module 15 script, **section 3.1** (base-rate fallacy) and **3.2** (ROC vs. PR). Conditional
probability / **Bayes' theorem**. Module 04 (metrics, cost-based threshold).

## Files

| File | Role |
|---|---|
| `flow_data.py` | Loading/preprocessing the data, base-rate subsampling. **Given.** |
| `base_rate.py` | The computational core. **This is your work** (5 TODOs). |
| `run.py` | The four-part analysis plus plots. Given. |
| `test_base_rate.py` | Test suite (**14 tests**). |

## Task

Five short functions in `base_rate.py`. **The difficulty is not the code, it is really
understanding the formulas:**

1. **`ppv_at_base_rate(tpr, fpr, pi)`** — Bayes:
   $$P(I\mid A)=\frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi+\text{FPR}\cdot(1-\pi)}$$
   (must also handle arrays for `pi`).
2. **`required_fpr(tpr, pi, target_ppv)`** — **rearrange** the formula for FPR (pen and paper).
3. **`alarms_per_day(...)`** — absolute numbers instead of percentages.
4. **`expected_cost(...)`** — false alarm × analyst time + missed attack × damage.
5. **`best_operating_point(...)`** — the cost-minimal threshold taken from the ROC curve
   (the bridge to the cost-based threshold from module 04).

**Sanity check:** TPR = 0.99, FPR = 0.001, π = 10⁻⁴ ⇒ **PPV ≈ 9 %**.

## What comes out at the end

`python test_base_rate.py` → **14/14 green** (<1 s). `python run.py` → the analysis (~2 s):

**The detector** (logistic regression on volume/timing features):
ROC-AUC **0.9940**, PR-AUC **0.9817** — looks excellent. Operating point "I want to see 99 % of
the attacks": TPR = 0.9911, **FPR = 6.55 %**.

**1) ROC lies, PR does not** (lower the base rate by thinning out the attacks, detector unchanged):

| π | ROC-AUC | PR-AUC |
|---|---|---|
| 1 % | 0.9956 | 0.9605 |
| 0.1 % | 0.9956 | 0.8305 |
| 0.01 % | **0.9999** | **0.4333** |

The **ROC-AUC even rises to 0.9999**, while the PR-AUC collapses to 0.43. The ROC is **blind to
the base rate** — which is why it lies here.

**2) The fallacy in numbers** (fixed operating point):

| Base rate π | PPV = P(attack \| alarm) | Share of false alarms |
|---|---|---|
| 3.36 % (test set) | 34.5 % | 65.5 % |
| 0.1 % | 1.49 % | 98.5 % |
| **0.01 % (realistic)** | **0.15 %** | **99.85 %** |

At 10 million flows/day and π = 10⁻⁴: **991 real** and **654,745 false** alarms — per day.

**3) Theory meets measurement** (the nicest part): the Bayes prediction and the empirically
measured precision on the thinned data agree — **13.26 % vs. 13.29 %** and **1.49 % vs. 1.49 %**.
The effect is **not a quirk of the dataset**, it is pure probability theory. *(That is real
science: make a prediction and measure it.)*

**4) What would be needed:** for PPV = 50 % at π = 10⁻⁴ you would need **FPR ≤ 9.9·10⁻⁵** —
**661× better** than now. **The bottleneck is the FPR, not the detection rate.** The cost-optimal
threshold lowers the expected cost from **3.7 million** to **0.89 million EUR/day** — by
*sacrificing recall* (TPR 0.986) for far fewer false alarms.

> **Why a deliberately "weak" detector?** With **all** KDD features a random forest reaches
> **FPR = 0.0** — and then there would be nothing to show. But that is an **artifact of the overly
> easy dataset** (script 3.6), not reality. We therefore use only **volume/timing features**. This
> is justified twice over: it avoids the leaky KDD artifacts **and** matches exactly the reality of
> **encrypted** traffic, where only metadata are available (script 2.1).

## Running / setup

```bash
/.../xtAI_Learn-Repository/.venv/bin/python test_base_rate.py   # 14 tests, <1 s
/.../xtAI_Learn-Repository/.venv/bin/python run.py              # analysis + plot, ~2 s
```
`scikit-learn`, `pandas`, `numpy` (+ `matplotlib`). `pytest` optional (`__main__` runner).

## Solution

Complete in [`solution/`](solution/). Try it yourself first — the tests tell you exactly which
formula is still off (among them Axelsson's numerical example, the inverse check, and the
coin-flip detector, for which PPV = π **must** come out).

## Going further

- **Precision@k**: a SOC works through the 100 most urgent alarms. How many of them are real?
  (A more realistic metric than any AUC.)
- At which **base rate** is the detector just about usable (PPV > 50 %)?
- What happens at **π = 0** (no attack in the network)? What does the formula say — and what does
  that mean?
- **Alert budget:** your team can handle 100 alarms/day. Which FPR can you afford, and what recall
  do you still get for it?

---

# Projekt 02 (medium) — Der Base-Rate-Fallacy: warum „99 % Erkennung" nichts wert ist (deutsche Fassung)

**Format: Python-Projekt** (`.py` + Tests). **Warum?** Der Kern ist eine **Formel**, die exakt
stimmen muss — dafür sind Unit-Tests das richtige Werkzeug (ein Notebook könnte man nicht
absichern). Außerdem trennt es sauber: Daten laden ↔ Rechenkern ↔ Analyse.

---

## Ziel

Projekt 01 endete mit einem Random Forest bei **99,99 %** und dem Verdacht, dass das zu schön
ist. Hier kommt die Abrechnung. Du zeigst — **empirisch und analytisch** —, warum ein
Detektor, der im Papier hervorragend aussieht, im **Betrieb** an Fehlalarmen erstickt.

Das ist **das** Kernthema der Angriffserkennung (Axelsson 2000) und der Grund, warum so viele
IDS-Papers praktisch wertlos sind.

## Vorwissen

Skript **Modul 15, Abschnitt 3.1** (Base-Rate-Fallacy) und **3.2** (ROC vs. PR). Bedingte
Wahrscheinlichkeit / **Satz von Bayes**. Modul 04 (Metriken, Kosten-Schwelle).

## Dateien

| Datei | Rolle |
|---|---|
| `flow_data.py` | Daten laden/vorverarbeiten, Basisrate-Subsampling. **Vorgegeben.** |
| `base_rate.py` | Der Rechenkern. **Hier ist deine Arbeit** (5 TODOs). |
| `run.py` | Die vierteilige Analyse + Plots. Vorgegeben. |
| `test_base_rate.py` | Test-Suite (**14 Tests**). |

## Aufgabe

In `base_rate.py` fünf kurze Funktionen. **Die Schwierigkeit ist nicht der Code, sondern die
Formeln wirklich zu verstehen:**

1. **`ppv_at_base_rate(tpr, fpr, pi)`** — Bayes:
   $$P(I\mid A)=\frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi+\text{FPR}\cdot(1-\pi)}$$
   (muss auch Arrays für `pi` können).
2. **`required_fpr(tpr, pi, target_ppv)`** — die Formel **nach FPR umstellen** (Papier & Bleistift).
3. **`alarms_per_day(...)`** — absolute Zahlen statt Prozente.
4. **`expected_cost(...)`** — Fehlalarm × Analystenzeit + verpasster Angriff × Schaden.
5. **`best_operating_point(...)`** — kostenminimale Schwelle aus der ROC-Kurve
   (die Brücke zur Kosten-Schwelle aus Modul 04).

**Kontrollrechnung:** TPR = 0,99, FPR = 0,001, π = 10⁻⁴ ⇒ **PPV ≈ 9 %**.

## Was am Ende herauskommt

`python test_base_rate.py` → **14/14 grün** (<1 s). `python run.py` → die Analyse (~2 s):

**Der Detektor** (logistische Regression auf Volumen-/Timing-Features):
ROC-AUC **0,9940**, PR-AUC **0,9817** — sieht hervorragend aus. Betriebspunkt „ich will 99 % der
Angriffe sehen": TPR = 0,9911, **FPR = 6,55 %**.

**1) ROC lügt, PR nicht** (Basisrate durch Ausdünnen der Angriffe senken, Detektor unverändert):

| π | ROC-AUC | PR-AUC |
|---|---|---|
| 1 % | 0,9956 | 0,9605 |
| 0,1 % | 0,9956 | 0,8305 |
| 0,01 % | **0,9999** | **0,4333** |

Die **ROC-AUC steigt sogar auf 0,9999**, während die PR-AUC auf 0,43 einbricht. Die ROC ist
**blind für die Basisrate** — deshalb lügt sie hier.

**2) Der Fallacy in Zahlen** (fester Betriebspunkt):

| Basisrate π | PPV = P(Angriff \| Alarm) | Fehlalarm-Anteil |
|---|---|---|
| 3,36 % (Testmenge) | 34,5 % | 65,5 % |
| 0,1 % | 1,49 % | 98,5 % |
| **0,01 % (realistisch)** | **0,15 %** | **99,85 %** |

Bei 10 Mio. Flows/Tag und π = 10⁻⁴: **991 echte** und **654 745 falsche** Alarme — pro Tag.

**3) Theorie trifft Messung** (der schönste Teil): Die Bayes-Vorhersage und die empirisch
gemessene Precision auf den ausgedünnten Daten stimmen überein — **13,26 % vs. 13,29 %** und
**1,49 % vs. 1,49 %**. Der Effekt ist **keine Marotte des Datensatzes**, sondern reine
Wahrscheinlichkeitsrechnung. *(Das ist echte Wissenschaft: eine Vorhersage machen und sie messen.)*

**4) Was nötig wäre:** Für PPV = 50 % bei π = 10⁻⁴ bräuchte man **FPR ≤ 9,9·10⁻⁵** — **661× besser**
als jetzt. **Der Engpass ist die FPR, nicht die Erkennungsrate.** Die kostenoptimale Schwelle
senkt die erwarteten Kosten von **3,7 Mio.** auf **0,89 Mio. EUR/Tag** — indem sie *Recall
opfert* (TPR 0,986) für viel weniger Fehlalarme.

> **Warum ein absichtlich „schwacher" Detektor?** Mit **allen** KDD-Features erreicht ein Random
> Forest **FPR = 0,0** — dann gäbe es nichts zu zeigen. Das ist aber ein **Artefakt des zu
> leichten Datensatzes** (Skript 3.6), nicht Realität. Wir nehmen daher nur **Volumen-/Timing-
> Features**. Das ist doppelt begründet: es vermeidet die leaky KDD-Artefakte **und** entspricht
> genau der Realität bei **verschlüsseltem** Verkehr, wo man nur Metadaten hat (Skript 2.1).

## Ausführen / Setup

```bash
/.../xtAI_Learn-Repository/.venv/bin/python test_base_rate.py   # 14 Tests, <1 s
/.../xtAI_Learn-Repository/.venv/bin/python run.py              # Analyse + Plot, ~2 s
```
`scikit-learn`, `pandas`, `numpy` (+ `matplotlib`). `pytest` optional (`__main__`-Runner).

## Lösung

Vollständig in [`solution/`](solution/). Erst selbst versuchen — die Tests sagen dir genau,
welche Formel noch hakt (u. a. Axelssons Zahlenbeispiel, die Umkehr-Probe und der
Münzwurf-Detektor, bei dem PPV = π herauskommen **muss**).

## Weiterdenken

- **Precision@k**: Ein SOC bearbeitet die 100 dringendsten Alarme. Wie viele davon sind echt?
  (Realistischere Metrik als jede AUC.)
- Bei welcher **Basisrate** ist der Detektor gerade noch brauchbar (PPV > 50 %)?
- Was passiert bei **π = 0** (kein Angriff im Netz)? Was sagt die Formel — und was heißt das?
- **Alert-Budget:** Dein Team schafft 100 Alarme/Tag. Welche FPR darfst du dir leisten, und
  welchen Recall bekommst du dafür noch?
