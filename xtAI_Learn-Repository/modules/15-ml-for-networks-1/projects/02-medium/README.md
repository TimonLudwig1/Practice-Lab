# Projekt 02 (medium) — Der Base-Rate-Fallacy: warum „99 % Erkennung" nichts wert ist

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

1. **`ppv_bei_basisrate(tpr, fpr, pi)`** — Bayes:
   $$P(I\mid A)=\frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi+\text{FPR}\cdot(1-\pi)}$$
   (muss auch Arrays für `pi` können).
2. **`benoetigte_fpr(tpr, pi, ziel_ppv)`** — die Formel **nach FPR umstellen** (Papier & Bleistift).
3. **`alarme_pro_tag(...)`** — absolute Zahlen statt Prozente.
4. **`erwartete_kosten(...)`** — Fehlalarm × Analystenzeit + verpasster Angriff × Schaden.
5. **`bester_betriebspunkt(...)`** — kostenminimale Schwelle aus der ROC-Kurve
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
