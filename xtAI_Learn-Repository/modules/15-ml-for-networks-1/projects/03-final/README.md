# Project 03 (final) — zero-day detection: who finds the attack nobody knows?

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Format: Python project, _without any given code_.** This final project gets **no scaffold** —
you build everything yourself: the scenario setup, the detectors, the evaluation, the tests. It is
the master-level examination piece of the module and consolidates **everything**: flow features
(P01), the base-rate calculation (P02), unsupervised methods (module 05) and sound evaluation
without leakage.

**Why a `.py` project?** Four components (scenario, detectors, evaluation, tests) interlock in a
testable way — and the central claim ("the zero-day was **never** in training") *has* to be secured
by a test, otherwise nobody will believe it.

---

## The scenario (and why it is the only honest one)

A real IDS faces a fundamental problem: **the next attack has not been invented yet.** Signatures
and supervised models only know what has already happened. The standard evaluation ("random split,
99 % accuracy") obscures this completely — it only tests whether the model *recognizes what it
already knows*.

This project therefore builds a **zero-day scenario**:

> In training, **only the attack `smurf` is known**. All other attack types (`neptune`, `satan`,
> `teardrop`, `portsweep`, `back`, `ipsweep`) do **not exist at all** in training — they only show
> up in production. A random split would be **self-deception** here (script 3.5: data leakage).

And then two philosophies compete against each other (script 2.2):
- **Misuse/supervised** — models *the bad*. Knows attacks (but only the known ones).
- **Anomaly/semi-supervised** — models *the normal*. **Never** sees an attack, only normal
  traffic.

## Learning objectives

- Design a **methodologically honest** experiment that measures what it claims to measure (hold
  back attack types entirely instead of splitting randomly).
- Implement **semi-supervised anomaly detection** (train on normal traffic only).
- Compare methods **fairly** — at the **same false alarm budget**, not at arbitrary default
  parameters.
- Translate the results into operational relevance with the **base-rate reality check** (P02).

## Assignment (step by step)

Build (file names are a suggestion):

1. **`flow_data.py` — the scenario.** Load/clean KDD99, build flow features (without the leaky
   artifacts such as `serror_rate`), and a `ZeroDayScenario` that separates cleanly:
   `normal_train` / `normal_test` / `known_train` / `known_test` / `zeroday` (a dict per type).
2. **`detectors.py` — the contenders.** One supervised detector (e.g. random forest) and **at
   least three** anomaly detectors that see **only** `normal_train` — e.g. **isolation forest**,
   **one-class SVM**, **LOF**, **Mahalanobis Gaussian**. Common interface `fit` / `score` /
   `predict`.
3. **`run.py` — the experiment.** Recall **per attack type** (known vs. zero-day) plus FPR on
   normal traffic, then the base-rate reality check (PPV, false alarms per day).
4. **`test_*.py` — tests.** In particular: **prove** that no zero-day type is in the training data.
   Also: bytes decoded correctly, FPR budget respected, core statements secured.

### The most important design trick: a shared FPR budget

If you compare detectors with their default parameters (`contamination`, `nu`, …), you compare
**apples with oranges** — each of them raises alarms at a different rate. Better: every detector
produces an **anomaly score**, and the **threshold** is set to the (100−1)-**percentile of the
scores on the training normal traffic**. Then *every* detector has **~1 % false alarms** by
construction, and the **zero-day recall is the only free quantity** — a fair comparison at the same
price.

## What should come out at the end

At the same **1 % FPR budget** (only `smurf` was known in training):

| Detector | FPR | smurf *(known)* | neptune | satan | teardrop | **avg. zero-day** |
|---|---|---|---|---|---|---|
| **Supervised RF** | 0.000 | 0.999 | **0.000** | 0.000 | 0.000 | **0.000** |
| Mahalanobis Gaussian | 0.009 | 1.000 | 0.982 | 0.867 | 0.182 | 0.350 |
| Isolation forest | 0.009 | 1.000 | 0.989 | 0.933 | 0.545 | 0.459 |
| One-class SVM (Nystroem) | 0.010 | 1.000 | 0.988 | 0.867 | 1.000 | 0.690 |
| **Local outlier factor** | 0.010 | 1.000 | 1.000 | 1.000 | 1.000 | **0.952** |

### The interpretation you should deliver

- **The supervised detector is perfect — and completely blind.** It detects `smurf` at 99.9 % with
  **zero** false alarms. And it misses **`neptune` 100 % of the time** — a SYN flood with 928
  flows that is *not subtle*. It just never saw it. **This is exactly what the standard evaluation
  never measures.**
- **The anomaly detectors have never seen an attack** — and still find up to **95 %** of the
  zero-days. That is the entire point of anomaly detection.
- **The price is in the FPR column:** 1 % instead of 0 %. Sounds like nothing — and is the reason
  why anomaly IDSs die in practice: at 10 million flows/day that is **~100,000 false alarms per
  day**, and the PPV drops (P02) to **< 1 %**. **The trade-off is not "which algorithm", it is
  "blindness towards the new" vs. "drowning in false alarms".** In practice you combine both:
  signatures for the known (cheap, precise) plus anomaly detection as a second line (expensive, but
  able to see).
- **Anomaly ≠ attack — stay honest:** `back` and `ipsweep` are **not found at all** by most
  detectors. `back` is an attack that looks like normal HTTP traffic — it *is* not statistically
  anomalous. Anomaly detection is not magic: it finds the *unusual*, not the *malicious*.

## Two pitfalls I stumbled over while building this

They are documented here because they are real lessons — and because they explain why results
sometimes look absurd **without** any error message appearing:

1. **`LocalOutlierFactor.predict()` is unusable on this data** — its `offset_` derails to
   ~−1.3·10⁶, so an alarm is **never** raised. Use `score_samples()` with your own threshold.
2. **Duplicates destroy LOF.** KDD99 is massively redundant (script 3.6): **13.4 %** of the normal
   traffic are **exact duplicates**. For LOF this is fatal — if the *k* neighbours of a point are
   identical to it, their distance is 0, the local density goes to ∞ and the LOF ratio explodes:

   | | max. training score | 99 % threshold | recall (smurf) |
   |---|---|---|---|
   | with duplicates | **1.55·10⁹** | 1.09·10⁶ | **0.000** |
   | deduplicated | 393 | 1.98 | **1.000** |

   So the redundancy of the dataset does not merely inflate metrics — it **paralyses entire classes
   of methods**. (Likewise: `SGDOneClassSVM` is **linear** and separates nothing here — score
   normal 3.946 vs. smurf 3.959; only a **Nystroem** kernel approximation makes it usable.)

## Reference solution

Complete and tested in [`solution/`](solution/) (scenario, 5 detectors, `run.py`, **11 tests**).
**Try it yourself first!**

```bash
/.../xtAI_Learn-Repository/.venv/bin/python solution/test_zeroday.py   # 11 tests, ~26 s
/.../xtAI_Learn-Repository/.venv/bin/python solution/run.py            # experiment + plot, ~10 s
```
Only `scikit-learn`/`pandas`/`numpy` (+ `matplotlib`). CPU.

> **Classifying the dataset** (script 3.6): KDD99 is **outdated, synthetic and too easy**. The
> *methodology* of this project transfers fully, the *absolute numbers* do **not**. For
> trustworthy statements: **UNSW-NB15** or **CIC-IDS2017**.

## Extensions (for the especially motivated)

- **A different zero-day set:** make `neptune` known and `smurf` the zero-day. Does the picture
  hold? How strongly does the result depend on the held-back type? (A result that hangs on *one*
  split is not a result.)
- **Ensemble:** combine supervised + anomaly (alarm if either fires). What does that do to recall
  and FPR — and to the PPV?
- **Autoencoder** (module 05/09) as an anomaly detector: reconstruction error as the score. Does it
  beat the simple Mahalanobis? (If not — what does that tell you?)
- **Vary the FPR budget** (0.1 % / 1 % / 5 %): plot zero-day recall over FPR — the actual decision
  curve for operations.
- **Concept drift** (script 3.3): train on the first half, test on the second.
- **Precision@100:** your team can handle 100 alarms/day. How many real zero-days are among them?

---

# Projekt 03 (final) — Zero-Day-Erkennung: Wer findet den Angriff, den niemand kennt? (deutsche Fassung)

**Format: Python-Projekt, _ohne Code-Vorgabe_.** Dieses Abschlussprojekt bekommt **kein
Gerüst** — du baust alles selbst: Szenario-Aufbau, Detektoren, Auswertung, Tests. Es ist die
Master-Prüfungsleistung des Moduls und konsolidiert **alles**: Flow-Features (P01),
Base-Rate-Rechnung (P02), unüberwachte Verfahren (Modul 05) und saubere Evaluation ohne Leakage.

**Warum ein `.py`-Projekt?** Vier Komponenten (Szenario, Detektoren, Auswertung, Tests) greifen
testbar ineinander — und die zentrale Behauptung („der Zero-Day war **nie** im Training") *muss*
durch einen Test abgesichert sein, sonst glaubt sie einem niemand.

---

## Das Szenario (und warum es das einzig ehrliche ist)

Ein reales IDS steht vor einem fundamentalen Problem: **Der nächste Angriff ist noch nicht
erfunden.** Signaturen und überwachte Modelle kennen nur, was schon passiert ist. Die
Standard-Evaluation („zufälliger Split, 99 % Accuracy") verschleiert das komplett — sie testet
nur, ob das Modell *Bekanntes wiedererkennt*.

Dieses Projekt baut deshalb ein **Zero-Day-Szenario**:

> Im Training ist **nur der Angriff `smurf` bekannt**. Alle anderen Angriffstypen (`neptune`,
> `satan`, `teardrop`, `portsweep`, `back`, `ipsweep`) existieren im Training **überhaupt
> nicht** — sie tauchen erst im Betrieb auf. Ein zufälliger Split wäre hier **Selbstbetrug**
> (Skript 3.5: Data Leakage).

Und dann treten zwei Philosophien gegeneinander an (Skript 2.2):
- **Misuse/Supervised** — modelliert *das Böse*. Kennt Angriffe (aber nur die bekannten).
- **Anomalie/Semi-supervised** — modelliert *das Normale*. Sieht **nie** einen Angriff,
  ausschließlich Normalverkehr.

## Lernziele

- Ein **methodisch ehrliches** Experiment entwerfen, das misst, was es zu messen vorgibt
  (Angriffstypen komplett zurückhalten statt zufällig splitten).
- **Semi-überwachte Anomalieerkennung** implementieren (nur auf Normalverkehr trainieren).
- Verfahren **fair** vergleichen — bei **gleichem Fehlalarm-Budget**, nicht bei zufälligen
  Default-Parametern.
- Die Ergebnisse mit dem **Base-Rate-Reality-Check** (P02) in Betriebsrelevanz übersetzen.

## Aufgabenstellung (Schritt für Schritt)

Baue (Dateinamen als Vorschlag):

1. **`flow_data.py` — Szenario.** KDD99 laden/säubern, Flow-Features bauen (ohne die leaky
   Artefakte wie `serror_rate`), und ein `ZeroDayScenario`, das sauber trennt:
   `normal_train` / `normal_test` / `known_train` / `known_test` / `zeroday` (dict je Typ).
2. **`detectors.py` — die Kontrahenten.** Ein überwachter Detektor (z. B. Random Forest) und
   **mindestens drei** Anomaliedetektoren, die **nur** `normal_train` sehen — z. B.
   **Isolation Forest**, **One-Class SVM**, **LOF**, **Mahalanobis-Gauß**. Gemeinsame
   Schnittstelle `fit` / `score` / `predict`.
3. **`run.py` — das Experiment.** Recall **je Angriffstyp** (bekannt vs. Zero-Day) + FPR auf
   Normalverkehr, dann der Base-Rate-Reality-Check (PPV, Fehlalarme/Tag).
4. **`test_*.py` — Tests.** Insbesondere: **beweise**, dass kein Zero-Day-Typ im Training
   steckt. Außerdem: Bytes korrekt dekodiert, FPR-Budget eingehalten, Kernaussagen abgesichert.

### Der wichtigste Design-Trick: ein gemeinsames FPR-Budget

Vergleicht man Detektoren mit ihren Default-Parametern (`contamination`, `nu`, …), vergleicht
man **Äpfel mit Birnen** — jeder alarmiert unterschiedlich oft. Besser: Jeder Detektor liefert
einen **Anomalie-Score**, und die **Schwelle** wird auf das (100−1)-**Perzentil der Scores auf
dem Trainings-Normalverkehr** gesetzt. Dann hat *jeder* per Konstruktion **~1 % Fehlalarme**,
und der **Zero-Day-Recall ist die einzige freie Größe** — ein fairer Vergleich bei gleichem Preis.

## Was am Ende herauskommen soll

Bei gleichem **1 %-FPR-Budget** (nur `smurf` war im Training bekannt):

| Detektor | FPR | smurf *(bekannt)* | neptune | satan | teardrop | **Ø Zero-Day** |
|---|---|---|---|---|---|---|
| **Supervised RF** | 0,000 | 0,999 | **0,000** | 0,000 | 0,000 | **0,000** |
| Mahalanobis-Gauß | 0,009 | 1,000 | 0,982 | 0,867 | 0,182 | 0,350 |
| Isolation Forest | 0,009 | 1,000 | 0,989 | 0,933 | 0,545 | 0,459 |
| One-Class SVM (Nystroem) | 0,010 | 1,000 | 0,988 | 0,867 | 1,000 | 0,690 |
| **Local Outlier Factor** | 0,010 | 1,000 | 1,000 | 1,000 | 1,000 | **0,952** |

### Die Interpretation, die du liefern sollst

- **Der überwachte Detektor ist perfekt — und vollkommen blind.** Er erkennt `smurf` zu 99,9 %
  bei **null** Fehlalarmen. Und er übersieht **`neptune` zu 100 %** — einen SYN-Flood mit 928
  Flows, der *nicht subtil* ist. Er hat ihn nur nie gesehen. **Genau das misst die
  Standard-Evaluation nie.**
- **Die Anomaliedetektoren haben nie einen Angriff gesehen** — und finden trotzdem bis zu
  **95 %** der Zero-Days. Das ist der ganze Sinn von Anomalieerkennung.
- **Der Preis steht in der FPR-Spalte:** 1 % statt 0 %. Klingt nach nichts — und ist der Grund,
  warum Anomalie-IDS in der Praxis sterben: bei 10 Mio. Flows/Tag sind das **~100 000
  Fehlalarme täglich**, und der PPV fällt (P02) auf **< 1 %**. **Der Zielkonflikt ist nicht
  „welcher Algorithmus", sondern „Blindheit gegenüber Neuem" vs. „Ertrinken in Fehlalarmen".**
  In der Praxis kombiniert man beides: Signaturen für Bekanntes (billig, präzise) + Anomalie
  als zweite Reihe (teuer, aber sehend).
- **Anomalie ≠ Angriff — ehrlich bleiben:** `back` und `ipsweep` werden von den meisten
  Detektoren **gar nicht** gefunden. `back` ist ein Angriff, der wie normaler HTTP-Verkehr
  aussieht — er *ist* statistisch nicht anomal. Anomalieerkennung ist keine Magie: sie findet
  *Ungewöhnliches*, nicht *Bösartiges*.

## Zwei Fallstricke, über die ich beim Bauen gestolpert bin

Sie stehen hier, weil sie echte Lektionen sind — und weil sie erklären, warum Ergebnisse
manchmal absurd aussehen, **ohne** dass eine Fehlermeldung kommt:

1. **`LocalOutlierFactor.predict()` ist auf diesen Daten unbrauchbar** — sein `offset_`
   entgleist auf ~−1,3·10⁶, sodass **nie** Alarm ausgelöst wird. Nutze `score_samples()` mit
   eigener Schwelle.
2. **Duplikate zerstören LOF.** KDD99 ist massiv redundant (Skript 3.6): **13,4 %** des
   Normalverkehrs sind **exakte Duplikate**. Für LOF ist das fatal — sind die *k* Nachbarn eines
   Punktes mit ihm identisch, ist ihr Abstand 0, die lokale Dichte geht gegen ∞ und der
   LOF-Quotient explodiert:

   | | max. Trainings-Score | 99 %-Schwelle | Recall (smurf) |
   |---|---|---|---|
   | mit Duplikaten | **1,55·10⁹** | 1,09·10⁶ | **0,000** |
   | dedupliziert | 393 | 1,98 | **1,000** |

   Die Redundanz des Datensatzes bläht also nicht nur Metriken auf — sie **legt ganze
   Verfahrensklassen lahm**. (Ebenso: `SGDOneClassSVM` ist **linear** und trennt hier gar nichts
   — Score normal 3,946 vs. smurf 3,959; erst eine **Nystroem**-Kernel-Approximation macht sie
   brauchbar.)

## Referenzlösung

Vollständig und getestet in [`solution/`](solution/) (Szenario, 5 Detektoren, `run.py`,
**11 Tests**). **Erst selbst versuchen!**

```bash
/.../xtAI_Learn-Repository/.venv/bin/python solution/test_zeroday.py   # 11 Tests, ~26 s
/.../xtAI_Learn-Repository/.venv/bin/python solution/run.py            # Experiment + Plot, ~10 s
```
Nur `scikit-learn`/`pandas`/`numpy` (+ `matplotlib`). CPU.

> **Datensatz-Einordnung** (Skript 3.6): KDD99 ist **veraltet, synthetisch und zu leicht**. Die
> *Methodik* dieses Projekts ist voll übertragbar, die *absoluten Zahlen* sind es **nicht**.
> Für belastbare Aussagen: **UNSW-NB15** oder **CIC-IDS2017**.

## Erweiterungen (für die besonders Motivierten)

- **Anderes Zero-Day-Set:** Mach `neptune` bekannt und `smurf` zum Zero-Day. Bleibt das Bild?
  Wie stark hängt das Ergebnis vom zurückgehaltenen Typ ab? (Ein Ergebnis, das an *einem*
  Split hängt, ist keins.)
- **Ensemble:** Kombiniere supervised + Anomalie (Alarm, wenn einer anschlägt). Was macht das
  mit Recall und FPR — und mit dem PPV?
- **Autoencoder** (Modul 05/09) als Anomaliedetektor: Rekonstruktionsfehler als Score. Schlägt
  er den simplen Mahalanobis? (Wenn nicht — was sagt das?)
- **FPR-Budget variieren** (0,1 % / 1 % / 5 %): Zeichne Zero-Day-Recall über FPR — die
  eigentliche Entscheidungskurve für den Betrieb.
- **Concept Drift** (Skript 3.3): Trainiere auf der ersten Hälfte, teste auf der zweiten.
- **Precision@100:** Dein Team schafft 100 Alarme/Tag. Wie viele echte Zero-Days sind darunter?
