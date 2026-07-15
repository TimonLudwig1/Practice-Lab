# Projekt 03 (final) — Zero-Day-Erkennung: Wer findet den Angriff, den niemand kennt?

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
   Artefakte wie `serror_rate`), und ein `ZeroDaySzenario`, das sauber trennt:
   `normal_train` / `normal_test` / `bekannt_train` / `bekannt_test` / `zeroday` (dict je Typ).
2. **`detektoren.py` — die Kontrahenten.** Ein überwachter Detektor (z. B. Random Forest) und
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

Vollständig und getestet in [`loesung/`](loesung/) (Szenario, 5 Detektoren, `run.py`,
**11 Tests**). **Erst selbst versuchen!**

```bash
/.../xtAI_Learn-Repository/.venv/bin/python loesung/test_zeroday.py   # 11 Tests, ~26 s
/.../xtAI_Learn-Repository/.venv/bin/python loesung/run.py            # Experiment + Plot, ~10 s
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
