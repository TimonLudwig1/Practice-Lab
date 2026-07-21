# Projekt 03 (final) — Eine XR-Nutzerstudie auswerten: 3 DoF vs. 6 DoF

**Format: Python-Projekt, _ohne Code-Vorgabe_.** Kein Gerüst — du baust Datengenerator,
Statistik-Werkzeuge, Auswertung und Tests selbst. Es ist die Master-Prüfungsleistung des Moduls
und schließt den Kreis: XR ist am Ende eine **empirische** Disziplin — über Präsenz und Übelkeit
entscheidet kein Benchmark, sondern ein **Mensch im Experiment** (Skript 5).

---

## Die Leitidee

Projekt 01 und 02 waren Physik und Mathematik — objektiv messbar. Dieses Projekt behandelt die
Fragen, die **nur am Menschen** beantwortbar sind: *Fühlt sich 6 DoF präsenter an? Wird von
3 DoF jemandem übler? Ist man schneller?* Und es behandelt sie so, wie eine ernstzunehmende
Studie es muss — mit den richtigen Tests, Effektstärken und Kontrollen.

> **Nicht das Ergebnis ist der Punkt** („6 DoF ist besser" wissen wir vorher), **sondern die
> Methodik**: eine Studie so auswerten, dass die Schlüsse **tragen**. Genau daran scheitern
> unzählige HCI-Papers.

## Das Szenario

Eine **Within-Subject-Studie** (Skript 5.2): 24 Probanden lösen dieselbe Aufgabe einmal mit
einem **3-DoF**-Headset (nur Orientierung) und einmal mit **6 DoF** (+ Position). Gemessen wird:

- **Presence** (IPQ, 1–7, höher = besser),
- **Sickness** (SSQ-Delta vorher/nachher, höher = schlechter),
- **Aufgabenzeit** (Sekunden, niedriger = besser),
- **Comfort** (1–7, höher = besser).

Weil individuelle Unterschiede in XR **riesig** sind (Anfälligkeit, VR-Erfahrung), ist
within-subject die richtige Wahl — jeder ist seine eigene Kontrolle. Der Preis:
**Reihenfolgeeffekte**, gegen die man **counterbalancen** muss (die Hälfte beginnt mit 3 DoF).

### Warum synthetische Daten — und warum das hier ein Vorteil ist

Eine echte Studie braucht Probanden, Wochen und eine Ethikkommission. Zum Erlernen der
**Auswertung** ist ein simulierter Datensatz sogar **besser**: Die „Wahrheit" (die echten
Effekte, die eingebauten Reihenfolgeeffekte) ist bekannt — man kann also prüfen, ob die
Statistik sie **wiederfindet**, und was passiert, wenn man Fehler macht. Der Generator wird
**vollständig offengelegt** und mit festem Seed reproduzierbar gemacht.

## Aufgabenstellung (Schritt für Schritt)

1. **`generate_study.py`** — der Datengenerator. Within-subject, counterbalanced, mit
   eingebauten Effekten *und* zwei **Reihenfolgeeffekten** (Carryover-Übelkeit in der zweiten
   Sitzung; Lerneffekt bei der Zeit). Plus eine Variante **ohne** Counterbalancing für die
   Gegenprobe.
2. **`stats_tools.py`** — die Auswertungsfunktionen, from scratch (pingouin/statsmodels fehlen):
   **Cohen's dz** (parametrische Effektstärke), **rank-biserial r** (Effektstärke zum Wilcoxon),
   ein gepaarter Vergleich (t-Test + Wilcoxon), **Bonferroni** und **Holm-Bonferroni**, ein
   **Order-Effekt-Check**.
3. **`run.py`** — die Auswertung in vier Schritten (siehe unten).
4. **`test_*.py`** — Tests: Effektstärken von Hand nachgerechnet, Korrekturen korrekt,
   `paired_comparison` stimmt mit scipy überein, die Analyse **findet die eingebauten Effekte**,
   und die Gegenprobe zeigt die Maskierung.

### Die vier methodischen Pflichten (Skript 5.2)

1. **Der richtige Test.** Likert-Items (IPQ, Comfort) sind **ordinal** → **Wilcoxon
   signed-rank**, nicht blind der t-Test. (Beides berichten und die Wahl begründen ist legitim.)
2. **Effektstärke, nicht nur p.** Ein signifikanter, aber winziger Effekt ist irrelevant.
   Immer **dz / rank-biserial** dazu.
3. **Mehrfachvergleiche korrigieren.** Vier Zielgrößen = vier Tests → bei $\alpha=0{,}05$ ist ein
   Zufallstreffer fast zu erwarten. **Bonferroni** ($\alpha/m$) oder Holm.
4. **Counterbalancing prüfen** und seinen Sinn belegen.

## Was am Ende herauskommen soll

**Die Vergleiche** (N=24, Wilcoxon):

| Zielgröße | 3 DoF | 6 DoF | Wilcoxon p | dz | r | nach Bonferroni |
|---|---|---|---|---|---|---|
| Presence | 4,0 | 4,9 | 0,0001 | −1,19 | −0,90 | **sig** |
| Sickness | 25,6 | 12,7 | <0,0001 | 1,12 | 0,87 | **sig** |
| Aufgabenzeit | 38,1 | 29,3 | <0,0001 | 1,36 | 0,99 | **sig** |
| **Comfort** | 4,5 | 5,0 | **0,027** | **−0,50** | −0,52 | **n.s.** ← kippt |

**Die drei Lektionen, die daraus fallen:**

1. **Effektstärke entlarvt den wackligen Befund.** Comfort ist roh signifikant (p=0,027), aber
   die **einzige** Zielgröße mit *kleiner* Effektstärke (dz 0,50 vs. > 1,1). Sie **kippt unter
   Bonferroni** ($\alpha/4 = 0{,}0125$). Beides — Korrektur *und* Effektstärke — zeigt in
   dieselbe Richtung: **dem Comfort-Befund sollte man nicht trauen.** *(Holm, weniger konservativ,
   behält ihn knapp — genau deshalb berichtet man immer die Effektstärke dazu, nicht nur die
   Signifikanz. Die Entscheidung hängt von der Korrektur ab; die Effektstärke ist der stabilere
   Kompass.)* Das ist derselbe Gedanke wie der **Base-Rate-Fallacy** aus Modul 15: viele Tests ×
   kleine Fehlerrate = viele Fehlalarme.

2. **Counterbalancing war nötig — und das lässt sich zeigen.** Es *gibt* Reihenfolgeeffekte:
   die zweite Sitzung ist übler (Carryover: 16,0 → 22,3) und schneller (Lernen: 35,8 → 31,6).
   Weil counterbalanced wurde, mitteln sie sich über die Bedingungen heraus.

3. **Ohne Counterbalancing wäre das Ergebnis falsch.** Die Gegenprobe (alle machen 3 DoF zuerst,
   6 DoF zweite): Der gemessene Sickness-Unterschied beträgt nur noch **5,0** statt der wahren
   **12** Punkte — der Carryover auf die zweite (6-DoF-)Sitzung **maskiert die Hälfte des echten
   Effekts.** Man hätte 6 DoF fälschlich als kaum besser eingestuft. **Ein Studiendesign-Fehler,
   den keine noch so gute Statistik hinterher reparieren kann.**

Laufzeit: **~1 s**.

## Referenzlösung

Vollständig und getestet in [`solution/`](solution/) (Generator, Statistik-Werkzeuge, `run.py`,
**17 Tests**). **Erst selbst versuchen!** Die Tests rechnen die Effektstärken von Hand nach und
prüfen, dass die Analyse die *eingebaute Wahrheit* wiederfindet — inklusive der Maskierung ohne
Counterbalancing.

```bash
/.../xtAI_Learn-Repository/.venv/bin/python solution/test_study.py   # 17 Tests, ~1 s
/.../xtAI_Learn-Repository/.venv/bin/python solution/run.py          # Auswertung + Plot, ~1 s
```
`numpy`, `pandas`, `scipy` (+ `matplotlib`). CPU, keine Hardware.

## Erweiterungen (für die besonders Motivierten)

- **Power-Analyse vorher** (Skript 5.2): Wie viele Probanden bräuchte man, um einen Effekt von
  dz = 0,5 mit 80 % Power zu finden? (Für den paired-t: $N \approx (z_{1-\alpha/2}+z_{1-\beta})^2/d_z^2$.)
  War N=24 für Comfort ausreichend?
- **Nichtparametrisch vs. parametrisch**: Konstruiere einen Fall (ein Ausreißer-Proband), in dem
  t-Test und Wilcoxon **unterschiedliche** Schlüsse ziehen. Welcher ist hier vertrauenswürdiger?
- **Between-Subject-Variante**: Simuliere dieselbe Studie between-subject (jeder nur eine
  Bedingung). Wie viel mehr Probanden braucht man für dieselbe Power? Warum?
- **SSQ-Subskalen** (Skript 5.1): Zerlege Sickness in *Nausea*, *Oculomotor*, *Disorientation* —
  jetzt hast du noch mehr Tests. Wie stark schlägt Bonferroni zu?
- **Simpson-artige Falle**: Baue eine Untergruppe (z. B. „VR-erfahren"), bei der der Effekt
  umgekehrt ist. Verschwindet er im Gesamtmittel?
- **Bootstrap-Konfidenzintervall** (Modul 03) für die mittlere Differenz statt/zusätzlich zum
  p-Wert — die modernere Art zu berichten.
