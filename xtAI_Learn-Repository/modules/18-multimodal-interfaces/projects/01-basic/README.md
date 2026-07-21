# P01 (basic) — Inverse-Varianz-Fusion: Wie das Gehirn zwei Sinne verschmilzt

**Modul 18 — Multimodal Interfaces** · Format: **Jupyter-Notebook**

## Ziel

Du rechnest das mathematische Herzstück multimodaler Fusion selbst nach — die **inverse-Varianz-Gewichtung** (Maximum-Likelihood-Integration) — und reproduzierst damit die drei Kernbefunde des berühmten Experiments von **Ernst & Banks (2002)**:

1. **Präzisionsgewinn**: Die Fusion zweier verrauschter Modalitäten ist *immer* präziser als jede einzelne ($\sigma_{fus}^2 \le \min$).
2. **Reliabilitätsadaptiver Wahrnehmungs-Shift**: Bei einem Sinneskonflikt wandert die Wahrnehmung zur *zuverlässigeren* Modalität — der eigentliche Clou.
3. **Diskriminationsgewinn**: Die Unterscheidungsschwelle (JND) sinkt.

Das ist zugleich die direkte Fortsetzung des **Komplementärfilters aus Modul 17** (Gyro + Accel): dieselbe Formel, andere Sinne.

## Warum dieses Format?

Ein **Notebook**, weil Formel, Simulation, Zahlen und Plots hier zusammengehören: Man sieht die Verteilungen schmaler werden und die Wahrnehmung kippen. Das ist der Kern der Anschauung.

## Warum synthetische Daten?

Das Konzept ist ein **Gesetz über Rauschen und Reliabilität**, kein Datensatz-Phänomen. Simulierte Gauß-Messungen mit *bekannten* Varianzen erlauben, die Theorie (inverse-Varianz-Formel) direkt gegen die Empirie (Stichproben-Streuung) zu prüfen — mit echten Sensordaten kennte man die wahren Varianzen nicht und könnte den Abgleich nicht sauber führen. Fester Seed (`np.random.default_rng(42)`) macht alles reproduzierbar.

## Vorwissen

Gauß-Verteilung, Varianz, etwas Wahrscheinlichkeitsrechnung. Lies **Kapitel 8–9** des [Modul-18-Skripts](../../README.md) (inverse-Varianz-Gewichtung, Ernst & Banks).

## Aufgabenstellung (Schritt für Schritt)

Öffne `cue_integration.ipynb`. Die meisten Zellen sind vorgegeben; an den `# TODO`-Stellen füllst du die Kernformeln ein:

- **Teil A** — die inverse-Varianz-**Gewichte** $w_V, w_H$, die **fusionierte Schätzung** und $\sigma_{fus}$. Zeige, dass die Fusion präziser ist als der bessere Sensor.
- **Teil B** — der **vorhergesagte Wahrnehmungswert** bei Konflikt aus den Gewichten. Beobachte, wie derselbe Konflikt (53 vs. 57 mm) *anders* wahrgenommen wird, je nachdem, ob das Sehen scharf oder verrauscht ist.
- **Teil C** (vorgegeben) — der **Diskriminationsgewinn** (2AFC, JND).
- **Teil D** (Text) — der Bogen zurück zu Modul 17.

## Was am Ende funktionieren soll

Das ausgeführte Notebook zeigt:
- Gewichte $w_V=0.8$, $w_H=0.2$; $\sigma_{fus}\approx 0.894$ mm $<$ 1 mm (empirische `std(fused)` trifft das).
- Wahrnehmungs-Shift von **53.8 → 56.2 mm** allein durch Verrauschen des Sehens.
- JND der Fusion (**0.85 mm**) kleiner als die des besten Einzelsinns (Sehen, 0.95 mm).
- Zwei Plots: schmalere Fusionsverteilung; Wahrnehmung, die vom Sehen zum Tasten kippt.

## Setup

Aus dem Repo-Wurzelverzeichnis (venv wie im Repo eingerichtet):

```bash
.venv/bin/python -m jupyter lab
```

Dann `cue_integration.ipynb` öffnen und Zelle für Zelle ausführen. Benötigt nur `numpy`, `scipy`, `matplotlib` (alle in der `.venv` vorhanden). Laufzeit: wenige Sekunden.

## Lösung

Die vollständige, ausgeführte Lösung liegt in [`solution/cue_integration_solution.ipynb`](solution/cue_integration_solution.ipynb) — **erst selbst probieren!**

## Weiter geht's

- **P02 (medium)**: early vs. late Fusion in der Klassifikation + Mutual Disambiguation — inklusive der Falle korrelierter Fehler.
- **P03 (final)**: ein vollständiger „Put-that-there"-Interpreter mit zeitlicher Fusion.
