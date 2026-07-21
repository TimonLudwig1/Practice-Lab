# Projekt 03 (final) — Nachfrage erklären und vorhersagen (Regression & Zeitreihen)

**Format:** Jupyter Notebook (`regression_forecasting.ipynb`) + Download-Skript.
**Warum dieses Format?** Modelldiagnose lebt von Plots direkt am Code (Residuen, Prognose vs. Realität) — und das Projekt erzählt eine durchgehende Analyse-Geschichte, die man als Bericht lesen können soll.

**Daten: echt.** Dieselben Bike-Sharing-Daten (UCI, Capital Bikeshare Washington D.C.) wie im Final-Projekt von Modul 02, diesmal auf **Tagesebene** (`day.csv`, 731 Tage). Der bewusste Anschluss ans bekannte Szenario ist Teil des Designs: Aus „beschreiben" (Modul 02) wird „erklären und vorhersagen" (Modul 03). Bonus der echten Daten: Der Testzeitraum enthält **Hurrikan Sandy** — die perfekte Lektion über Modellgrenzen.

## Ziel

1. **Erklären:** Einfache → quadratische → multiple Regression auf die Tagesnachfrage; Koeffizienten ceteris paribus interpretieren; per Residuenplot die Nichtlinearität finden (Wohlfühltemperatur ≈ 29 °C).
2. **Vorhersagen:** Lag-Features bauen, zeitlich splitten, gegen naive/saisonale Baselines antreten (Modell-MAE ≈ 859 vs. 952/1458).
3. **Leakage-Demo:** Zufalls-Split liefert scheinbar MAE ≈ 625 — und du kannst erklären, warum das gelogen ist.

## Vorwissen

- Modul-Skript Abschnitte 1.1–1.2, 2.1, 2.2 (Pflicht)
- Modul 02 komplett (v. a. das Bike-Sharing-Final-Projekt)
- scikit-learn wird hier zum ersten Mal benutzt; die nötige API (`fit`/`predict`/`score`) ist in den Aufgaben-Hinweisen des Notebooks beschrieben (Code schreibst du selbst)

## Aufgaben

1. Daten holen: `python datasets/download_data.py` (oder `day.csv` aus Modul 02 kopieren).
2. Notebook durcharbeiten: einfache Regression → Residuenplot → Quadratterm → multiple Regression (jeden Koeffizienten in einem Satz interpretieren!) → Lag-Features + zeitlicher Split + Baselines → Leakage-Demo → Fazit an die Betriebsleitung.
3. Die drei schlechtesten Prognosetage anschauen und erklären (Sandy!).

## Was am Ende funktionieren soll

- $R^2$: 0,394 (linear) → 0,453 (quadratisch) → 0,794 (multiple); Scheitel bei 25–31 °C.
- MAE-Rangfolge: Modell (≈859) < naiv (≈952) < saisonal-naiv (≈1458); deine Selbstchecks (in den Aufgaben genannte Wertebereiche) treffen zu.
- Zufalls-Split-MAE ≈ 625 — mit deiner Erklärung, warum dieser Wert nicht zählt.
- Ein 4–5-Punkte-Fazit in Alltagssprache.

## Lösung

Vollständig ausgeführte Musterlösung: [`solution/solution.ipynb`](solution/solution.ipynb).
