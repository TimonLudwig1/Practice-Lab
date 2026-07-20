# T39 – Direkt geschätzte Krankenhaus-Fixed-Effects

## Ausgangslage

Krankenhäuser werden unterschiedlich häufig beobachtet. Dauerhafte Versorgungsqualität beeinflusst sowohl das typische Personalniveau als auch Patientenscores. Ein Dummy-Modell schätzt neben dem gemeinsamen Personaleffekt einen eigenen Krankenhausintercept. Diese Fixed Effects können anschließend beschrieben werden, sind aber selbst unsichere Schätzwerte.

## Lernziele

- einheitsspezifische Fixed Effects aus einer Dummy-Regression rekonstruieren,
- Referenznormalisierung und Zentrierung unterscheiden,
- Standardfehler einer Summe aus Intercept und Dummykoeffizient berechnen,
- Ranglisten und Gruppenvergleiche geschätzter Fixed Effects kritisch beurteilen,
- den Zusammenhang zwischen Paneltiefe und Schätzunsicherheit untersuchen,
- Fixed Effects nicht vorschnell als kausale Einheitseigenschaften interpretieren.

## Aufgaben

1. Schätze Patientenscore auf Pflegepersonal und Krankenhausdummies.
2. Rekonstruiere jeden Krankenhausintercept aus Referenzintercept und gegebenenfalls Dummykoeffizient.
3. Berechne Standardfehler und 95%-Intervalle der Intercepts mit der Kovarianzmatrix.
4. Zentriere die Intercepts für eine normalisierungsunabhängige Darstellung.
5. Vergleiche in der Simulation geschätzte und wahre zentrierte Krankenhausqualität.
6. Untersuche absoluten Schätzfehler und Intervallbreite in Abhängigkeit von der Zahl beobachteter Perioden.
7. Erstelle eine Rangliste und berechne die Überlappung des geschätzten und wahren obersten Quartils.
8. Vergleiche die Verteilung nach urbanem Standort und erkläre, warum dies keine kausale Standortwirkung ist.

## Ausführen

```bash
python3 exercises/T39-analyzing-fixed-effects/starter.py
python3 exercises/T39-analyzing-fixed-effects/solution.py
```

Die Lösung erzeugt `data/hospital_panel.csv`, Ergebnistabellen und `results/analyzing_fixed_effects.png`.

## Denkfragen

- Welche Größe wird durch die Wahl des Referenzkrankenhauses verändert?
- Warum sind Rankings bei kurzen Panels besonders instabil?
- Welche Probleme entstehen, wenn geschätzte Fixed Effects anschließend wie fehlerfrei gemessene Outcomes verwendet werden?
