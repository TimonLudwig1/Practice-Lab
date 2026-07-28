# 19 — Project Brief: Production ML System for Taxi Trip Duration

Difficulty: ⚫ Advanced / Portfolio | Topic: MLOps / End-to-End Pipeline

---

## Project Brief

**From:** CTO, UrbanRide Analytics
**To:** ML Platform Engineer (you)
**Re:** Productionize the trip-duration model

Our prototype notebook predicts NYC taxi trip durations. Your job is not to improve the model — it is to build the **production system around it**: versioned training, a serving API, and monitoring that catches drift before our customers do. We will judge the system, not the RMSE.

## Business Context
Dispatch quotes ETAs to customers at booking time. Quotes more than ~25% off erode trust. Trip patterns shift across months (seasons, events, fare changes), so yesterday's model quietly rots — that is the problem you are solving.

## Data
NYC TLC Yellow Taxi trip records (public parquet files, one per month):
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Use ≥6 consecutive months. Train on month *t*, serve/evaluate on *t+1*, …; later months are your drift scenario (they genuinely drift). Target: trip duration from pickup/dropoff timestamps. Features: only what is known at booking time.

## Technical Constraints
- Python; experiment tracking + model registry with **MLflow**
- Serving: **FastAPI** endpoint `POST /predict` returning duration + model version; p95 latency < 200 ms locally
- All pipeline steps (ingest → validate → featurize → train → evaluate → register) runnable as CLI commands; no manual notebook steps in the production path
- Data validation gate: malformed/out-of-range records must be rejected with logged reasons
- Containerized serving (Dockerfile)
- Tests: unit tests for feature logic + an integration test hitting the live endpoint

## Deliverables
1. Repository with the pipeline as importable modules + CLI entry points
2. MLflow tracking: every training run logged (params, metrics, artifacts); promotion criterion for "production" stage documented and enforced in code
3. Running FastAPI service loading the registered production model
4. Monitoring report: input drift (e.g., PSI/KS on key features) and prediction-quality drift across the held-out months, with an automated alert threshold and a written retraining policy
5. A drift incident walkthrough: show the month where drift bites, show your monitor catching it, retrain, show recovery
6. `README` with architecture diagram and one-command quickstart

## Evaluation Rubric
| Criterion | Bar |
|---|---|
| Reproducibility | Fresh clone → documented commands → working system |
| Pipeline hygiene | No leakage; validation gate demonstrably rejects bad records |
| Serving | Correct, versioned responses; p95 < 200 ms; containerized |
| Monitoring | Drift detected on real months, not synthetic toys; sensible thresholds |
| Engineering quality | Tests pass; modules typed/documented; honest limitations section |

---

# Deutsche Übersetzung

# 19 — Projektauftrag: Produktives ML-System für die Dauer von Taxifahrten

Schwierigkeit: ⚫ Fortgeschritten / Portfolio | Thema: MLOps und vollständige Pipeline

---

## Projektauftrag

**Von:** CTO, UrbanRide Analytics
**An:** ML Platform Engineer
**Betreff:** Produktivsetzung des Modells für Fahrtdauern

Unser Prototyp-Notebook prognostiziert die Dauer von Taxifahrten in New York. Deine Aufgabe ist nicht die Verbesserung des Modells, sondern der Aufbau des **Produktionssystems darum herum**: versioniertes Training, Bereitstellungs-API und Überwachung, die Drift erkennt, bevor Kunden sie bemerken. Bewertet wird das System und nicht der RMSE.

## Geschäftlicher Kontext
Der Dispositionsdienst nennt Kunden bei der Buchung eine erwartete Ankunftszeit. Abweichungen von mehr als etwa 25 % schädigen das Vertrauen. Fahrtmuster verändern sich durch Jahreszeiten, Veranstaltungen und Tarifänderungen, wodurch ältere Modelle unbemerkt schlechter werden. Dieses Problem soll das System lösen.

## Daten
Öffentliche NYC-TLC-Daten zu Yellow-Taxi-Fahrten als monatliche Parquet-Dateien:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Verwende mindestens sechs aufeinanderfolgende Monate. Trainiere auf Monat *t* und stelle das Modell für *t+1* bereit beziehungsweise bewerte es dort. Spätere Monate bilden ein reales Driftszenario. Die Zielvariable ist die Fahrtdauer aus Abhol- und Absetzzeitpunkt; als Merkmale dürfen nur bei der Buchung bekannte Informationen dienen.

## Technische Vorgaben
- Python sowie Experimentverfolgung und Modellregister mit **MLflow**
- Bereitstellung über einen **FastAPI**-Endpunkt `POST /predict`, der Dauer und Modellversion zurückgibt; lokale p95-Latenz unter 200 ms
- Sämtliche Pipelineschritte von Einlesen über Validierung, Merkmalsbildung, Training und Bewertung bis Registrierung als CLI-Befehle; keine manuellen Notebook-Schritte im Produktionspfad
- Datenvalidierung, die fehlerhafte oder unzulässige Datensätze unter Protokollierung des Grundes zurückweist
- Containerisierte Bereitstellung mit Dockerfile
- Unit-Tests für Merkmalslogik und Integrationstest gegen den laufenden Endpunkt

## Liefergegenstände
1. Repository mit der Pipeline als importierbare Module und CLI-Einstiegspunkte
2. MLflow-Verfolgung aller Trainingsläufe mit Parametern, Metriken und Artefakten sowie im Code erzwungenem Kriterium für die Beförderung nach „Production“
3. Laufender FastAPI-Dienst, der das registrierte Produktionsmodell lädt
4. Monitoringbericht über Eingabedrift, beispielsweise PSI oder KS, und Qualitätsdrift der Vorhersagen über die zurückgehaltenen Monate, mit automatischem Alarm und dokumentierter Nachtrainierungsstrategie
5. Ablauf eines Driftvorfalls: betroffenen Monat zeigen, Erkennung durch Monitoring, Nachtraining und nachgewiesene Erholung
6. `README` mit Architekturdiagramm und Schnellstart über einen Befehl

## Bewertungskriterien
| Kriterium | Anforderung |
|---|---|
| Reproduzierbarkeit | Frischer Klon, dokumentierte Befehle und funktionierendes System |
| Pipeline-Qualität | Kein Datenleck; Validierung weist fehlerhafte Datensätze nachweislich zurück |
| Bereitstellung | Korrekte versionierte Antworten, p95 unter 200 ms und Containerisierung |
| Monitoring | Drift in echten Monaten statt synthetischen Beispielen erkannt; sinnvolle Schwellenwerte |
| Engineering-Qualität | Tests bestanden, Module typisiert und dokumentiert, ehrlicher Abschnitt zu Einschränkungen |
