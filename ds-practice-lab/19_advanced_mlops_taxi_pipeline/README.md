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
