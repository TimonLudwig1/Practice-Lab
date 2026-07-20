# Abhängigkeitsanalyse der synthetischen Datenpipeline

## Überblick

- Seed: `1203`
- Tasks: **24**
- Abhängigkeiten: **39**
- Quellen: `ingest_customers`, `ingest_orders`, `ingest_products`
- Senken: `notify_complete`
- DAG-Prüfung: **bestanden** (Kahn verarbeitet alle Tasks)

## Gültige Ausführungsreihenfolge

1. `ingest_customers`
2. `ingest_orders`
3. `ingest_products`
4. `validate_customers`
5. `validate_orders`
6. `validate_products`
7. `archive_raw`
8. `clean_customers`
9. `clean_orders`
10. `clean_products`
11. `join_sales`
12. `aggregate_daily`
13. `build_customer_features`
14. `build_product_features`
15. `quality_sales`
16. `train_churn_model`
17. `train_demand_model`
18. `publish_dashboard`
19. `score_customers`
20. `forecast_demand`
21. `quality_models`
22. `publish_scores`
23. `publish_forecast`
24. `notify_complete`

Diese Reihenfolge ist eine von möglicherweise mehreren gültigen Toposorts. Unabhängige Tasks dürfen parallel laufen.

## Frühestmögliche Ausführungswellen

| Welle | Tasks |
|---:|---|
| 0 | `ingest_customers`, `ingest_orders`, `ingest_products` |
| 1 | `validate_customers`, `validate_orders`, `validate_products`, `archive_raw` |
| 2 | `clean_customers`, `clean_orders`, `clean_products` |
| 3 | `join_sales` |
| 4 | `aggregate_daily` |
| 5 | `build_customer_features`, `build_product_features`, `quality_sales` |
| 6 | `train_churn_model`, `train_demand_model`, `publish_dashboard` |
| 7 | `score_customers`, `forecast_demand`, `quality_models` |
| 8 | `publish_scores`, `publish_forecast` |
| 9 | `notify_complete` |

## Laufzeitkritischer Pfad

`ingest_orders` → `validate_orders` → `clean_orders` → `join_sales` → `aggregate_daily` → `build_customer_features` → `train_churn_model` → `score_customers` → `publish_scores` → `notify_complete`

Gesamtdauer bei unbegrenzter Parallelität: **113 Minuten**.

Ein Task auf diesem Pfad verzögert bei eigener Verzögerung den frühestmöglichen Pipeline-Abschluss, sofern kein anderer Pfad gleich lang wird.

## Kritische Knoten nach Ausfallreichweite

Hier bedeutet *kritisch*: Ein Ausfall blockiert viele transitive Nachfolger. Der ausgefallene Task selbst wird separat als nicht verfügbar gezählt.

| Rang | Task | direkte Blockaden | blockierte Nachfolger | nicht verfügbar | Anteil |
|---:|---|---:|---:|---:|---:|
| 1 | `ingest_customers` | 2 | 17 | 18 | 75.0 % |
| 2 | `ingest_orders` | 2 | 17 | 18 | 75.0 % |
| 3 | `ingest_products` | 2 | 17 | 18 | 75.0 % |
| 4 | `validate_customers` | 1 | 15 | 16 | 66.7 % |
| 5 | `validate_orders` | 2 | 15 | 16 | 66.7 % |
| 6 | `validate_products` | 1 | 15 | 16 | 66.7 % |
| 7 | `clean_customers` | 2 | 14 | 15 | 62.5 % |
| 8 | `clean_orders` | 2 | 14 | 15 | 62.5 % |

## Interpretation

Der größte einzelne Blast Radius entsteht bei `ingest_customers`: Mit diesem Task sind 18 von 24 Tasks nicht verfügbar.

Die Ausfallanalyse ist reine Erreichbarkeit: Von einem ausgefallenen Knoten aus werden alle Nachfolger per BFS markiert. Das entspricht dem Grundmodell eines Schedulers wie Airflow: Ein Task kann nur laufen, wenn sämtliche Voraussetzungen erfolgreich waren.

Der Blast-Radius-Rang und der laufzeitkritische Pfad beantworten verschiedene Fragen. Ein früher Ingest-Task kann viele Nachfolger blockieren, ohne auf dem längsten Laufzeitpfad zu liegen; ein langer Modell-Task kann den Abschluss bestimmen, obwohl er weniger Nachfolger besitzt.
