# 15 — Applied Operations: Forecast-Driven Inventory Optimization `[from your lectures]`

Difficulty: 🔴 Hard | Topic: Operations Analytics (Forecasting + Simulation + Optimization)

## 🎯 Project Goal
You are the data scientist for a retail chain. Build the full operations pipeline: forecast SKU-level demand, feed the forecasts into an inventory simulation, and optimize the replenishment policy per SKU to minimize total cost at a required service level.

## 📊 Dataset Description
**Kaggle: Store Item Demand Forecasting Challenge** — 5 years of daily sales for 50 items across 10 stores (~913k rows, clean, no missing values).
Download: https://www.kaggle.com/c/demand-forecasting-kernels-only/data (requires a free Kaggle account; use `train.csv`).

Scope to keep it tractable: pick **one store and 10 items** of varying volume.

System assumptions (your simulation must implement these):
- Daily review; orders placed at end of day arrive after a 7-day lead time
- Costs per item: holding 0.02€/unit/day, ordering 8€ fixed per order line, stockout = lost sale at the item's margin (assume margin = 30% of an assumed price; document your price assumptions)
- Management requires ≥95% fill rate per item

## 📏 Evaluation Metric
- Forecasting: MAE per item on the final 90 days (held out from all tuning)
- Operations: total cost (holding + ordering + stockout) over the 90-day evaluation window, simulated with your policy driven only by information available at decision time
- Constraint: fill rate ≥ 95% per item — solutions violating it are infeasible regardless of cost

## 🏁 Success Criteria
- Demand forecasts that beat a seasonal-naive baseline per item
- A discrete-event or day-loop simulation of the inventory system, validated against hand-computed toy cases
- Replenishment policy parameters optimized per item (any method: grid search over (s,S), newsvendor-style analytics, or smarter)
- Final report: cost breakdown per item, achieved fill rates, and the cost of the 95% service-level constraint (how much would relaxing it to 90% save?)
- Honest discussion of where forecast error hurts the inventory decision most
