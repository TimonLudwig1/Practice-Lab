# 17 — Deep Learning Time Series: Household Energy Forecasting (TensorFlow)

Difficulty: 🔴 Hard | Topic: Deep Learning for Sequences (TensorFlow/Keras)

## 🎯 Project Goal
Forecast household electricity consumption 24 hours ahead using TensorFlow/Keras sequence models — and determine, with rigorous baselines, whether deep learning actually earns its complexity on this problem.

## 📊 Dataset Description
**UCI Individual Household Electric Power Consumption** — one household, one-minute resolution, December 2006 to November 2010 (~2M rows): global active power, sub-metering for kitchen/laundry/heating, voltage.
Download: https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip

Resample to hourly before modeling. The file has its own parsing quirks (separator, missing-value encoding) — handling them is part of the job.

## 📏 Evaluation Metric
- Task: predict hourly `Global_active_power` for the next 24 hours, rolling over a held-out final year
- Metrics: MAE and RMSE across the test year, reported per forecast horizon (h+1 vs h+24)
- Mandatory baselines: seasonal-naive (same hour yesterday / same hour last week) and a gradient-boosted tree on lag features — the deep model's value is measured **relative to these**

## 🏁 Success Criteria
- Clean hourly dataset with documented handling of missing periods
- A `tf.data` (or equivalent) windowing pipeline producing (input window → 24-step target) samples without leakage
- At least two TensorFlow architectures trained and compared (e.g., LSTM/GRU vs 1D-CNN or seq2seq), with sensible callbacks (early stopping) and learning curves shown
- Final comparison table: naive vs boosted trees vs both deep models, per-horizon MAE
- A verdict you can defend: did deep learning win, and if not, what would it take (more households? exogenous weather data?)
- Training code in `.py` modules under a structure you design yourself; notebooks only for exploration and reporting
