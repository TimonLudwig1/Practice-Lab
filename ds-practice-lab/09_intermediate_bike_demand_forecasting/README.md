# 09 — Time Series: Bike Rental Demand Forecasting 🚲

Difficulty: 🟡 Intermediate | Topic: Time Series Forecasting

## 🎯 Project Goal
Forecast daily bike rental demand and learn the core discipline of time series work: **respecting time** in features, validation, and baselines.

## 📊 Dataset Description
**UCI Bike Sharing Dataset** — 731 daily records (2011–2012) from Washington D.C.'s Capital Bikeshare: rental count (`cnt`), plus weather (temp, humidity, windspeed), season, holiday/workingday flags.

Download: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip → unzip into `data/raw/`, you need `day.csv`. (The notebook has a loading snippet assuming that location.)

## 💡 Suggested Approach (high-level)
1. EDA with time on the x-axis: full series, seasonal patterns, weekday profiles, weather vs demand. Learn `ax.plot` with datetime axes and `fig.autofmt_xdate()`.
2. Establish **naive baselines first**: predict yesterday's value; predict the value from 7 days ago. Every model must beat these — many published models don't.
3. Build lag/rolling features: lag-1, lag-7, rolling 7-day mean, plus calendar features (month, weekday) and the weather columns. Think hard about which information would actually be *available at prediction time*.
4. Validate with a **time-based split** — never a random shuffle. Look at `sklearn.model_selection.TimeSeriesSplit` and understand why `train_test_split` would cheat here.
5. Compare: linear regression on your features vs a gradient-boosted tree. Metric: MAE and MAPE.
6. Inspect the worst forecast days. What do they have in common? (Spoiler: look at late 2012 weather events and holidays.)
7. Stretch goal: try a classical statistical model (`statsmodels` SARIMAX) on the same split and compare honestly.

## 🏁 Success Criteria
- Both naive baselines implemented and reported
- All features verifiably available at prediction time (no future leakage — write one sentence per feature group arguing why)
- Time-based validation; final MAE beats the seasonal-naive baseline by ≥15%
- A figure showing forecast vs actuals on the test period, plus a short error analysis of the worst days

## 🔗 Useful References
- [TimeSeriesSplit docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- Look up: *seasonal naive forecast*, *lag features*, *data leakage in time series*
- [Forecasting: Principles and Practice (free book)](https://otexts.com/fpp3/) — chapters 1, 5, 7
