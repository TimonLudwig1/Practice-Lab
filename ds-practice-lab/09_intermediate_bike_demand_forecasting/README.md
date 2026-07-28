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

---

# Deutsche Übersetzung

# 09 — Zeitreihen: Prognose der Fahrradnachfrage 🚲

Schwierigkeit: 🟡 Mittel | Thema: Zeitreihenprognose

## 🎯 Projektziel
Prognostiziere die tägliche Nachfrage nach Leihfahrrädern und lerne die zentrale Regel der Zeitreihenanalyse: **Die zeitliche Reihenfolge muss bei Merkmalen, Validierung und Baselines berücksichtigt werden.**

## 📊 Beschreibung des Datensatzes
Der **UCI Bike Sharing Dataset** enthält 731 Tagesbeobachtungen aus den Jahren 2011 und 2012 zum Capital-Bikeshare-System in Washington, D.C. Neben der Anzahl der Ausleihen `cnt` gibt es Wetterwerte, Jahreszeit sowie Feiertags- und Arbeitstagsmerkmale.

Download: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip. Entpacke `day.csv` nach `data/raw/`; das Notebook erwartet die Datei dort.

## 💡 Empfohlenes Vorgehen
1. Führe eine EDA mit Zeit auf der x-Achse durch: vollständige Reihe, saisonale Muster, Wochentagsprofile und Wetter im Verhältnis zur Nachfrage. Übe `ax.plot` mit Datumsachsen und `fig.autofmt_xdate()`.
2. Erstelle zuerst **naive Baselines**: den Wert des Vortags sowie den Wert von vor sieben Tagen. Jedes Modell muss diese einfachen Vorhersagen schlagen.
3. Erzeuge verzögerte und rollierende Merkmale wie Lag 1, Lag 7 und den gleitenden 7-Tage-Mittelwert sowie Kalender- und Wettermerkmale. Prüfe sorgfältig, welche Informationen zum Vorhersagezeitpunkt tatsächlich verfügbar wären.
4. Validiere mit einer **zeitbasierten Aufteilung** und niemals durch zufälliges Mischen. Verwende `sklearn.model_selection.TimeSeriesSplit` und verstehe, warum `train_test_split` hier Informationen aus der Zukunft verraten würde.
5. Vergleiche lineare Regression mit einem Gradient-Boosting-Baum. Verwende MAE und MAPE als Metriken.
6. Untersuche die Tage mit den größten Prognosefehlern und suche Gemeinsamkeiten, insbesondere Wetterereignisse und Feiertage Ende 2012.
7. Als Erweiterung kannst du auf derselben Aufteilung ein klassisches statistisches Modell wie `statsmodels` SARIMAX testen.

## 🏁 Erfolgskriterien
- Beide naiven Baselines sind implementiert und dokumentiert.
- Alle Merkmale sind nachweislich zum Prognosezeitpunkt verfügbar; begründe dies je Merkmalsgruppe in einem Satz.
- Zeitbasierte Validierung; der abschließende MAE ist mindestens 15 % besser als die saisonal-naive Baseline.
- Eine Abbildung von Prognose und tatsächlichen Werten im Testzeitraum sowie eine kurze Analyse der schlechtesten Tage.

## 🔗 Nützliche Quellen
- [Dokumentation zu TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- Suchbegriffe: *seasonal naive forecast*, *lag features*, *data leakage in time series*
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/), Kapitel 1, 5 und 7
