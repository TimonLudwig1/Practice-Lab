# 10 — Unsupervised Learning: Customer Segmentation 🛒

Difficulty: 🟡 Intermediate | Topic: Clustering / Unsupervised Learning

## 🎯 Project Goal
Segment e-commerce customers from raw transaction logs using RFM features and k-means — and learn the uncomfortable truth of unsupervised learning: *there is no accuracy score; you must argue your clusters make sense.*

## 📊 Dataset Description
**UCI Online Retail** — ~540k transaction line items from a UK online retailer (Dec 2010–Dec 2011): invoice number, product, quantity, price, customer ID, country.

Download: https://archive.ics.uci.edu/static/public/352/online+retail.zip → unzip the `.xlsx` into `data/raw/`. Loading takes a minute (`pd.read_excel`, needs `openpyxl`); cache as parquet/csv in `data/processed/` after first load. The notebook has a loading snippet.

Data quality is intentionally messy: cancellations (invoices starting with "C"), negative quantities, missing customer IDs — cleaning is part of the project.

## 💡 Suggested Approach (high-level)
1. Clean: remove cancellations, rows without customer ID, non-product entries; create a revenue column.
2. Aggregate transactions → one row per customer with **RFM**: Recency (days since last purchase), Frequency (number of orders), Monetary (total revenue).
3. RFM values are heavily skewed — look at the distributions and deal with it (log transform) before scaling. K-means is distance-based; skew + no scaling = one feature dominates everything.
4. Choose k: elbow method on inertia AND silhouette score. They may disagree — discuss.
5. Run k-means, then **profile the clusters**: mean R/F/M per cluster, cluster sizes, and give each segment a marketing-style name ("champions", "at-risk", "one-time buyers", …).
6. Visualize: 2D scatter of two RFM dimensions colored by cluster, plus a heatmap/bar chart of normalized cluster profiles.
7. Sanity checks: rerun with different random seeds — are the segments stable? Try `k±1` — does the story change?
8. Stretch: compare with DBSCAN or hierarchical clustering; what do they do differently with the outlier "whale" customers?

## 🏁 Success Criteria
- Documented cleaning decisions (how many rows removed, why)
- k chosen with explicit reasoning from elbow + silhouette evidence
- Each cluster profiled and named, with a one-line recommended marketing action per segment
- Stability check across seeds reported
- All figures in matplotlib with readable labels

## 🔗 Useful References
- [sklearn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) & [silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- Look up: *RFM analysis*, *elbow method*, *why k-means needs scaling*
- `df.groupby("CustomerID").agg(...)` — the workhorse of step 2

---

# Deutsche Übersetzung

# 10 — Unüberwachtes Lernen: Kundensegmentierung 🛒

Schwierigkeit: 🟡 Mittel | Thema: Clustering und unüberwachtes Lernen

## 🎯 Projektziel
Segmentiere E-Commerce-Kunden aus rohen Transaktionsdaten mithilfe von RFM-Merkmalen und k-Means. Dabei lernst du eine schwierige Eigenschaft des unüberwachten Lernens: *Es gibt keine Accuracy; du musst begründen, warum deine Cluster sinnvoll sind.*

## 📊 Beschreibung des Datensatzes
**UCI Online Retail** enthält etwa 540.000 Transaktionspositionen eines britischen Onlinehändlers von Dezember 2010 bis Dezember 2011. Zu den Spalten gehören Rechnungsnummer, Produkt, Menge, Preis, Kunden-ID und Land.

Download: https://archive.ics.uci.edu/static/public/352/online+retail.zip. Entpacke die `.xlsx`-Datei nach `data/raw/`. Das Einlesen mit `pd.read_excel` und `openpyxl` dauert etwas; speichere danach eine Parquet- oder CSV-Kopie unter `data/processed/`.

Die Daten enthalten absichtlich Probleme wie Stornierungen, negative Mengen und fehlende Kunden-IDs. Die Bereinigung ist Teil des Projekts.

## 💡 Empfohlenes Vorgehen
1. Entferne Stornierungen, Zeilen ohne Kunden-ID und Einträge, die keine Produkte darstellen, und erzeuge eine Umsatzspalte.
2. Verdichte die Transaktionen auf eine Zeile pro Kunde mit **RFM**: Recency als Zeit seit dem letzten Kauf, Frequency als Anzahl der Bestellungen und Monetary als Gesamtumsatz.
3. RFM-Werte sind stark schief verteilt. Untersuche die Verteilungen, transformiere sie beispielsweise logarithmisch und skaliere sie. Da k-Means abstandsbasiert ist, würde sonst ein Merkmal dominieren.
4. Wähle k anhand von Ellenbogenmethode und Silhouettenwert. Falls beide widersprechen, diskutiere die Abwägung.
5. Führe k-Means aus und **beschreibe die Cluster** anhand der mittleren R-, F- und M-Werte sowie ihrer Größen. Vergib verständliche Namen wie „Champions“, „gefährdet“ oder „Einmalkäufer“.
6. Visualisiere zwei RFM-Dimensionen als nach Clustern eingefärbtes Streudiagramm sowie normalisierte Clusterprofile als Heatmap oder Balkendiagramm.
7. Prüfe die Stabilität mit verschiedenen Zufalls-Seeds und mit `k±1`. Untersuche, ob sich die inhaltliche Geschichte verändert.
8. Als Erweiterung kannst du DBSCAN oder hierarchisches Clustering vergleichen und untersuchen, wie sie mit extrem umsatzstarken Ausreißern umgehen.

## 🏁 Erfolgskriterien
- Dokumentierte Bereinigungsentscheidungen einschließlich Anzahl und Grund entfernter Zeilen
- Begründete Wahl von k anhand von Ellenbogen- und Silhouettenanalyse
- Beschreibung und Benennung jedes Clusters mit einer empfohlenen Marketingmaßnahme pro Segment
- Dokumentierte Stabilitätsprüfung über mehrere Seeds
- Sämtliche Abbildungen in Matplotlib mit lesbaren Beschriftungen

## 🔗 Nützliche Quellen
- [sklearn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) und [silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- Suchbegriffe: *RFM analysis*, *elbow method*, *why k-means needs scaling*
- `df.groupby("CustomerID").agg(...)` als zentraler Arbeitsschritt von Schritt 2
