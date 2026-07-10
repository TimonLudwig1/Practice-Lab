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
