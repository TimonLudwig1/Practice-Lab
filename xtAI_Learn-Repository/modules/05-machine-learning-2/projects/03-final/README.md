# Project 03 (final): Customer segmentation — the unsupervised learning capstone

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

## Goal

The **final project** of module 05. You consolidate the complete unsupervised part of the module on a real data set and answer a real business question:

> A Portuguese wholesale distributor knows its 440 customers only through their annual spending in six product categories. **Are there natural customer segments — how many, what do they look like, and how sure are we?**

You go through the full pipeline: **EDA → preprocessing → dimensionality reduction (PCA, t-SNE) → four clustering methods (k-means, GMM/EM, DBSCAN, Ward) → model selection (silhouette, BIC, ARI) → business interpretation.**

The actual lesson is that the methods **partly contradict each other** — and that you can justify which one you believe, and why.

**Why this format (Jupyter notebook):** segmentation lives on the interplay of plot, metric and interpretation text side by side.

**Why real data (UCI *Wholesale Customers*):** compact (440×6, runs in seconds on any CPU), but with all the real pitfalls — strong right skew, overlapping clusters and a held-back external truth (`Channel`: Horeca vs. retail) against which you can validate your purely data-driven result honestly. That is exactly what makes the data set ideal for pitting the methods of the module *against each other*. The download (about 15 KB) happens automatically in the first notebook cell and is excluded from the repository via `.gitignore`.

## Prior knowledge

- Projects 01 and 02 of this module.
- Script sections: **2.6** (k-means), **2.7** (mixture models and EM, BIC), **2.8** (hierarchical and DBSCAN), **2.9** (PCA), **2.10** (t-SNE).
- External validation of clusters (adjusted Rand index) — introduced in the notebook.

## Setup

```bash
source .venv/bin/activate
jupyter lab modules/05-machine-learning-2/projects/03-final/customer_segmentation.ipynb
```

Pure CPU, no GPU needed. Runtime of the whole notebook (including t-SNE): under a minute. The required libraries (`numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`) are already in the repository `requirements.txt`.

## Tasks (step by step)

Unlike in the first two projects, there is **no prescribed solution code** here — only the data infrastructure (the download) is given. You write everything yourself. Every task comes with a **reflection question** to be answered in writing.

1. **EDA and preprocessing:** show the right skew, explain why clustering on the raw data fails, and apply `log1p` + standardization.
2. **Dimensionality reduction:** PCA (scree plot, biplot, interpret the loadings), optionally t-SNE.
3. **Four clustering methods:** k-means (elbow + silhouette), GMM/EM (BIC over the number of components *and* the covariance type), DBSCAN (k-distance plot), Ward (dendrogram) — every hyperparameter chosen **with a justification**.
4. **Validation:** a comparison table with internal measures (silhouette, Davies-Bouldin, Calinski-Harabasz) **and** the external adjusted Rand index against the held-back `Channel`.
5. **Name the segments and give a business recommendation:** median spending profiles, telling names, a concrete recommended action.

**Important:** the columns `Channel` and `Region` are **never used for the clustering** — only for the validation at the end.

## What should work in the end

- All four methods run, and every hyperparameter is justified (not guessed).
- Your comparison table contains internal measures **and** the ARI against `Channel`.
- You can explain in words:
  - why **k-means, GMM and Ward agree on about 2 main segments**,
  - why the **GMM with full covariance** reaches the best ARI (about 0.6) — which shape of the data matches its assumption,
  - why **DBSCAN yields no meaningful structure** here (which underlying assumption is violated),
  - why the best internal measure (BIC) and the best external measure (ARI) do **not** have to be the same solution.
- Two named segments (gastronomy/Horeca vs. retail) with a median profile and a business recommendation.

## Reference solution

[`solution/solution.ipynb`](solution/solution.ipynb) — fully executed, with all plots, the comparison table and detailed interpretation texts. Try it yourself first, then compare.

---
---

# Projekt 03 (final): Kundensegmentierung — Unsupervised-Learning-Capstone (deutsche Fassung)

## Ziel

Das **Abschlussprojekt** von Modul 05. Du konsolidierst den kompletten unüberwachten Teil des Moduls an einem echten Datensatz und beantwortest eine reale Geschäftsfrage:

> Ein portugiesischer Großhandels-Distributor kennt seine 440 Kunden nur über deren jährliche Ausgaben in sechs Produktkategorien. **Gibt es natürliche Kundensegmente — wie viele, wie sehen sie aus, und wie sicher sind wir uns?**

Du durchläufst die volle Pipeline: **EDA → Vorverarbeitung → Dimensionsreduktion (PCA, t-SNE) → vier Clustering-Verfahren (k-Means, GMM/EM, DBSCAN, Ward) → Modellwahl (Silhouette, BIC, ARI) → geschäftliche Interpretation.**

Die eigentliche Lektion ist, dass die Verfahren sich **teilweise widersprechen** — und dass du begründen kannst, welchem du warum glaubst.

**Warum dieses Format (Jupyter Notebook):** Segmentierung lebt vom Wechselspiel aus Plot, Kennzahl und Interpretationstext direkt nebeneinander.

**Warum echte Daten (UCI *Wholesale Customers*):** Kompakt (440×6, läuft in Sekunden auf jeder CPU), aber mit allen realen Tücken — starke Rechtsschiefe, überlappende Cluster und eine zurückgehaltene externe Wahrheit (`Channel`: Horeca vs. Retail), gegen die du dein rein datengetriebenes Ergebnis ehrlich validieren kannst. Genau das macht den Datensatz ideal, um die Verfahren des Moduls *gegeneinander* zu stellen. Der Download (~15 KB) passiert automatisch in der ersten Notebook-Zelle und wird per `.gitignore` vom Repo ausgeschlossen.

## Vorwissen

- Projekt 01 & 02 dieses Moduls.
- Skript-Abschnitte: **2.6** (k-Means), **2.7** (Mixture Models & EM, BIC), **2.8** (Hierarchisch & DBSCAN), **2.9** (PCA), **2.10** (t-SNE).
- Externe Validierung von Clustern (Adjusted Rand Index) — wird im Notebook eingeführt.

## Setup

```bash
source .venv/bin/activate
jupyter lab modules/05-machine-learning-2/projects/03-final/customer_segmentation.ipynb
```

Reine CPU, keine GPU nötig. Laufzeit des ganzen Notebooks (inkl. t-SNE): unter einer Minute. Benötigte Bibliotheken (`numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`) sind bereits in der Repo-`requirements.txt`.

## Aufgabenstellung (Schritt für Schritt)

Anders als in den ersten beiden Projekten gibt es hier **keinen vorgeschriebenen Lösungscode** — nur die Dateninfrastruktur (Download) ist vorgegeben. Du schreibst alles selbst. Zu jeder Aufgabe gehört eine schriftlich zu beantwortende **Reflexionsfrage**.

1. **EDA & Vorverarbeitung:** Rechtsschiefe zeigen, begründen, warum rohes Clustering scheitert, und `log1p` + Standardisierung anwenden.
2. **Dimensionsreduktion:** PCA (Scree, Biplot, Ladungen interpretieren), optional t-SNE.
3. **Vier Clustering-Verfahren:** k-Means (Elbow + Silhouette), GMM/EM (BIC über Komponenten *und* Kovarianztyp), DBSCAN (k-Distanz-Plot), Ward (Dendrogramm) — jeder Hyperparameter **begründet** gewählt.
4. **Validierung:** Vergleichstabelle mit internen Maßen (Silhouette, Davies-Bouldin, Calinski-Harabasz) **und** dem externen Adjusted Rand Index gegen den zurückgehaltenen `Channel`.
5. **Segmente benennen & Geschäftsempfehlung:** Median-Ausgabenprofile, sprechende Namen, konkrete Handlungsempfehlung.

**Wichtig:** Die Spalten `Channel` und `Region` werden **nie fürs Clustering** benutzt — nur zur Validierung am Ende.

## Was am Ende funktionieren soll

- Alle vier Verfahren laufen, jeder Hyperparameter ist begründet (nicht geraten).
- Deine Vergleichstabelle enthält interne Maße **und** den ARI gegen `Channel`.
- Du kannst in Worten erklären:
  - warum **k-Means, GMM und Ward übereinstimmend ~2 Hauptsegmente** finden,
  - warum das **GMM mit voller Kovarianz** den besten ARI (~0,6) erreicht — welche Datenform seine Annahme trifft,
  - warum **DBSCAN hier keine sinnvolle Struktur** liefert (welche Grundannahme verletzt ist),
  - warum bestes internes Maß (BIC) und bestes externes Maß (ARI) **nicht** dieselbe Lösung sein müssen.
- Zwei benannte Segmente (Gastronomie/Horeca vs. Einzelhandel/Retail) mit Median-Profil und einer Geschäftsempfehlung.

## Musterlösung

[`solution/solution.ipynb`](solution/solution.ipynb) — vollständig ausgeführt, mit allen Plots, der Vergleichstabelle und ausführlichen Interpretationstexten. Erst selbst versuchen, dann vergleichen.
