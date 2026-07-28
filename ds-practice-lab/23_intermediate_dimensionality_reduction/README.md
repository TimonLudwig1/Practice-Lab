# 23 — Dimensionality Reduction & Manifold Learning 🗺️

Difficulty: 🟡 Intermediate | Topic: Unsupervised Learning / Dimensionality Reduction

## 🎯 Project Goal
Take high-dimensional data and learn to **see it**. Use PCA as the linear, interpretable workhorse, then contrast it with nonlinear manifold methods (t-SNE / UMAP) — and understand precisely what each one is and isn't telling you. A great matplotlib project: most of the output is plots you have to read correctly.

## 📊 Dataset Description
Primary: **`sklearn.datasets.load_digits()`** — 1797 images of handwritten digits, 8×8 = 64 features each, with labels 0–9. Built in, no download. The labels are used **only to color plots**, never to fit the reduction (these methods are unsupervised).

Stretch dataset: **UCI Human Activity Recognition (HAR)** — 561 sensor features, 6 activity classes. Download from UCI if you want a harder, higher-dimensional case. Notebook loads `digits` by default.

## 💡 Suggested Approach (high-level)
1. Standardize the features, then fit **PCA**. Plot the explained-variance and cumulative-variance curves: how many components capture 90% of the variance? This number alone is a useful story about the data.
2. Project to 2D with PCA and scatter, colored by the (held-out) label. Then visualize what the top principal components *are* — for digits, reshape the component loadings back to 8×8 and view them as images ("eigen-digits").
3. Run **t-SNE** to 2D and compare the scatter to the PCA one. The clusters will look much cleaner — understand *why*, and why that cleanliness is partly an illusion.
4. Read t-SNE critically: vary `perplexity`, re-run, and observe that cluster sizes and inter-cluster distances are **not** meaningful. Write down the three things a t-SNE plot must NOT be used to claim.
5. (Optional) Add UMAP and compare speed + global-structure preservation against t-SNE.
6. Tie it back to a downstream task: does a classifier on the top-k PCA components do nearly as well as on all 64 features? Quantify the accuracy-vs-dimensionality trade-off.

## 🏁 Success Criteria
- Explained-variance + cumulative-variance plots, with the "components for 90% variance" number stated
- A 2D PCA scatter colored by label, **and** the top principal components visualized in the original feature space (eigen-digits)
- A t-SNE 2D scatter compared side by side with the PCA one
- A short, explicit list of what t-SNE distances/sizes do and do not mean, demonstrated by re-running at ≥2 perplexity values
- A downstream check: classifier accuracy on top-k PCA components vs all features, with the trade-off stated in one sentence

## 🔗 Useful References
- [scikit-learn PCA docs](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- ["How to Use t-SNE Effectively" (distill.pub)](https://distill.pub/2016/misread-tsne/) — read this before trusting any t-SNE plot
- Look up: *explained variance ratio*, *t-SNE perplexity*, *curse of dimensionality*, *UMAP vs t-SNE*

---

# Deutsche Übersetzung

# 23 — Dimensionsreduktion und Manifold Learning 🗺️

Schwierigkeit: 🟡 Mittel | Thema: Unüberwachtes Lernen und Dimensionsreduktion

## 🎯 Projektziel
Lerne, hochdimensionale Daten **sichtbar zu machen**. Verwende PCA als lineares und interpretierbares Standardverfahren und vergleiche sie mit nichtlinearen Manifold-Methoden wie t-SNE und UMAP. Verstehe genau, welche Aussagen die Verfahren erlauben und welche nicht.

## 📊 Beschreibung des Datensatzes
Primär wird `sklearn.datasets.load_digits()` verwendet: 1.797 Bilder handgeschriebener Ziffern mit jeweils 8×8 beziehungsweise 64 Merkmalen und Labels von 0 bis 9. Der Datensatz ist eingebaut. Die Labels dienen **nur zum Einfärben der Grafiken** und werden nicht zur Anpassung der unüberwachten Reduktion verwendet.

Als Erweiterung bietet sich der **UCI Human Activity Recognition**-Datensatz mit 561 Sensormerkmalen und sechs Aktivitätsklassen an. Das Notebook lädt standardmäßig `digits`.

## 💡 Empfohlenes Vorgehen
1. Standardisiere die Merkmale und passe **PCA** an. Zeichne erklärte und kumulierte Varianz und bestimme die Anzahl Komponenten für 90 % der Varianz.
2. Projiziere per PCA in zwei Dimensionen und färbe das Streudiagramm nach den zurückgehaltenen Labels. Forme die Ladungen der wichtigsten Komponenten zurück zu 8×8-Bildern, um „Eigen-Ziffern“ zu betrachten.
3. Führe **t-SNE** in zwei Dimensionen durch und vergleiche die Darstellung mit PCA. Verstehe, warum die Cluster sauberer aussehen und weshalb dieser Eindruck teilweise täuscht.
4. Variiere die `perplexity` und beobachte, dass Clustergrößen und Abstände zwischen Clustern **nicht** inhaltlich interpretierbar sind. Notiere drei Aussagen, die aus einer t-SNE-Grafik nicht abgeleitet werden dürfen.
5. Ergänze optional UMAP und vergleiche Geschwindigkeit und Erhalt globaler Strukturen mit t-SNE.
6. Prüfe eine nachgelagerte Aufgabe: Erreicht ein Klassifikator auf den wichtigsten k PCA-Komponenten fast dieselbe Leistung wie auf allen 64 Merkmalen? Quantifiziere den Kompromiss zwischen Accuracy und Dimension.

## 🏁 Erfolgskriterien
- Diagramme der erklärten und kumulierten Varianz mit dokumentierter Komponentenanzahl für 90 % Varianz
- Zweidimensionales PCA-Streudiagramm nach Label sowie Darstellung der wichtigsten Komponenten im ursprünglichen Merkmalsraum
- Direkter Vergleich eines zweidimensionalen t-SNE- und PCA-Streudiagramms
- Explizite Liste interpretierbarer und nicht interpretierbarer Eigenschaften von t-SNE, gezeigt mit mindestens zwei Perplexity-Werten
- Vergleich der Klassifikationsleistung auf den wichtigsten PCA-Komponenten und allen Merkmalen mit kurzem Fazit zum Kompromiss

## 🔗 Nützliche Quellen
- [scikit-learn-Dokumentation zu PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [„How to Use t-SNE Effectively“](https://distill.pub/2016/misread-tsne/) — vor der Interpretation von t-SNE-Grafiken lesen
- Suchbegriffe: *explained variance ratio*, *t-SNE perplexity*, *curse of dimensionality*, *UMAP vs t-SNE*
