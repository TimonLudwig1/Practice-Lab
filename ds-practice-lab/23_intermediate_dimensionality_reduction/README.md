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
