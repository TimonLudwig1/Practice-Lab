# Projekt 03 (final): Kundensegmentierung — Unsupervised-Learning-Capstone

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
jupyter lab module/05-machine-learning-2/projekte/03-final/kundensegmentierung.ipynb
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

[`loesung/loesung.ipynb`](loesung/loesung.ipynb) — vollständig ausgeführt, mit allen Plots, der Vergleichstabelle und ausführlichen Interpretationstexten. Erst selbst versuchen, dann vergleichen.
