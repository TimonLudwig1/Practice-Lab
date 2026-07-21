"""Kernbausteine fuer P02 (medium) — early vs. late Fusion, Mutual Disambiguation,
Korrelations-Falle. Vollstaendige Loesung.

Modul 18 — Multimodal Interfaces.

Zwei Experimente:
  (1) KOMPLEMENTARITAET: zwei Modalitaeten, jede allein mehrdeutig (~50 %),
      zusammen fast perfekt. Zeigt early- vs. late-Fusion und Missing-Modality.
  (2) REDUNDANZ + KORRELATIONS-FALLE: zwei Modalitaeten schaetzen dasselbe;
      der Fusionsgewinn verschwindet, wenn ihr Rauschen korreliert ist.
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB


# --------------------------------------------------------------------------
# Datensatz 1 — Komplementaritaet (XOR-artig)
# --------------------------------------------------------------------------
def make_complementary(n_per_class=600, sigma=1.0, seed=0):
    """4 Klassen y in {0,1,2,3}. Bit1 = y>>1, Bit0 = y&1.

    Modalitaet A ist NUR ueber Bit1 informativ (trennt {0,1} von {2,3}),
    Modalitaet B ist NUR ueber Bit0 informativ (trennt {0,2} von {1,3}).
    => Jede Modalitaet allein kann hoechstens 50 % erreichen (zwei Klassen
       teilen dieselbe Verteilung). Zusammen sind alle 4 Klassen trennbar.

    Rueckgabe: XA (n,1), XB (n,1), y (n,).
    """
    rng = np.random.default_rng(seed)
    classes = np.repeat(np.arange(4), n_per_class)
    bit1 = classes >> 1          # informativ fuer A
    bit0 = classes & 1           # informativ fuer B
    muA = np.array([-2.0, 2.0])  # Zentren fuer Bit1
    muB = np.array([-2.0, 2.0])  # Zentren fuer Bit0
    XA = (muA[bit1] + rng.normal(0, sigma, classes.size)).reshape(-1, 1)
    XB = (muB[bit0] + rng.normal(0, sigma, classes.size)).reshape(-1, 1)
    perm = rng.permutation(classes.size)
    return XA[perm], XB[perm], classes[perm]


# --------------------------------------------------------------------------
# Datensatz 2 — Redundanz mit einstellbarer Rausch-Korrelation
# --------------------------------------------------------------------------
def make_redundant(n=6000, sep=1.0, sigma=1.0, rho=0.0, seed=0):
    """Binaere Klassifikation. Beide Modalitaeten messen dieselbe Klasse
    c in {-sep, +sep}, jeweils plus Gauss-Rauschen. Das Rauschen der beiden
    Modalitaeten ist mit Korrelationskoeffizient rho korreliert.

      x_A = c + e_A,  x_B = c + e_B,  corr(e_A, e_B) = rho.

    rho=0  -> unabhaengige Fehler -> Fusion hilft maximal.
    rho=1  -> identisches Rauschen -> Fusion bringt NICHTS (beide sagen dasselbe Falsche).

    Rueckgabe: XA (n,1), XB (n,1), y (n,) mit y in {0,1}.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    c = np.where(y == 1, sep, -sep).astype(float)
    # bivariate Normalverteilung fuer (e_A, e_B) mit Korrelation rho
    cov = sigma**2 * np.array([[1.0, rho], [rho, 1.0]])
    e = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    XA = (c + e[:, 0]).reshape(-1, 1)
    XB = (c + e[:, 1]).reshape(-1, 1)
    return XA, XB, y


# --------------------------------------------------------------------------
# Klassifikator je Modalitaet
# --------------------------------------------------------------------------
def fit_modality(X, y):
    """Trainiert einen probabilistischen Klassifikator (GaussianNB) auf EINER
    Modalitaet. Gibt das gefittete Modell zurueck."""
    return GaussianNB().fit(X, y)


# --------------------------------------------------------------------------
# FUSION — der Kern (in der Stub-Version zu implementieren)
# --------------------------------------------------------------------------
def late_fusion_proba(proba_list, prior):
    """Late Fusion per Bayes-Produktregel unter bedingter Unabhaengigkeit:

        P(y | z_A, z_B, ...) proportional zu  prod_m P(y | z_m) / P(y)^(M-1)

    Jeder Faktor P(y | z_m) ist die Posterior einer Modalitaet (n, K).
    Der geteilte Prior P(y) darf nur EINMAL zaehlen, deshalb wird durch
    P(y)^(M-1) geteilt. Anschliessend zeilenweise normalisieren.

    proba_list: Liste von (n, K)-Arrays (je Modalitaet eine Posterior-Matrix).
    prior:      (K,)-Vektor P(y).
    Rueckgabe:  (n, K) fusionierte, normalisierte Posterior.
    """
    M = len(proba_list)
    prior = np.asarray(prior, dtype=float)
    # Produkt der Posteriors
    prod = np.ones_like(proba_list[0], dtype=float)
    for p in proba_list:
        prod = prod * p
    # geteilten Prior herausdividieren (M-1 mal zu viel drin)
    prod = prod / np.maximum(prior[None, :] ** (M - 1), 1e-300)
    # zeilenweise normalisieren
    denom = prod.sum(axis=1, keepdims=True)
    return prod / np.maximum(denom, 1e-300)


def predict_from_proba(proba):
    """argmax je Zeile."""
    return np.asarray(proba).argmax(axis=1)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# --------------------------------------------------------------------------
# Hilfen fuer Early Fusion
# --------------------------------------------------------------------------
def early_fusion_fit_predict(XA_tr, XB_tr, y_tr, XA_te, XB_te):
    """Early Fusion: Feature-Vektoren konkatenieren und EIN Modell trainieren."""
    Xtr = np.hstack([XA_tr, XB_tr])
    Xte = np.hstack([XA_te, XB_te])
    model = GaussianNB().fit(Xtr, y_tr)
    return model.predict(Xte)
