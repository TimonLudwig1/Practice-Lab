"""Kernbausteine fuer P02 (medium) — early vs. late Fusion, Mutual Disambiguation,
Korrelations-Falle.

Modul 18 — Multimodal Interfaces.

>>> DEINE AUFGABE <<<
Die Datengenerierung und die Klassifikatoren sind vorgegeben. Implementiere die
mit `# TODO` markierten FUSIONS-Funktionen. Der Rest (run.py, test_multimodal.py)
laeuft dann durch. Vollstaendige Loesung in loesung/.

Zwei Experimente:
  (1) KOMPLEMENTARITAET: zwei Modalitaeten, jede allein mehrdeutig (~50 %),
      zusammen fast perfekt. Zeigt early- vs. late-Fusion und Missing-Modality.
  (2) REDUNDANZ + KORRELATIONS-FALLE: zwei Modalitaeten schaetzen dasselbe;
      der Fusionsgewinn verschwindet, wenn ihr Rauschen korreliert ist.
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB


# --------------------------------------------------------------------------
# Datensatz 1 — Komplementaritaet (XOR-artig)   [vorgegeben]
# --------------------------------------------------------------------------
def make_complementary(n_per_class=600, sigma=1.0, seed=0):
    """4 Klassen y in {0,1,2,3}. Bit1 = y>>1, Bit0 = y&1.

    Modalitaet A ist NUR ueber Bit1 informativ (trennt {0,1} von {2,3}),
    Modalitaet B ist NUR ueber Bit0 informativ (trennt {0,2} von {1,3}).
    => Jede Modalitaet allein kann hoechstens 50 % erreichen. Zusammen sind
       alle 4 Klassen trennbar.

    Rueckgabe: XA (n,1), XB (n,1), y (n,).
    """
    rng = np.random.default_rng(seed)
    classes = np.repeat(np.arange(4), n_per_class)
    bit1 = classes >> 1
    bit0 = classes & 1
    muA = np.array([-2.0, 2.0])
    muB = np.array([-2.0, 2.0])
    XA = (muA[bit1] + rng.normal(0, sigma, classes.size)).reshape(-1, 1)
    XB = (muB[bit0] + rng.normal(0, sigma, classes.size)).reshape(-1, 1)
    perm = rng.permutation(classes.size)
    return XA[perm], XB[perm], classes[perm]


# --------------------------------------------------------------------------
# Datensatz 2 — Redundanz mit einstellbarer Rausch-Korrelation   [vorgegeben]
# --------------------------------------------------------------------------
def make_redundant(n=6000, sep=1.0, sigma=1.0, rho=0.0, seed=0):
    """Binaere Klassifikation. Beide Modalitaeten messen dieselbe Klasse
    c in {-sep, +sep}, jeweils plus Gauss-Rauschen mit Korrelation rho.

      x_A = c + e_A,  x_B = c + e_B,  corr(e_A, e_B) = rho.

    rho=0 -> unabhaengig -> Fusion hilft maximal; rho=1 -> Fusion bringt nichts.

    Rueckgabe: XA (n,1), XB (n,1), y (n,) mit y in {0,1}.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    c = np.where(y == 1, sep, -sep).astype(float)
    cov = sigma**2 * np.array([[1.0, rho], [rho, 1.0]])
    e = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    XA = (c + e[:, 0]).reshape(-1, 1)
    XB = (c + e[:, 1]).reshape(-1, 1)
    return XA, XB, y


# --------------------------------------------------------------------------
# Klassifikator je Modalitaet   [vorgegeben]
# --------------------------------------------------------------------------
def fit_modality(X, y):
    """Trainiert einen probabilistischen Klassifikator (GaussianNB) auf EINER
    Modalitaet."""
    return GaussianNB().fit(X, y)


def predict_from_proba(proba):
    """argmax je Zeile.   [vorgegeben]"""
    return np.asarray(proba).argmax(axis=1)


def accuracy(y_true, y_pred):
    """[vorgegeben]"""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# --------------------------------------------------------------------------
# FUSION — der Kern   >>> HIER BIST DU DRAN <<<
# --------------------------------------------------------------------------
def late_fusion_proba(proba_list, prior):
    """Late Fusion per Bayes-Produktregel unter bedingter Unabhaengigkeit:

        P(y | z_A, z_B, ...) proportional zu  prod_m P(y | z_m) / P(y)^(M-1)

    - Multipliziere die Posterior-Matrizen aus proba_list elementweise.
    - Teile den geteilten Prior heraus: er steckt M-mal drin, darf aber nur
      EINMAL zaehlen -> durch prior^(M-1) teilen (M = Zahl der Modalitaeten).
    - Normalisiere jede Zeile, sodass sie sich zu 1 summiert.

    proba_list: Liste von (n, K)-Arrays.  prior: (K,)-Vektor P(y).
    Rueckgabe:  (n, K) fusionierte, normalisierte Posterior.

    Tipp gegen numerische Probleme: mit np.maximum(..., 1e-300) gegen 0 sichern.
    """
    # TODO: implementiere die Bayes-Produktfusion
    raise NotImplementedError


def early_fusion_fit_predict(XA_tr, XB_tr, y_tr, XA_te, XB_te):
    """Early Fusion: konkateniere die Feature-Vektoren beider Modalitaeten
    (np.hstack), trainiere EIN GaussianNB darauf und gib die Vorhersagen auf
    den Testdaten zurueck.

    >>> HIER BIST DU DRAN <<<
    """
    # TODO: implementiere early fusion (hstack der Features + ein Modell)
    raise NotImplementedError
