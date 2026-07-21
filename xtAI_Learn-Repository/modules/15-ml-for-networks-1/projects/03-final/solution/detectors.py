"""Detektoren: ueberwacht (Misuse) vs. semi-ueberwacht (Anomalie).

Der entscheidende Unterschied ist NICHT der Algorithmus, sondern WAS SIE ZU SEHEN BEKOMMEN:
  - SupervisedDetektor : Normalverkehr + BEKANNTE Angriffe (mit Labels)  -> modelliert das Boese
  - *AnomalieDetektor  : NUR Normalverkehr, voellig ohne Angriffslabels  -> modelliert das Normale

Alle bieten dieselbe Schnittstelle:  fit(...) / score(X) / predict(X) -> 1 = Alarm, 0 = harmlos.

WICHTIGE DESIGN-ENTSCHEIDUNG - alle Anomaliedetektoren teilen sich EIN FPR-Budget:
Statt jedem Verfahren seine eigene, unvergleichbare Schwellenlogik zu lassen (`contamination`,
`nu`, `offset_` ...), berechnen wir fuer jeden Detektor einen Anomalie-Score (hoeher = anomaler)
und setzen die Schwelle auf das (100 - ziel_fpr)-Perzentil der Scores AUF DEM TRAININGS-
NORMALVERKEHR. Damit hat jeder Detektor per Konstruktion ~ziel_fpr Prozent Fehlalarme, und der
Zero-Day-Recall wird zur einzigen freien Groesse -> ein FAIRER Vergleich bei gleichem Preis.

(Nebenbei umgeht das zwei echte Fallstricke, ueber die ich beim Bauen gestolpert bin:
 - LocalOutlierFactor.predict() ist hier unbrauchbar: sein `offset_` entgleist auf diesen Daten
   auf ~-1.3e6, sodass NIE Alarm ausgeloest wird - obwohl die Scores sauber trennen.
 - SGDOneClassSVM ist LINEAR und trennt hier gar nichts (Score normal 3.946 vs. smurf 3.959).
   Erst mit einer Kernel-Approximation (Nystroem) funktioniert es.)
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance

ZIEL_FPR_PROZENT = 1.0      # gemeinsames Fehlalarm-Budget aller Anomaliedetektoren


class SupervisedDetektor:
    """Klassischer Misuse-Detektor: lernt, die BEKANNTEN Angriffe zu erkennen."""
    name = "Supervised RF (kennt bekannte Angriffe)"

    def __init__(self, random_state=0):
        self.modell = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                             random_state=random_state)

    def fit(self, X, y):
        self.modell.fit(X, y)
        return self

    def score(self, X):
        return self.modell.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.modell.predict(X).astype(int)


class _AnomalieBasis:
    """Geruest: skalieren, NUR auf Normalverkehr fitten, Schwelle aufs FPR-Budget kalibrieren."""
    name = "Anomalie"
    dedupliziere = False        # von LOF ueberschrieben - siehe LOFDetektor

    def __init__(self, ziel_fpr_prozent=ZIEL_FPR_PROZENT):
        self.ziel_fpr_prozent = ziel_fpr_prozent

    def fit(self, X_normal):
        if self.dedupliziere:
            X_normal = X_normal.drop_duplicates()
        self.scaler = StandardScaler().fit(X_normal)
        Xs = self.scaler.transform(X_normal)
        self._fit_kern(Xs)
        # Schwelle so, dass auf dem Trainings-Normalverkehr ~ziel_fpr_prozent Alarme entstehen
        train_scores = self._roh_score(Xs)
        self.schwelle_ = float(np.percentile(train_scores, 100.0 - self.ziel_fpr_prozent))
        return self

    def score(self, X):
        return self._roh_score(self.scaler.transform(X))

    def predict(self, X):
        return (self.score(X) > self.schwelle_).astype(int)


class IsolationForestDetektor(_AnomalieBasis):
    """Isoliert Punkte durch zufaellige Splits: Anomalien brauchen weniger Splits."""
    name = "Isolation Forest"

    def __init__(self, random_state=0, **kw):
        super().__init__(**kw)
        self.random_state = random_state

    def _fit_kern(self, Xs):
        self.modell = IsolationForest(n_estimators=200, random_state=self.random_state).fit(Xs)

    def _roh_score(self, Xs):
        return -self.modell.score_samples(Xs)      # hoeher = anomaler


class MahalanobisDetektor(_AnomalieBasis):
    """Gaussmodell des Normalverkehrs: Alarm bei grosser Mahalanobis-Distanz.

    Das simpelste denkbare Verfahren - und eine faire Messlatte: schlaegt ein komplexer
    Detektor das hier nicht, ist er seinen Aufwand nicht wert.
    """
    name = "Mahalanobis-Gauss"

    def _fit_kern(self, Xs):
        # winziges Rauschen gegen singulaere Kovarianz (konstante/duplizierte Spalten)
        Xs = Xs + 1e-9 * np.random.default_rng(0).standard_normal(Xs.shape)
        self.cov = EmpiricalCovariance().fit(Xs)

    def _roh_score(self, Xs):
        return self.cov.mahalanobis(Xs)


class OneClassSVMDetektor(_AnomalieBasis):
    """One-Class SVM mit Nystroem-Kernel-Approximation.

    Die exakte OneClassSVM ist bei ~70k Punkten zu langsam (O(n^2)); die reine SGD-Variante
    ist linear und trennt hier NICHTS. Nystroem approximiert den RBF-Kernel und macht die
    schnelle SGD-Variante brauchbar.
    """
    name = "One-Class SVM (Nystroem)"

    def __init__(self, nu=0.01, gamma=0.1, n_components=200, random_state=0, **kw):
        super().__init__(**kw)
        self.nu, self.gamma = nu, gamma
        self.n_components, self.random_state = n_components, random_state

    def _fit_kern(self, Xs):
        self.modell = make_pipeline(
            Nystroem(gamma=self.gamma, n_components=self.n_components,
                     random_state=self.random_state),
            SGDOneClassSVM(nu=self.nu, random_state=self.random_state),
        ).fit(Xs)

    def _roh_score(self, Xs):
        return -self.modell.decision_function(Xs)


class LOFDetektor(_AnomalieBasis):
    """Local Outlier Factor: vergleicht die lokale Dichte eines Punktes mit der seiner Nachbarn.

    ZWEI Fallstricke, die diesen Detektor hier zuerst voellig unbrauchbar gemacht haben:

    1. **predict() nicht benutzen.** Sein `offset_` entgleist auf diesen Daten auf ~-1.3e6,
       sodass NIE Alarm ausgeloest wird. Wir nutzen score_samples() + eigene Schwelle.

    2. **Duplikate zerstoeren LOF** (deshalb `dedupliziere = True`). KDD99 ist massiv redundant
       (Skript 3.6): 13.4 % des Normalverkehrs sind EXAKTE Duplikate. Fuer LOF ist das fatal -
       sind die k Nachbarn eines Punktes mit ihm identisch, ist deren Abstand 0, die lokale
       Dichte geht gegen unendlich und der LOF-Quotient explodiert. Gemessen:
           mit Duplikaten : Trainings-Scores bis 1.55e9  -> 99%-Schwelle 1.09e6 -> Recall 0.000
           dedupliziert   : Trainings-Scores bis 393     -> 99%-Schwelle 1.98   -> Recall 1.000
       Die Redundanz des Datensatzes blaeht also nicht nur Metriken auf, sie legt ganze
       Verfahrensklassen lahm.
    """
    name = "Local Outlier Factor"
    dedupliziere = True

    def __init__(self, n_neighbors=20, **kw):
        super().__init__(**kw)
        self.n_neighbors = n_neighbors

    def _fit_kern(self, Xs):
        self.modell = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True,
                                         n_jobs=-1).fit(Xs)

    def _roh_score(self, Xs):
        return -self.modell.score_samples(Xs)
