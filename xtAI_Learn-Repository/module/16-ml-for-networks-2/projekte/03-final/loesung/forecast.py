"""Forecasting-Modelle und Metriken — bewusst schlicht gehalten.

Die Reihenfolge ist Absicht (Skript 3.2): erst die Baseline, die man schlagen MUSS, dann
zeitliche Features, dann erst der Graph. Wer mit dem GNN anfaengt, weiss nie, ob es etwas
beitraegt.

`statsmodels` (ARIMA/SARIMA) ist in dieser Umgebung nicht installiert; das Skript (3.3)
erklaert SARIMA formal. Praktisch sind Lag-Features + Ridge hier ohnehin flexibler.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

STUNDE, TAG, WOCHE = 1, 24, 24 * 7


def saisonal_naiv(Y: np.ndarray, test_idx, m: int = WOCHE) -> np.ndarray:
    """y_hat[t] = y[t-m].  'Naechsten Dienstag 20 Uhr wie letzten Dienstag 20 Uhr.'

    Die Messlatte. Bei Netzverkehr erschreckend stark.
    """
    return Y[[k - m for k in test_idx]]


def mase(y_wahr: np.ndarray, y_pred: np.ndarray, y_naiv: np.ndarray) -> float:
    """MAE relativ zur saisonal-naiven Baseline.  < 1 = besser als naiv, >= 1 = wertlos.

    Skalenfrei -> ueber verschieden grosse Knoten vergleichbar (Skript 3.2).
    """
    return mean_absolute_error(y_wahr, y_pred) / mean_absolute_error(y_wahr, y_naiv)


def baue_features(Y: np.ndarray, A_norm: np.ndarray, zeit_idx, mit_graph: bool):
    """Zeitreihe -> Tabelle.  Rueckgabe: (X, y), gestapelt ueber alle Knoten.

    Features je (Zeitpunkt, Knoten):
      - Lags des Knotens selbst: t-1, t-2, t-3, t-24 (gestern), t-168 (letzte Woche)
      - Kalender ZYKLISCH kodiert: sin/cos von Stunde und Wochentag.
        (Roh waeren 23 Uhr und 0 Uhr maximal weit entfernt, obwohl benachbart! Skript 3.4)
      - nur bei mit_graph: Nachbar-Mittel zu t-1 und t-2 sowie das 2-Hop-Mittel zu t-1.
        Das ist die 'raeumliche' Information - genau das, was ein GNN aggregieren wuerde.
    """
    n = Y.shape[1]
    X_list, y_list = [], []
    for k in zeit_idx:
        lags = np.stack([Y[k - 1], Y[k - 2], Y[k - 3], Y[k - TAG], Y[k - WOCHE]], axis=1)
        h, d = k % 24, (k // 24) % 7
        kalender = np.tile([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)], (n, 1))
        teile = [lags, kalender]
        if mit_graph:
            nachbar1 = A_norm @ Y[k - 1]
            nachbar2 = A_norm @ Y[k - 2]
            zwei_hop = A_norm @ nachbar1
            teile += [nachbar1[:, None], nachbar2[:, None], zwei_hop[:, None]]
        X_list.append(np.hstack(teile))
        y_list.append(Y[k])
    return np.vstack(X_list), np.concatenate(y_list)


def zeit_split(T: int, test_stunden: int = WOCHE):
    """ZEITBASIERTER Split: die letzte Woche ist Test.

    Ein zufaelliger Split waere hier fatal - man wuerde auf der Zukunft trainieren und auf der
    Vergangenheit testen (Skript 3.4 / Modul 15, 3.5).
    """
    split = T - test_stunden
    train_idx = range(WOCHE + 1, split)      # erst ab t>WOCHE gibt es alle Lags
    test_idx = range(split, T)
    return train_idx, test_idx


def trainiere_ridge(Y, A_norm, train_idx, test_idx, mit_graph: bool, alpha: float = 1.0):
    """Ridge auf Lag-Features. Rueckgabe: (y_wahr, y_pred)."""
    X_tr, y_tr = baue_features(Y, A_norm, train_idx, mit_graph)
    X_te, y_te = baue_features(Y, A_norm, test_idx, mit_graph)
    modell = Ridge(alpha=alpha).fit(X_tr, y_tr)
    return y_te, modell.predict(X_te)
