"""Forecasting models and metrics — deliberately kept plain.

The order is intentional (script 3.2): first the baseline that you MUST beat, then temporal
features, and only then the graph. Whoever starts with the GNN never knows whether it contributes
anything.

`statsmodels` (ARIMA/SARIMA) is not installed in this environment; the script (3.3) explains
SARIMA formally. In practice lag features + ridge are more flexible here anyway.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

HOUR, DAY, WEEK = 1, 24, 24 * 7


def seasonal_naive(Y: np.ndarray, test_idx, m: int = WEEK) -> np.ndarray:
    """y_hat[t] = y[t-m].  'Next Tuesday 8 p.m. like last Tuesday 8 p.m.'

    The yardstick. For network traffic frighteningly strong.
    """
    return Y[[k - m for k in test_idx]]


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> float:
    """MAE relative to the seasonal-naive baseline.  < 1 = better than naive, >= 1 = worthless.

    Scale-free -> comparable across nodes of different size (script 3.2).
    """
    return mean_absolute_error(y_true, y_pred) / mean_absolute_error(y_true, y_naive)


def build_features(Y: np.ndarray, A_norm: np.ndarray, time_idx, with_graph: bool):
    """Time series -> table.  Returns: (X, y), stacked over all nodes.

    Features per (time point, node):
      - lags of the node itself: t-1, t-2, t-3, t-24 (yesterday), t-168 (last week)
      - the calendar CYCLICALLY encoded: sin/cos of the hour and the weekday.
        (Raw, 11 p.m. and midnight would be maximally far apart although they are adjacent!
        Script 3.4)
      - only if with_graph: the neighbour mean at t-1 and t-2 as well as the 2-hop mean at t-1.
        That is the 'spatial' information - exactly what a GNN would aggregate.
    """
    n = Y.shape[1]
    X_list, y_list = [], []
    for k in time_idx:
        lags = np.stack([Y[k - 1], Y[k - 2], Y[k - 3], Y[k - DAY], Y[k - WEEK]], axis=1)
        h, d = k % 24, (k // 24) % 7
        calendar = np.tile([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)], (n, 1))
        parts = [lags, calendar]
        if with_graph:
            neighbor_1 = A_norm @ Y[k - 1]
            neighbor_2 = A_norm @ Y[k - 2]
            two_hop = A_norm @ neighbor_1
            parts += [neighbor_1[:, None], neighbor_2[:, None], two_hop[:, None]]
        X_list.append(np.hstack(parts))
        y_list.append(Y[k])
    return np.vstack(X_list), np.concatenate(y_list)


def time_split(T: int, test_hours: int = WEEK):
    """A TIME-BASED split: the last week is the test set.

    A random split would be fatal here - you would train on the future and test on the past
    (script 3.4 / module 15, 3.5).
    """
    split = T - test_hours
    train_idx = range(WEEK + 1, split)      # only from t>WEEK on do all lags exist
    test_idx = range(split, T)
    return train_idx, test_idx


def train_ridge(Y, A_norm, train_idx, test_idx, with_graph: bool, alpha: float = 1.0):
    """Ridge on lag features. Returns: (y_true, y_pred)."""
    X_tr, y_tr = build_features(Y, A_norm, train_idx, with_graph)
    X_te, y_te = build_features(Y, A_norm, test_idx, with_graph)
    model = Ridge(alpha=alpha).fit(X_tr, y_tr)
    return y_te, model.predict(X_te)
