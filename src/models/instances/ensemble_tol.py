from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import TimeSeriesSplit

def _tol_accuracy(y, yhat, k=50):
    return float(np.mean(np.abs(y - yhat) <= k))

class EnsembleTolK(BaseEstimator, RegressorMixin):
    """
    Trains several base models and picks blend weights to maximize Tol@k on an inner CV split.
    Works with train.py unchanged.
    """
    def __init__(self, base_names: List[str], k: int = 50, cv_splits: int = 3, alphas: Optional[List[float]] = None,
                 random_state: int = 42, registry_make=None, verbose: int = 1):
        self.base_names = base_names
        self.k = int(k)
        self.cv_splits = int(cv_splits)
        self.alphas = alphas if alphas is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
        self.random_state = random_state
        self.registry_make = registry_make
        self.verbose = verbose
        self.bases_ = None
        self.weights_ = None

    def _log(self, *a):
        if self.verbose:
            print("[EnsembleTolK]", *a, flush=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        # 1) build bases
        from src.models.registry import make_model as _make if self.registry_make is None else self.registry_make
        self.bases_ = [clone(_make(nm, random_state=self.random_state)) for nm in self.base_names]

        # 2) inner split: use last fold as validation to select weights
        cv = TimeSeriesSplit(n_splits=self.cv_splits)
        tr_idx, va_idx = list(cv.split(X))[-1]  # last fold
        Xtr, Xva, ytr, yva = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]

        # fit bases on inner-train
        for b in self.bases_:
            b.fit(Xtr, ytr)

        # collect val preds
        P = []
        for b in self.bases_:
            P.append(b.predict(Xva))
        P = np.vstack(P)  # shape: [n_bases, n_val]

        # simple 2-model blend search if exactly 2 bases; else uniform grid over simplex
        best_w, best_acc = None, -1.0
        if len(self.bases_) == 2:
            for a in self.alphas:
                w = np.array([a, 1.0 - a], dtype=np.float32)
                yhat = (w[0] * P[0] + w[1] * P[1])
                acc = _tol_accuracy(yva, yhat, k=self.k)
                if acc > best_acc:
                    best_acc, best_w = acc, w
        else:
            # naive uniform weights as baseline
            w = np.ones(len(self.bases_), dtype=np.float32) / len(self.bases_)
            yhat = (w.reshape(-1,1) * P).sum(0)
            best_acc, best_w = _tol_accuracy(yva, yhat, k=self.k), w

        self.weights_ = best_w
        self._log(f"selected weights={self.weights_}  Tol@{self.k}={best_acc:.3f}")

        # 3) refit bases on ALL data
        for i, name in enumerate(self.base_names):
            self.bases_[i] = clone(self.bases_[i])
            self.bases_[i].fit(X, y)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        preds = [b.predict(X) for b in self.bases_]
        P = np.vstack(preds)
        return (self.weights_.reshape(-1,1) * P).sum(0)
