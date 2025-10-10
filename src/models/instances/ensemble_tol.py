# src/models/instances/ensemble_tol.py
from __future__ import annotations
from typing import List, Optional, Callable
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import TimeSeriesSplit

# We import make_model at module scope so joblib can pickle this class cleanly.
from src.models.registry import make_model


def _tol_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int = 50) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_pred) <= k))


class EnsembleTolK(BaseEstimator, RegressorMixin):
    """
    Fit several base models, then choose blend weights that maximize Tol@k
    on an inner CV split. Works as a normal sklearn regressor, so train.py
    stays unchanged.

    Parameters
    ----------
    base_names : list[str]
        Model names to fetch from registry.make_model (e.g., ["lgbm_poisson_log1p", "cat_mae"])
    k : int
        Tolerance threshold for Tol@k.
    cv_splits : int
        TimeSeriesSplit folds for the inner validation used to pick weights.
    alphas : list[float] | None
        If exactly 2 bases, we grid-search alpha in [0..1] over these values
        (α for model0; 1-α for model1). If more than 2 bases, we default to
        uniform weights.
    random_state : int
        Passed to registry.make_model for reproducibility.
    verbose : int
        0 = silent, 1 = prints selection summary.
    maker : Callable | None
        Optional alternate factory; defaults to src.models.registry.make_model.
    """

    def __init__(
        self,
        base_names: List[str],
        k: int = 50,
        cv_splits: int = 3,
        alphas: Optional[List[float]] = None,
        random_state: int = 42,
        verbose: int = 1,
        maker: Optional[Callable[..., object]] = None,
    ):
        self.base_names = base_names
        self.k = int(k)
        self.cv_splits = int(cv_splits)
        self.alphas = alphas if alphas is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
        self.random_state = random_state
        self.verbose = verbose
        self.maker = maker  # if None, we use registry.make_model

        # fitted members
        self.bases_ = None
        self.weights_ = None

    # --- helpers ---
    def _log(self, *args):
        if self.verbose:
            print("[EnsembleTolK]", *args, flush=True)

    # --- sklearn API ---
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # 1) build base models from registry
        factory = self.maker if self.maker is not None else make_model
        self.bases_ = [clone(factory(nm, random_state=self.random_state)) for nm in self.base_names]

        # 2) inner split: use last fold as validation to pick weights
        cv = TimeSeriesSplit(n_splits=self.cv_splits)
        tr_idx, va_idx = list(cv.split(X))[-1]
        Xtr, Xva, ytr, yva = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]

        # fit bases on inner-train
        for b in self.bases_:
            b.fit(Xtr, ytr)

        # collect validation predictions
        preds_val = np.vstack([b.predict(Xva) for b in self.bases_])  # shape [n_bases, n_val]

        # 3) search weights
        if len(self.bases_) == 2:
            best_acc, best_w = -1.0, None
            for a in self.alphas:
                w = np.array([a, 1.0 - a], dtype=np.float32)
                yhat = (w[0] * preds_val[0] + w[1] * preds_val[1])
                acc = _tol_accuracy(yva, yhat, k=self.k)
                if acc > best_acc:
                    best_acc, best_w = acc, w
            self.weights_ = best_w
            self._log(f"selected α={self.weights_[0]:.2f}/{self.weights_[1]:.2f}  Tol@{self.k}={best_acc:.3f}")
        else:
            # fallback: uniform weights
            nb = preds_val.shape[0]
            self.weights_ = np.ones(nb, dtype=np.float32) / nb
            acc = _tol_accuracy(yva, (self.weights_.reshape(-1, 1) * preds_val).sum(0), k=self.k)
            self._log(f"selected uniform weights over {nb} bases  Tol@{self.k}={acc:.3f}")

        # 4) refit bases on ALL data
        for i, name in enumerate(self.base_names):
            self.bases_[i] = clone(factory(name, random_state=self.random_state))
            self.bases_[i].fit(X, y)

        return self

    def predict(self, X):
        if self.bases_ is None or self.weights_ is None:
            raise RuntimeError("EnsembleTolK is not fitted yet.")
        X = np.asarray(X, dtype=np.float32)
        P = np.vstack([b.predict(X) for b in self.bases_])
        return (self.weights_.reshape(-1, 1) * P).sum(0)
