# src/models/auto_q80.py
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

def _cartesian(grid: Dict[str, List[Any]]):
    """Yield dicts for the full cartesian product of a param grid."""
    if not grid:
        yield {}
        return
    from itertools import product
    keys = list(grid.keys())
    vals = [v if isinstance(v, (list, tuple)) else [v] for v in grid.values()]
    for combo in product(*vals):
        yield {k: v for k, v in zip(keys, combo)}

class AutoQ80(BaseEstimator, RegressorMixin):
    """
    Auto-quantile HGB (q=0.80) with internal grid search using TimeSeriesSplit.
    - Works as a normal sklearn regressor.
    - On .fit(X, y), it tries all combos in the grid, selects the one with the
      lowest mean MAE across the CV folds, then refits on **all** data.
    - No change needed in train.py; just call it via registry.

    Parameters
    ----------
    cv_splits : int
        Number of splits for TimeSeriesSplit.
    random_state : int
        Passed to the underlying HGBRegressor.
    grid : dict[str, list]
        Grid over HGB parameters (keys match HGB kwargs).
        Default covers the crucial knobs for Q80.
    verbose : int
        >0 prints per-combo MAE; 0 is quiet.
    """

    def __init__(
        self,
        cv_splits: int = 5,
        random_state: int = 42,
        grid: Optional[Dict[str, List[Any]]] = None,
        verbose: int = 0,
    ):
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.grid = grid
        self.verbose = verbose
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_cv_mae_: Optional[float] = None
        self.model_: Optional[HistGradientBoostingRegressor] = None

    def _base(self, **overrides):
        params = dict(
            loss="quantile",
            quantile=0.80,
            learning_rate=0.06,
            max_iter=600,
            max_depth=None,
            min_samples_leaf=100,
            max_bins=255,
            random_state=self.random_state,
        )
        if overrides:
            params.update(overrides)
        return HistGradientBoostingRegressor(**params)

    def _default_grid(self) -> Dict[str, List[Any]]:
        return {
            "learning_rate": [0.04, 0.06, 0.08],
            "max_iter": [400, 600, 800],
            "min_samples_leaf": [50, 100, 200],
            "max_depth": [None, 6, 8],
            "max_bins": [255],   # usually stable; keep fixed unless you need to tune
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        grid = self.grid if self.grid is not None else self._default_grid()
        cv = TimeSeriesSplit(n_splits=self.cv_splits)

        best_mae = np.inf
        best_params: Dict[str, Any] = {}
        best_model: Optional[HistGradientBoostingRegressor] = None

        for params in _cartesian(grid):
            # Build one candidate
            model = self._base(**params)

            # Cross-validated MAE (lower is better)
            fold_mae = []
            for tr_idx, va_idx in cv.split(X):
                Xtr, Xva = X[tr_idx], X[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]
                m = clone(model)
                m.fit(Xtr, ytr)
                yhat = m.predict(Xva)
                fold_mae.append(mean_absolute_error(yva, yhat))

            cv_mae = float(np.mean(fold_mae))
            if self.verbose:
                #print how many models have been tried so far and how many in total
                print(f"[AutoQ80] Tried {len(fold_mae)} of {self.cv_splits} folds. ", end="")
                print(f"[AutoQ80] params={params}  CV-MAE={cv_mae:.3f}")
                

            if cv_mae < best_mae:
                best_mae = cv_mae
                best_params = params
                best_model = clone(model)
                if self.verbose:
                    print(f"  New best! params={best_params}  CV-MAE={best_mae:.3f}")

        if best_model is None:
            # Fallback: use defaults
            best_model = self._base()
            best_params = {}
            best_mae = np.nan

        # Refit on ALL data using the best params
        best_model.fit(X, y)

        self.model_ = best_model
        self.best_params_ = best_params
        self.best_cv_mae_ = best_mae
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("AutoQ80 is not fitted yet.")
        X = np.asarray(X, dtype=np.float32)
        return self.model_.predict(X)
