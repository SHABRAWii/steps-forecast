# src/models/auto_q80.py
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np, time, math, os
from itertools import product
from dataclasses import dataclass

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

from joblib import Parallel, delayed, parallel_backend
try:
    from threadpoolctl import threadpool_limits
except Exception:
    # fallback no-op if not available
    class threadpool_limits:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

def _cartesian(grid: Dict[str, List[Any]]):
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    vals = [v if isinstance(v, (list, tuple)) else [v] for v in grid.values()]
    for combo in product(*vals):
        yield {k: v for k, v in zip(keys, combo)}

@dataclass
class _EvalResult:
    params: Dict[str, Any]
    cv_mae: float
    n_folds: int
    time_sec: float

class AutoQ80(BaseEstimator, RegressorMixin):
    """
    Auto-quantile HGB (q=0.80) with internal grid search using TimeSeriesSplit.

    Speed knobs (use more RAM/CPU to go faster):
      - outer_n_jobs: number of parallel workers evaluating *different param combos*
      - inner_threads: math threads per worker (OpenMP/MKL/BLAS). Set to 1–2 when outer_n_jobs>1.
      - parallelize: "combos" (default), "off" (sequential)
      - prune_if_worse: in sequential mode, stop evaluating a combo if partial mean > best-so-far
      - verbose: 0 silent, 1 combos summaries + new best, 2 add per-fold logs

    No changes needed in train.py—this stays a normal sklearn regressor.
    """

    def __init__(
        self,
        cv_splits: int = 5,
        random_state: int = 42,
        grid: Optional[Dict[str, List[Any]]] = None,
        verbose: int = 0,
        outer_n_jobs: int = 1,        # >1 => parallel across combos (more RAM & CPU)
        inner_threads: int = 0,       # 0 => leave as-is; else cap math threads per worker
        parallelize: str = "combos",  # "combos" or "off"
        prune_if_worse: bool = True,  # only applies when parallelize=="off"
    ):
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.grid = grid
        self.verbose = int(verbose)
        self.outer_n_jobs = int(outer_n_jobs)
        self.inner_threads = int(inner_threads)
        self.parallelize = str(parallelize)
        self.prune_if_worse = bool(prune_if_worse)

        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_cv_mae_: Optional[float] = None
        self.model_: Optional[HistGradientBoostingRegressor] = None

    # ---------------- internal helpers ----------------
    def _default_grid(self) -> Dict[str, List[Any]]:
        return {
            "learning_rate": [0.04, 0.06, 0.08],
            "max_iter": [400, 600, 800],
            "min_samples_leaf": [50, 100, 200],
            "max_depth": [None, 6, 8],
            "max_bins": [255],
        }

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
            early_stopping=True,           # speedup
            validation_fraction=0.1,
            n_iter_no_change=30,
        )
        params.update(overrides or {})
        return HistGradientBoostingRegressor(**params)

    def _log(self, msg: str):
        if self.verbose > 0:
            print(msg, flush=True)

    def _eval_combo_seq(self, X, y, params, best_so_far=np.inf) -> _EvalResult:
        """Evaluate one param combo sequentially (supports pruning)."""
        t0 = time.time()
        model = self._base(**params)
        cv = TimeSeriesSplit(n_splits=self.cv_splits)
        fold_mae: List[float] = []

        with threadpool_limits(self.inner_threads) if self.inner_threads > 0 else threadpool_limits():
            for j, (tr_idx, va_idx) in enumerate(cv.split(X), start=1):
                if self.verbose >= 2:
                    self._log(f"[AutoQ80]  fold {j}/{self.cv_splits} …")
                m = clone(model)
                m.fit(X[tr_idx], y[tr_idx])
                yhat = m.predict(X[va_idx])
                fold_mae.append(mean_absolute_error(y[va_idx], yhat))

                if self.prune_if_worse and len(fold_mae) >= 2:
                    partial_mean = float(np.mean(fold_mae))
                    if partial_mean > best_so_far:
                        # prune this combo (already worse than best)
                        break

        return _EvalResult(params=params, cv_mae=float(np.mean(fold_mae)), n_folds=len(fold_mae), time_sec=time.time()-t0)

    def _eval_combo_worker(self, X, y, params) -> _EvalResult:
        """Worker for parallel path; no pruning (workers don't share live best)."""
        t0 = time.time()
        model = self._base(**params)
        cv = TimeSeriesSplit(n_splits=self.cv_splits)
        fold_mae: List[float] = []

        with threadpool_limits(self.inner_threads) if self.inner_threads > 0 else threadpool_limits():
            for tr_idx, va_idx in cv.split(X):
                m = clone(model)
                m.fit(X[tr_idx], y[tr_idx])
                yhat = m.predict(X[va_idx])
                fold_mae.append(mean_absolute_error(y[va_idx], yhat))

        return _EvalResult(params=params, cv_mae=float(np.mean(fold_mae)), n_folds=len(fold_mae), time_sec=time.time()-t0)

    # ---------------- sklearn API ----------------
    def fit(self, X, y):
        # Make sure arrays are compact in memory (helps speed at the cost of RAM if copies are made)
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        y = np.asarray(y)

        grid = self.grid if self.grid is not None else self._default_grid()
        combos = list(_cartesian(grid))
        total = len(combos)

        self._log(f"[AutoQ80] starting search over {total} combos | cv={self.cv_splits} | "
                  f"parallelize={self.parallelize} | outer_n_jobs={self.outer_n_jobs} | inner_threads={self.inner_threads}")

        best_mae = math.inf
        best_params: Dict[str, Any] = {}
        best_model: Optional[HistGradientBoostingRegressor] = None
        t_all = time.time()

        if self.parallelize == "combos" and self.outer_n_jobs > 1 and total > 1:
            # Parallel evaluation across parameter combos (more RAM/CPU -> faster)
            # Use processes (loky) so each worker has its own OpenMP threads (limited by inner_threads).
            with parallel_backend("loky"):
                results: List[_EvalResult] = Parallel(n_jobs=self.outer_n_jobs, prefer="processes", verbose=0)(
                    delayed(self._eval_combo_worker)(X, y, p) for p in combos
                )
            # Summarize and pick best
            for i, res in enumerate(results, start=1):
                self._log(f"[AutoQ80] combo {i}/{total}  CV-MAE={res.cv_mae:.3f}  folds={res.n_folds}  took {res.time_sec:.1f}s  params={res.params}")
                if res.cv_mae < best_mae:
                    best_mae = res.cv_mae
                    best_params = res.params
                    self._log(f"[AutoQ80] 🚀 new best: MAE_CV={best_mae:.3f}  params={best_params}")

        else:
            # Sequential path with pruning
            for i, params in enumerate(combos, start=1):
                self._log(f"[AutoQ80] combo {i}/{total}  (prune_if_worse={self.prune_if_worse})  params={params}")
                res = self._eval_combo_seq(X, y, params, best_so_far=best_mae)
                eta = (time.time() - t_all) / i * (total - i) if i > 0 else float("nan")
                self._log(f"[AutoQ80]  → CV-MAE={res.cv_mae:.3f} (folds={res.n_folds})  took {res.time_sec:.1f}s  ETA ~{eta:.1f}s")
                if res.cv_mae < best_mae:
                    best_mae = res.cv_mae
                    best_params = params
                    self._log(f"[AutoQ80] 🚀 new best: MAE_CV={best_mae:.3f}  params={best_params}")

        # Refit on ALL data with the winner
        self._log(f"[AutoQ80] refitting best params on ALL data: {best_params}")
        best_model = self._base(**best_params)
        with threadpool_limits(self.inner_threads) if self.inner_threads > 0 else threadpool_limits():
            best_model.fit(X, y)

        self.model_ = best_model
        self.best_params_ = best_params
        self.best_cv_mae_ = float(best_mae)
        self._log(f"[AutoQ80] DONE in {time.time()-t_all:.1f}s  best_CV_MAE={self.best_cv_mae_:.3f}")
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("AutoQ80 is not fitted yet.")
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        return self.model_.predict(X)
