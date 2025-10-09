from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


def make_model(name: str, random_state=42, n_jobs=-1):
    name = name.lower()
    if name == "naive_last":
        # handled specially in train loop (predict X["steps_lag1"])
        return "NAIVE"
    if name == "linreg":
        return Pipeline([("scaler", StandardScaler(with_mean=False)), ("lr", LinearRegression(n_jobs=n_jobs))])
    if name == "ridge":
        return Ridge(
            alpha=1.0,
            solver="sag",      # or "saga"
            max_iter=2000,
            tol=1e-3,
            # fit_intercept=True (default)
        )
        # return Pipeline([("scaler", StandardScaler(with_mean=False)), ("ridge", RidgeCV(alphas=[0.1,1,10,100]))])
    if name == "rf100":
        return RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=n_jobs)
    if name == "ridge_sgd":
        return SGDRegressor(loss="squared_error", penalty="l2",
                            alpha=1e-4, learning_rate="optimal",
                            max_iter=20, tol=1e-3,
                            random_state=random_state,
                            n_iter_no_change=5, early_stopping=True)
    if name == "rf200":
        return RandomForestRegressor(n_estimators=200, n_jobs=n_jobs, random_state=random_state)

    if name == "et200":
        return ExtraTreesRegressor(n_estimators=200, n_jobs=n_jobs, random_state=random_state)

    # HistGradientBoosting — squared error
    if name == "hgb_l2":
        return HistGradientBoostingRegressor(loss="squared_error",
                                             max_depth=None,
                                             max_bins=255,
                                             learning_rate=0.06,
                                             max_iter=300,
                                             random_state=random_state)

    # HistGradientBoosting — absolute error (MAE)
    if name == "hgb_mae":
        return HistGradientBoostingRegressor(loss="absolute_error",
                                             max_depth=None,
                                             max_bins=255,
                                             learning_rate=0.06,
                                             max_iter=400,
                                             random_state=random_state)

    # HistGradientBoosting — Poisson (good for counts >=0). Wrap in log1p/expm1 to stabilize.
    if name == "hgb_poisson_log1p":
        base = HistGradientBoostingRegressor(loss="poisson",
                                             max_depth=None,
                                             max_bins=255,
                                             learning_rate=0.06,
                                             max_iter=400,
                                             random_state=random_state)
        return TransformedTargetRegressor(
            regressor=base,
            func=lambda y: np.log1p(np.clip(y, 0, None)),
            inverse_func=lambda y: np.expm1(y)
        )

    # Robust linear baseline (Huber)
    if name == "huber":
        return HuberRegressor(epsilon=1.35, alpha=1e-4)
    
    # ---- Blended HGB-MAE (base + peak-weighted) -----------------------------
    if name == "hgb_mae_blend":
        return HGBMAEBlendPeak(
            alpha=0.8,           # you can adjust later
            low=80, high=400,    # mid-peak band to emphasize
            max_w=1.5,           # gentle up-weight
            learning_rate=0.06,
            max_iter=400,
            max_depth=None,
            min_samples_leaf=100,
            max_bins=255,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model: {name}")

def _peak_weights(y, low=80, high=400, max_w=1.5):
    """
    Gentle, selective weights:
    - weight=1 below `low`
    - ramps linearly up to `max_w` at `high`
    - weight=1 again above `high` (don’t chase extreme outliers)
    """
    import numpy as np
    y = np.asarray(y, dtype="float32")
    w = np.ones_like(y, dtype="float32")
    mask = (y >= low) & (y <= high)
    w[mask] = 1.0 + (max_w - 1.0) * ((y[mask] - low) / (high - low))
    return w


class HGBMAEBlendPeak(BaseEstimator, RegressorMixin):
    """
    Two HGB-MAE models:
      - base: unweighted
      - peak: trained with selective sample weights
    Predicts alpha * base + (1 - alpha) * peak.
    """

    def __init__(
        self,
        alpha=0.8,                  # blend weight towards base (0.5..0.9 is typical)
        # weighting params
        low=80, high=400, max_w=1.5,
        # HGB params (kept modest; tweak if desired)
        learning_rate=0.06,
        max_iter=400,
        max_depth=None,
        min_samples_leaf=100,
        max_bins=255,
        random_state=42,
    ):
        self.alpha = alpha
        self.low = low
        self.high = high
        self.max_w = max_w
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.random_state = random_state

        self.base_ = None
        self.peak_ = None

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingRegressor
        import numpy as np

        self.base_ = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_bins=self.max_bins,
            random_state=self.random_state,
        )
        self.base_.fit(X, y)

        self.peak_ = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_bins=self.max_bins,
            random_state=self.random_state,
        )
        w = _peak_weights(y, low=self.low, high=self.high, max_w=self.max_w)
        self.peak_.fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        import numpy as np
        yb = self.base_.predict(X)
        yp = self.peak_.predict(X)
        return self.alpha * yb + (1.0 - self.alpha) * yp
