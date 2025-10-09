from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from src.models.instances.autoQ80 import AutoQ80


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
            alpha=0.5,      # try 0.7–0.9
            k=200.0,        # try 150–300
            power=1.0,      # try 0.7–1.3
            cap=3.0,        # start gentler than 5.0 if tol-acc drops
            learning_rate=0.06,
            max_iter=400,
            max_depth=None,
            min_samples_leaf=100,
            max_bins=255,
            random_state=random_state,
        )
    if name == "two_stage_hgb":
        return TwoStageHGB(
            thr=50,
            pos_class_weight=3.0,
            reg_lr=0.06, reg_max_iter=400,
            reg_min_samples_leaf=100,
            random_state=random_state,
        )
    if name == "hgb_q80":
        return HGBQuantile(quantile=0.8, random_state=random_state)

    if name == "hgb_mae_qblend":
        return HGBMaePlusQBlend(alpha=0.75, q=0.8, random_state=random_state)
    if name == "hgb_q80_auto":
        return AutoQ80(cv_splits=5, random_state=random_state, grid=None, verbose=1)

    raise ValueError(f"Unknown model: {name}")

def _peak_weights_pow(y, k=200.0, power=1.0, cap=5.0):
    """
    Your weighting: w = 1 + min((|y|/k)^power, cap)
    - Up-weights larger targets smoothly.
    - `k` is the scale (steps) where weights start to grow.
    - `cap` limits how large the weight can get.
    """
    y = np.asarray(y, dtype="float32")
    w = 1.0 + np.minimum((np.abs(y) / float(k)) ** float(power), float(cap))
    return w.astype("float32")
class HGBMAEBlendPeak(BaseEstimator, RegressorMixin):
    """
    Blend of two HGB-MAE models:
      - base: unweighted MAE model
      - peak: same model, trained with your power-law sample weights
    Prediction: ŷ = alpha * ŷ_base + (1 - alpha) * ŷ_peak
    """

    def __init__(
        self,
        alpha=0.8,                 # blend weight toward base (0.5..0.9)
        k=200.0, power=1.0, cap=5.0,   # <-- your weighting params
        learning_rate=0.06,
        max_iter=400,
        max_depth=None,
        min_samples_leaf=100,
        max_bins=255,
        random_state=42,
    ):
        self.alpha = alpha
        self.k = k
        self.power = power
        self.cap = cap
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
        w = _peak_weights_pow(y, k=self.k, power=self.power, cap=self.cap)
        self.peak_.fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        yb = self.base_.predict(X)
        yp = self.peak_.predict(X)
        return self.alpha * yb + (1.0 - self.alpha) * yp
class TwoStageHGB(BaseEstimator, RegressorMixin):
    """
    Stage A: classify active (steps >= thr) with HGB-Classifier
    Stage B: regress steps on active rows with HGB-Regressor (MAE)
    Prediction: y_hat = P(active) * y_hat_reg
    """
    def __init__(
        self,
        thr=50,                    # activity threshold in steps for current slot
        clf_lr=0.1, clf_max_depth=6, clf_min_samples_leaf=200, clf_max_bins=255,
        reg_lr=0.06, reg_max_depth=None, reg_min_samples_leaf=100, reg_max_bins=255, reg_max_iter=400,
        pos_class_weight=3.0,      # upweight active class (usually rare)
        random_state=42,
    ):
        self.thr = thr
        self.clf_lr = clf_lr
        self.clf_max_depth = clf_max_depth
        self.clf_min_samples_leaf = clf_min_samples_leaf
        self.clf_max_bins = clf_max_bins
        self.reg_lr = reg_lr
        self.reg_max_depth = reg_max_depth
        self.reg_min_samples_leaf = reg_min_samples_leaf
        self.reg_max_bins = reg_max_bins
        self.reg_max_iter = reg_max_iter
        self.pos_class_weight = pos_class_weight
        self.random_state = random_state
        self.clf_ = None
        self.reg_ = None

    def fit(self, X, y):
        y = np.asarray(y)
        active = (y >= self.thr).astype(np.int32)

        # ---- Classifier (active vs inactive) ----
        # class_weight via sample weights
        w = np.ones_like(y, dtype="float32")
        w[active == 1] = self.pos_class_weight

        self.clf_ = HistGradientBoostingClassifier(
            learning_rate=self.clf_lr,
            max_depth=self.clf_max_depth,
            min_samples_leaf=self.clf_min_samples_leaf,
            max_bins=self.clf_max_bins,
            random_state=self.random_state,
        )
        self.clf_.fit(X, active, sample_weight=w)

        # ---- Regressor (only on active rows) ----
        self.reg_ = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=self.reg_lr,
            max_depth=self.reg_max_depth,
            min_samples_leaf=self.reg_min_samples_leaf,
            max_bins=self.reg_max_bins,
            max_iter=self.reg_max_iter,
            random_state=self.random_state,
        )
        m = active.astype(bool)
        if m.sum() == 0:
            # fallback: train on all rows to avoid crash
            self.reg_.fit(X, y)
        else:
            self.reg_.fit(X[m], y[m])
        return self

    def predict(self, X):
        p_active = self.clf_.predict_proba(X)[:, 1]    # [0,1]
        y_reg = self.reg_.predict(X)                   # ≥0 but model may output small negatives
        y_pred = p_active * y_reg
        # clamp tiny negatives (rare numerical artifact)
        return np.maximum(y_pred, 0.0)
class HGBQuantile(BaseEstimator, RegressorMixin):
    def __init__(self, quantile=0.8, learning_rate=0.06, max_iter=600,
                 max_depth=None, min_samples_leaf=100, max_bins=255, random_state=42):
        self.quantile = quantile
        self.params = dict(loss="quantile", quantile=quantile,
                           learning_rate=learning_rate, max_iter=max_iter,
                           max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                           max_bins=max_bins, random_state=random_state)
        self.model_ = None

    def fit(self, X, y):
        self.model_ = HistGradientBoostingRegressor(**self.params)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

class HGBMaePlusQBlend(BaseEstimator, RegressorMixin):
    """ŷ = a*ŷ_MAE + (1-a)*ŷ_Q, with a chosen later (config)"""
    def __init__(self, alpha=0.75, q=0.8, random_state=42):
        self.alpha = alpha
        self.q = q
        self.random_state = random_state
        self.base_ = HistGradientBoostingRegressor(loss="absolute_error",
                                                   learning_rate=0.06, max_iter=400,
                                                   max_depth=None, min_samples_leaf=100,
                                                   max_bins=255, random_state=random_state)
        self.qr_   = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                                   learning_rate=0.06, max_iter=600,
                                                   max_depth=None, min_samples_leaf=100,
                                                   max_bins=255, random_state=random_state)
    def fit(self, X, y):
        self.base_.fit(X, y)
        self.qr_.fit(X, y)
        return self
    def predict(self, X):
        yb = self.base_.predict(X)
        yq = self.qr_.predict(X)
        return self.alpha*yb + (1.0-self.alpha)*yq