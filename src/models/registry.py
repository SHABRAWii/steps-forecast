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
    raise ValueError(f"Unknown model: {name}")
