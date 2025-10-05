import numpy as np

def mae(y, yhat):  return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat)**2)))
def mape(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    denom = np.clip(np.abs(y), 1e-6, None)
    return float(np.mean(np.abs((y - yhat) / denom) * 100))
def r2(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return float(1 - ss_res/ss_tot) if ss_tot > 0 else float("nan")
def abs_error(y, yhat):
    return np.abs(y - yhat)

def perc_abs_error(y, yhat, q=90):
    """Percentile (q) of absolute error, e.g., q=90 for P90."""
    return float(np.percentile(abs_error(y, yhat), q))

def tol_accuracy(y, yhat, k=100):
    """Tolerance accuracy: share of points with |err| <= k steps."""
    return float((abs_error(y, yhat) <= k).mean())

def directional_accuracy(y, yhat, ref_prev):
    """
    Directional accuracy vs last step:
    compares sign(y - ref_prev) to sign(yhat - ref_prev).
    If ref_prev is missing, returns np.nan.
    """
    if ref_prev is None:
        return np.nan
    ref_prev = np.asarray(ref_prev)
    mask = ~np.isnan(ref_prev)
    if mask.sum() == 0:
        return np.nan
    true_dir = np.sign(y[mask] - ref_prev[mask])
    pred_dir = np.sign(yhat[mask] - ref_prev[mask])
    return float((true_dir == pred_dir).mean())

def wape(y, yhat):
    y = np.asarray(y)
    denom = np.abs(y).sum()
    if denom == 0:
        return np.nan
    return float(np.abs(y - yhat).sum() / denom)

def smape(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    denom = np.abs(y) + np.abs(yhat)
    denom = np.where(denom == 0, 1.0, denom)  # avoid /0 when both zero
    return float(np.mean(2.0 * np.abs(y - yhat) / denom))