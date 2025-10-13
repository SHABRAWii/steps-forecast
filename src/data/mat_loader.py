import numpy as np, pandas as pd
from scipy.io import loadmat

def datenum_to_dt(x):
    from datetime import datetime, timedelta
    x = float(x)
    return datetime.fromordinal(int(x)) + timedelta(days=x % 1) - timedelta(days=366)

def load_tidy_from_compat(mat_path: str, varname="DataComplete", min_days=50) -> pd.DataFrame:
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    dc = mat[varname]
    if hasattr(dc, "ravel"): dc = dc.ravel().tolist()

    rows = []
    for i, cell in enumerate(dc):
        uid = cell.get("UserID", None)
        steps = np.asarray(cell.get("Steps", []), float)
        hr    = np.asarray(cell.get("HR", []), float)
        date_dnum  = np.asarray(cell.get("Date_dnum", []), float)   # shape (C,)
        dt_dnum    = np.asarray(cell.get("DateTime_dnum", []), float)  # shape (R,C)

        if date_dnum.size < min_days:
            continue # skip users with too few days
        if steps.size == 0 or dt_dnum.size == 0 or date_dnum.size == 0:
            continue

        R, C = steps.shape
        # convert times
        date = np.array([pd.Timestamp(datenum_to_dt(x)) for x in date_dnum])
        dt   = np.array([pd.Timestamp(datenum_to_dt(x)) for x in dt_dnum.ravel()]).reshape(R, C)

        for r in range(R):
            for c in range(C):
                rows.append({
                    "user_id": uid,
                    "row_idx": r,
                    "col_idx": c,
                    "date":    date[c],
                    "ts":      dt[r, c],
                    "steps":   steps[r, c],
                    "hr":      hr[r, c] if hr.size else float("nan"),
                })
    return pd.DataFrame(rows)
