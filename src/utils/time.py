# src/utils/time.py
from __future__ import annotations
import numpy as np
import pandas as pd

def time_split_global(df: pd.DataFrame, ts_col: str, train_until: str, valid_until: str):
    """
    Single global time cut:
      - train: <= train_until
      - val:   (train_until, valid_until]
      - test:  > valid_until
    """
    ts_train_until = pd.Timestamp(train_until)
    ts_valid_until = pd.Timestamp(valid_until)

    tr = df[df[ts_col] <= ts_train_until]
    va = df[(df[ts_col] > ts_train_until) & (df[ts_col] <= ts_valid_until)]
    te = df[df[ts_col] > ts_valid_until]
    return tr, va, te


def time_split_per_user(df: pd.DataFrame, ts_col: str, val_days: int = 14, test_days: int = 14):
    """
    Per-user fixed-day split on DISTINCT days:
      For each user timeline:
        - last `test_days` days  -> test
        - preceding `val_days`   -> val
        - rest                   -> train
    """
    parts = []
    for uid, g in df.sort_values(ts_col).groupby("user_id", sort=False):
        if g.empty:
            continue
        g = g.copy()
        g["day"] = g[ts_col].dt.floor("D")
        days = g["day"].drop_duplicates().sort_values()
        n = len(days)

        if n <= (val_days + test_days):
            tr_idx = g.index
            va_idx = pd.Index([], dtype=tr_idx.dtype)
            te_idx = pd.Index([], dtype=tr_idx.dtype)
        else:
            te_days = set(days.iloc[-test_days:]) if test_days > 0 else set()
            va_days = set(days.iloc[-(test_days + val_days):-test_days]) if val_days > 0 else set()

            te_idx = g[g["day"].isin(te_days)].index
            va_idx = g[g["day"].isin(va_days)].index
            tr_idx = g[~g.index.isin(va_idx.union(te_idx))].index

        parts.append((df.loc[tr_idx], df.loc[va_idx], df.loc[te_idx]))

    if not parts:
        empty = df.iloc[0:0]
        return empty, empty, empty

    tr = pd.concat([p[0] for p in parts], axis=0).sort_values(ts_col)
    va = pd.concat([p[1] for p in parts], axis=0).sort_values(ts_col)
    te = pd.concat([p[2] for p in parts], axis=0).sort_values(ts_col)
    return tr, va, te


def time_split_per_user_pct(
    df: pd.DataFrame,
    ts_col: str,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
    min_days_for_any_split: int = 10,
    min_val_days: int = 1,
    min_test_days: int = 1,
):
    """
    Per-user PERCENTAGE split on DISTINCT days:
      For each user timeline:
        - last ~test_pct of days  -> test
        - preceding ~val_pct      -> val
        - rest                    -> train

    Notes:
      * Percentages are applied to the number of DISTINCT days per user.
      * We ensure at least one train day remains.
      * Users with < min_days_for_any_split days are put entirely in train.
    """
    assert 0 <= val_pct < 1 and 0 <= test_pct < 1 and (val_pct + test_pct) < 1, \
        "val_pct and test_pct must be in [0,1) and sum to < 1"

    parts = []
    for uid, g in df.sort_values(ts_col).groupby("user_id", sort=False):
        if g.empty:
            continue

        g = g.copy()
        g["day"] = g[ts_col].dt.floor("D")
        days = g["day"].drop_duplicates().sort_values()
        n = len(days)

        # Too few days → all train for stability
        if n < min_days_for_any_split:
            tr_idx = g.index
            va_idx = pd.Index([], dtype=tr_idx.dtype)
            te_idx = pd.Index([], dtype=tr_idx.dtype)
            parts.append((df.loc[tr_idx], df.loc[va_idx], df.loc[te_idx]))
            continue

        # Raw counts from percentages
        n_test = int(np.floor(n * test_pct))
        n_val  = int(np.floor(n * val_pct))

        # Enforce minimum day counts if splitting
        if test_pct > 0:
            n_test = max(n_test, min_test_days)
        if val_pct > 0:
            n_val  = max(n_val,  min_val_days)

        # Leave at least 1 train day
        max_for_splits = max(0, n - 1)
        if n_val + n_test > max_for_splits:
            overflow = (n_val + n_test) - max_for_splits
            # reduce validation first, then test
            reduce_val = min(n_val, overflow)
            n_val -= reduce_val
            overflow -= reduce_val
            n_test -= min(n_test, overflow)

        te_days = set(days.iloc[-n_test:]) if n_test > 0 else set()
        va_days = set(days.iloc[-(n_test + n_val):-n_test]) if n_val > 0 else set()

        te_idx = g[g["day"].isin(te_days)].index
        va_idx = g[g["day"].isin(va_days)].index
        tr_idx = g[~g.index.isin(va_idx.union(te_idx))].index

        parts.append((df.loc[tr_idx], df.loc[va_idx], df.loc[te_idx]))

    if not parts:
        empty = df.iloc[0:0]
        return empty, empty, empty

    tr = pd.concat([p[0] for p in parts], axis=0).sort_values(ts_col)
    va = pd.concat([p[1] for p in parts], axis=0).sort_values(ts_col)
    te = pd.concat([p[2] for p in parts], axis=0).sort_values(ts_col)
    return tr, va, te
