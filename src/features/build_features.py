# src/features/build_features.py
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from src.utils.io import read_parquet, write_parquet

# ---------- helpers: optional robustness for irregular data ----------

def add_slot_and_day(df: pd.DataFrame) -> pd.DataFrame:
    """Add slot index (0..63 for 15-min slots) and day column."""
    df = df.sort_values(["user_id", "ts"])
    df["slot_in_day"] = df["ts"].dt.hour * 4 + (df["ts"].dt.minute // 15)
    df["day"] = df["ts"].dt.floor("D")
    return df

def add_gap_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Minutes since previous observed row per user."""
    df = df.sort_values(["user_id", "ts"])
    prev_ts = df.groupby("user_id")["ts"].shift(1)
    df["delta_minutes_from_prev"] = (df["ts"] - prev_ts).dt.total_seconds() / 60.0
    return df

# ---------- calendar, lags, rolls, targets ----------

def add_calendar(df: pd.DataFrame, which: list[str] | None = None) -> pd.DataFrame:
    """
    Add selected calendar features. which can contain any of:
    'dow', 'hour', 'month'. If None/empty -> do nothing.
    """
    if not which:
        return df
    if "dow" in which:
        df["dow"] = df["ts"].dt.dayofweek   # 0=Mon..6=Sun
    if "hour" in which:
        df["hour"] = df["ts"].dt.hour
    if "month" in which:
        df["month"] = df["ts"].dt.month
    return df

def make_lags(df: pd.DataFrame, target_col="steps", lags=(1,2,3,4,8,16,32,64)) -> pd.DataFrame:
    for L in lags:
        df[f"{target_col}_lag{L}"] = df.groupby("user_id")[target_col].shift(L)
    return df

def make_rolls(df: pd.DataFrame, cols=("steps","hr"), windows=(4,16,64)) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        for w in windows:
            df[f"{c}_mean_{w}"] = df.groupby("user_id")[c].transform(lambda s: s.rolling(w, min_periods=1).mean())
            df[f"{c}_std_{w}"]  = df.groupby("user_id")[c].transform(lambda s: s.rolling(w, min_periods=1).std())
    return df
def make_rolls_past_only(df, cols=("hr",), windows=(4,16,64)):
    # windows computed on series shifted by 1 → only <= t-1 contributes
    for c in cols:
        s = df.groupby("user_id", sort=False)[c].shift(1)  # exclude current slot
        for w in windows:
            df[f"{c}_mean_{w}"] = (
                s.groupby(df["user_id"]).transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
            df[f"{c}_std_{w}"] = (
                s.groupby(df["user_id"]).transform(lambda x: x.rolling(w, min_periods=2).std())
            )
    return df


def add_targets_interval(df: pd.DataFrame, horizon=1) -> pd.DataFrame:
    df["steps_t"] = df.groupby("user_id")["steps"].shift(-horizon)
    return df

def add_targets_daily_total(df: pd.DataFrame) -> pd.DataFrame:
    # aggregate daily total per user, then create next-day target
    daily = (df.assign(day=df["ts"].dt.floor("D"))
               .groupby(["user_id", "day"], as_index=False)
               .agg(steps=("steps","sum"), hr=("hr","mean")))
    daily = daily.sort_values(["user_id","day"])
    daily["steps_t"] = daily.groupby("user_id")["steps"].shift(-1)
    return daily
def add_hr_dynamics(df):
    # needs hr_lag1 at least
    if "hr_lag1" in df.columns:
        df["hr_delta1"] = (df["hr"] - df["hr_lag1"]).astype("float32")
    # rolling max & std on past-only HR
    g = df.groupby("user_id", sort=False)["hr"].shift(1)  # <= t-1
    for w in (4, 16, 64):
        df[f"hr_max_{w}"] = g.groupby(df["user_id"]).transform(lambda x: x.rolling(w, min_periods=1).max()).astype("float32")
        df[f"hr_std_{w}"] = g.groupby(df["user_id"]).transform(lambda x: x.rolling(w, min_periods=2).std()).astype("float32")
    return df

# ---------- main pipeline ----------
def main(in_path, out_path, task, horizon, use_hr, lags, rolls, calendar, user_id_encoding, 
         allow_step_inputs, add_slot_day=True, add_gap=True, strict_15min=False):
    

    """
    - add_slot_day:   adds slot_in_day (0..63) and 'day' columns
    - add_gap:        adds delta_minutes_from_prev
    - strict_15min:   keep rows where previous gap == 15 or gap is NaN (first row of a user/day)
    - calendar:       list of selected calendar fields (dow/hour/month); empty -> none
    """
    df = read_parquet(in_path).sort_values(["user_id","ts"]).reset_index(drop=True)

    if add_slot_day:
        df = add_slot_and_day(df)
    if add_gap:
        df = add_gap_minutes(df)
    if strict_15min:
        df = df[(df["delta_minutes_from_prev"].isna()) | (df["delta_minutes_from_prev"] == 15)]

    # selective calendar
    if calendar: df = add_calendar(df)

    if task == "interval":
        # TARGET: current or future depending on horizon
        df = add_targets_interval(df, horizon=horizon)
        df = df.dropna(subset=["steps_t"])  # for horizon=0, this keeps all rows

        if allow_step_inputs:
            # (not your case) original behavior that can include step features
            if lags:   df = make_lags(df, "steps", tuple(lags))
            if use_hr and lags: df = make_lags(df, "hr", tuple(lags))
            if rolls:
                cols = ("steps","hr") if use_hr else ("steps",)
                df = make_rolls(df, cols, tuple(rolls))
        else:
            # STRICT: no step-derived inputs; HR-only and past-only rolls
            if use_hr and lags:
                df = make_lags(df, "hr", tuple(lags))             # hr_lag1, hr_lag2, ...
            if use_hr:
                df = add_hr_dynamics(df)
            if use_hr and rolls:
                df = make_rolls_past_only(df, ("hr",), tuple(rolls))  # hr_mean_*, hr_std_*

            # ensure raw 'steps' is not used as a feature later
            # (we keep it in the frame because steps_t equals steps, but we'll drop it from X)
    else:
        df = add_targets_daily_total(df)

    # user id encoding (optional)
    if user_id_encoding == "onehot" and "user_id" in df.columns:
        # Keep the original 'user_id' for grouping/filtering later.
        dummies = pd.get_dummies(
            df["user_id"],
            prefix="user_id",
            drop_first=True,
            dtype="uint8",
        )
        df = pd.concat([df, dummies], axis=1)

    write_parquet(df, out_path)
    print("Saved features:", out_path, "rows:", len(df))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", choices=["interval","daily_total"], required=True)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--use_hr", action="store_true")
    ap.add_argument("--lags", nargs="*", type=int, default=[1,2,3,4,8,16,32,64])
    ap.add_argument("--rolls", nargs="*", type=int, default=[4,16,64])
    ap.add_argument("--calendar", nargs="*", default=["dow","hour","month"])  # selective now
    ap.add_argument("--uid_enc", choices=["onehot","drop"], default="onehot")
    ap.add_argument("--allow_step_inputs", action="store_true")

    # NEW toggles for irregular data handling
    ap.add_argument("--add_slot_day", action="store_true")
    ap.add_argument("--add_gap", action="store_true")
    ap.add_argument("--strict_15min", action="store_true")

    a = ap.parse_args()
    main(
        a.inp, a.out, a.task, a.horizon, a.use_hr, a.lags, a.rolls, a.calendar, a.uid_enc,
        add_slot_day=a.add_slot_day, add_gap=a.add_gap, strict_15min=a.strict_15min, allow_step_inputs=a.allow_step_inputs
    )
