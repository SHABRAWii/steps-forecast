#!/usr/bin/env python3
# analyze/mat_analyze.py  — no console prints; writes one Excel workbook with sheets
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

# ---------- run utils ----------
def timestamp():
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def new_run_dir(base: str) -> str:
    p = Path(base) / timestamp()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def write_json(obj, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

# ---------- MATLAB compat loader ----------
def datenum_to_dt(x):
    # MATLAB datenum: days since 0000-01-00. Convert to Python datetime.
    from datetime import datetime, timedelta
    x = float(x)
    return datetime.fromordinal(int(x)) + timedelta(days=x % 1) - timedelta(days=366)

def load_tidy_from_compat(mat_path: str, varname="DataComplete") -> pd.DataFrame:
    """
    Reads DataStepsOmar_complete_compat.mat to a tidy DataFrame with columns:
      user_id, row_idx, col_idx, date, ts, steps_t, heart_rate
    Assumes the struct has fields: UserID, Steps, HR, Date_dnum, DateTime_dnum
    """
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    if varname not in mat:
        raise SystemExit(f"Variable '{varname}' not found. Keys: {list(mat.keys())}")
    dc = mat[varname]
    if hasattr(dc, "ravel"):
        dc = dc.ravel().tolist()

    rows = []
    for cell in dc:
        uid = cell.get("UserID", None)
        steps = np.asarray(cell.get("Steps", []), float)
        hr    = np.asarray(cell.get("HR", []), float)
        date_dnum  = np.asarray(cell.get("Date_dnum", []), float)        # (C,)
        dt_dnum    = np.asarray(cell.get("DateTime_dnum", []), float)    # (R,C)

        if steps.size == 0 or dt_dnum.size == 0 or date_dnum.size == 0:
            continue

        # shapes
        if steps.ndim == 1: steps = steps.reshape(1, -1)
        if hr.size and hr.ndim == 1: hr = hr.reshape(1, -1)
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
                    "steps_t":   steps[r, c],
                    "heart_rate": hr[r, c] if hr.size else float("nan"),
                })
    df = pd.DataFrame(rows)
    # types / sort
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.date
    df = df.sort_values(["user_id","ts"]).reset_index(drop=True)
    return df


def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with any timezone-aware datetime columns converted to tz-naive (Excel requirement).
    Uses the recommended dtype check to avoid pandas deprecation warnings.
    """
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype

        # tz-aware datetime64
        if isinstance(dtype, pd.DatetimeTZDtype):
            out[col] = out[col].dt.tz_convert("UTC").dt.tz_localize(None)
            continue

        # objects that may contain Timestamp with tz
        if pd.api.types.is_object_dtype(dtype):
            try:
                s = pd.to_datetime(out[col], errors="ignore", utc=True)
                if isinstance(s.dtype, pd.DatetimeTZDtype):
                    out[col] = s.dt.tz_convert("UTC").dt.tz_localize(None)
                else:
                    out[col] = s
            except Exception:
                # leave as-is if not parseable
                pass

    return out


# ---------- summary/statistics ----------
def summarize_dataset(df: pd.DataFrame, slot_mins: float, active_thr: float):
    d = df.copy()
    d["dow"] = pd.to_datetime(d["ts"]).dt.dayofweek
    d["hour"] = pd.to_datetime(d["ts"]).dt.hour
    slots_per_day_expected = int(round(24*60/float(slot_mins)))
    per_user = d.groupby("user_id", as_index=False).agg(
        rows=("ts","size"),
        days=("date", lambda x: len(pd.unique(x))),
        first_ts=("ts","min"),
        last_ts=("ts","max"),
        mean_steps=("steps_t","mean"),
        median_steps=("steps_t","median"),
        zero_frac=("steps_t", lambda x: float((np.asarray(x)==0).mean())),
    )
    per_user["rows_per_day"] = per_user["rows"] / per_user["days"].replace(0, np.nan)
    per_user["days_complete_approx"] = (per_user["rows"] // slots_per_day_expected).astype(int)

    by_uday = d.groupby(["user_id","date"]).size().rename("rows").reset_index()
    by_uday["complete"] = (by_uday["rows"] >= slots_per_day_expected).astype(int)
    comp_overall = by_uday["complete"].mean() if len(by_uday) else np.nan

    hr_cov = 1.0 - d["heart_rate"].isna().mean() if "heart_rate" in d.columns else np.nan
    active_frac = float((d["steps_t"] >= active_thr).mean()) if len(d) else np.nan

    hod = d.groupby("hour")["steps_t"].agg(["count", "mean", "median", "std"]).reset_index()
    prior = d.groupby(["user_id","dow","hour"], as_index=False)["steps_t"].mean().rename(columns={"steps_t":"prior_usr_dow_hour"})

    lite = dict(
        n_rows=int(len(df)),
        n_users=int(df["user_id"].nunique()),
        date_start=str(pd.to_datetime(d["ts"]).min()),
        date_end=str(pd.to_datetime(d["ts"]).max()),
        slots_per_day_expected=slots_per_day_expected,
        active_threshold=active_thr,
        active_fraction=active_frac,
        per_day_complete_fraction=float(comp_overall) if comp_overall==comp_overall else None,
        hr_col_present=bool("heart_rate" in d.columns),
        hr_coverage=float(hr_cov) if hr_cov==hr_cov else None,
    )
    return lite, per_user, hod, prior, by_uday

def write_excel(run_dir: str,
                summary_kv: dict,
                per_user: pd.DataFrame,
                hod: pd.DataFrame,
                prior: pd.DataFrame,
                by_uday: pd.DataFrame,
                dataset_flat: pd.DataFrame):
    # Build a key/value DataFrame for summary
    kv_rows = [{"key": k, "value": v} for k, v in summary_kv.items()]
    df_kv = pd.DataFrame(kv_rows)

    # Excel writer
    xlsx_path = Path(run_dir) / "analyze_report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        # Small readme
        readme = pd.DataFrame({
            "info": [
                "This workbook contains the analysis outputs.",
                "Sheets: SUMMARY, PER_USER, HOUR_OF_DAY, PRIOR_USER_DOW_HOUR, BY_USER_DAY, DATASET_SAMPLE",
                "Note: DATASET_SAMPLE shows up to 100,000 rows to keep Excel responsive."
            ]
        })
        readme.to_excel(xw, index=False, sheet_name="README")

        df_kv.to_excel(xw, index=False, sheet_name="SUMMARY")
        _per_user = _strip_tz(per_user)
        _hod = _strip_tz(hod)
        _prior = _strip_tz(prior)
        _by_uday = _strip_tz(by_uday)
        _per_user.to_excel(xw, index=False, sheet_name="PER_USER")
        _hod.to_excel(xw, index=False, sheet_name="HOUR_OF_DAY")
        _prior.to_excel(xw, index=False, sheet_name="PRIOR_USER_DOW_HOUR")
        _by_uday.to_excel(xw, index=False, sheet_name="BY_USER_DAY")

        # Sample of the raw dataset (avoid Excel 1,048,576 row limit)
        n_sample = min(len(dataset_flat), 100_000)
        _sample = _strip_tz(dataset_flat.head(n_sample))
        _sample.to_excel(xw, index=False, sheet_name="DATASET_SAMPLE")

    return str(xlsx_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML config path under analyze/config/*.yaml")
    ap.add_argument("--mat", help="Path to .mat file", default=None)
    ap.add_argument("--var", default="DataComplete", help="Variable name inside MAT")
    ap.add_argument("--out", default="analyze/runs", help="Base output dir for runs")
    ap.add_argument("--slot-mins", type=float, default=22.5, help="Use 22.5 for 64 segments/day")
    ap.add_argument("--active-thr", type=float, default=50.0)
    args = ap.parse_args()

    # Config support
    cfg = {}
    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    mat_path = args.mat or cfg.get("data",{}).get("mat_path")
    varname  = args.var or cfg.get("data",{}).get("varname","DataComplete")
    out_base = args.out or cfg.get("runtime",{}).get("out_dir","analyze/runs")
    slot_mins = cfg.get("analysis",{}).get("slot_mins", args.slot_mins)
    active_thr = cfg.get("analysis",{}).get("active_thr", args.active_thr)

    if not mat_path:
        raise SystemExit("Provide --mat or a config with data.mat_path")

    run_dir = new_run_dir(out_base)
    run_cfg = {"data":{"mat_path":mat_path,"varname":varname},
               "analysis":{"slot_mins":slot_mins,"active_thr":active_thr},
               "runtime":{"out_dir":out_base,"run_dir":run_dir}}
    write_json(run_cfg, Path(run_dir)/"config_used.json")

    # Load & analyze
    df = load_tidy_from_compat(mat_path, varname=varname)
    df.to_csv(Path(run_dir)/"dataset_flat.csv", index=False)  # full table for programmatic use

    summary_kv, per_user, hod, prior, by_uday = summarize_dataset(df, slot_mins=slot_mins, active_thr=active_thr)

    # Save JSON copies for programmatic use
    write_json(summary_kv, Path(run_dir)/"summary.json")
    # CSVs remain useful for downstream code
    per_user.to_csv(Path(run_dir)/"per_user_stats.csv", index=False)
    hod.to_csv(Path(run_dir)/"hour_of_day_stats.csv", index=False)
    prior.to_csv(Path(run_dir)/"prior_user_dow_hour.csv", index=False)
    by_uday.to_csv(Path(run_dir)/"by_user_day.csv", index=False)

    # Excel workbook with sheets
    write_excel(run_dir, summary_kv, per_user, hod, prior, by_uday, df)

if __name__ == "__main__":
    main()
