# src/tools/recover_run.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import yaml

from src.utils.io import read_parquet
from src.models.metrics import (
    mae, rmse, mape, r2, perc_abs_error, tol_accuracy,
    directional_accuracy, wape, smape
)
from src.utils.time import (
    time_split_global, time_split_per_user, time_split_per_user_pct
)
from src.models.train import select_xy  # reuse the same feature selection

def _load_cfg(run_dir: Path, fallback_cfg: str | None):
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if fallback_cfg:
        return yaml.safe_load(open(fallback_cfg, "r", encoding="utf-8"))
    raise FileNotFoundError(f"No config.json in {run_dir} and no --config provided.")

def _time_split(df: pd.DataFrame, cfg: dict):
    split_cfg = cfg.get("split", {})
    mode = split_cfg.get("mode", "global_time")
    if mode == "per_user":
        return time_split_per_user(
            df, "ts",
            val_days=split_cfg.get("per_user", {}).get("val_days", 14),
            test_days=split_cfg.get("per_user", {}).get("test_days", 14),
        )
    elif mode == "per_user_pct":
        p = split_cfg.get("per_user_pct", {})
        return time_split_per_user_pct(
            df, "ts",
            val_pct=p.get("val_pct", 0.15),
            test_pct=p.get("test_pct", 0.15),
            min_days_for_any_split=p.get("min_days_for_any_split", 10),
            min_val_days=p.get("min_val_days", 1),
            min_test_days=p.get("min_test_days", 1),
        )
    else:
        return time_split_global(
            df, "ts",
            split_cfg["train_until"],
            split_cfg["valid_until"],
        )

def main(run_dir: str, config_fallback: str | None, tol_list: list[int] | None):
    run_dir = Path(run_dir)
    cfg = _load_cfg(run_dir, config_fallback)

    # ---- load processed/features parquet
    df = read_parquet(cfg["data"]["processed_path"])
    tr, va, te = _time_split(df, cfg)

    # select X/y (same as training)
    Xtr, ytr = select_xy(tr, cfg["task"])
    Xva, yva = select_xy(va, cfg["task"])
    Xte, yte = select_xy(te, cfg["task"])

    # default tolerance thresholds
    if tol_list is None:
        tol_list = cfg.get("eval", {}).get("tolerance_steps_list", None) or \
                   [cfg.get("eval", {}).get("tolerance_steps", 100)]
    tol_list = list(map(int, tol_list))

    # prepare references for directional accuracy
    ref_prev_val = va["steps_lag1"].to_numpy() if "steps_lag1" in va.columns else None
    ref_prev_test = te["steps_lag1"].to_numpy() if "steps_lag1" in te.columns else None

    records = []

    # ---- add naive_last baseline
    if "steps_lag1" in va.columns and "steps_lag1" in te.columns:
        yhat_va = va["steps_lag1"].to_numpy()
        yhat_te = te["steps_lag1"].to_numpy()
        row = {
            "model": "naive_last",
            "MAE_val": mae(yva, yhat_va), "RMSE_val": rmse(yva, yhat_va),
            "MAPE_val": mape(yva, yhat_va), "R2_val": r2(yva, yhat_va),
            "MAE_test": mae(yte, yhat_te), "RMSE_test": rmse(yte, yhat_te),
            "MAPE_test": mape(yte, yhat_te), "R2_test": r2(yte, yhat_te),
            "WAPE_val": wape(yva, yhat_va), "WAPE_test": wape(yte, yhat_te),
            "SMAPE_val": smape(yva, yhat_va), "SMAPE_test": smape(yte, yhat_te),
            "P50_abs_err_val": perc_abs_error(yva, yhat_va, 50),
            "P75_abs_err_val": perc_abs_error(yva, yhat_va, 75),
            "P90_abs_err_val": perc_abs_error(yva, yhat_va, 90),
            "P95_abs_err_val": perc_abs_error(yva, yhat_va, 95),
            "P50_abs_err_test": perc_abs_error(yte, yhat_te, 50),
            "P75_abs_err_test": perc_abs_error(yte, yhat_te, 75),
            "P90_abs_err_test": perc_abs_error(yte, yhat_te, 90),
            "P95_abs_err_test": perc_abs_error(yte, yhat_te, 95),
            "DirectionalAcc_val": directional_accuracy(yva, yhat_va, ref_prev_val),
            "DirectionalAcc_test": directional_accuracy(yte, yhat_te, ref_prev_test),
            "model_path": "", "run_dir": str(run_dir),
        }
        for K in tol_list:
            row[f"ToleranceAcc@{K}_val"] = tol_accuracy(yva, yhat_va, K)
            row[f"ToleranceAcc@{K}_test"] = tol_accuracy(yte, yhat_te, K)
        records.append(row)

    # ---- evaluate every saved model artifact in this run
    art_dir = run_dir / "artifacts"
    for p in sorted(art_dir.glob("*.joblib")):
        name = p.stem  # filename without .joblib
        try:
            model = joblib.load(p)
            # Prepare data (float32 helps memory; pipeline handles its own preprocessing)
            Xva_f = Xva.fillna(0).astype("float32")
            Xte_f = Xte.fillna(0).astype("float32")
            yhat_va = model.predict(Xva_f)
            yhat_te = model.predict(Xte_f)

            row = {
                "model": name,
                "MAE_val": mae(yva, yhat_va), "RMSE_val": rmse(yva, yhat_va),
                "MAPE_val": mape(yva, yhat_va), "R2_val": r2(yva, yhat_va),
                "MAE_test": mae(yte, yhat_te), "RMSE_test": rmse(yte, yhat_te),
                "MAPE_test": mape(yte, yhat_te), "R2_test": r2(yte, yhat_te),
                "WAPE_val": wape(yva, yhat_va), "WAPE_test": wape(yte, yhat_te),
                "SMAPE_val": smape(yva, yhat_va), "SMAPE_test": smape(yte, yhat_te),
                "P50_abs_err_val": perc_abs_error(yva, yhat_va, 50),
                "P75_abs_err_val": perc_abs_error(yva, yhat_va, 75),
                "P90_abs_err_val": perc_abs_error(yva, yhat_va, 90),
                "P95_abs_err_val": perc_abs_error(yva, yhat_va, 95),
                "P50_abs_err_test": perc_abs_error(yte, yhat_te, 50),
                "P75_abs_err_test": perc_abs_error(yte, yhat_te, 75),
                "P90_abs_err_test": perc_abs_error(yte, yhat_te, 90),
                "P95_abs_err_test": perc_abs_error(yte, yhat_te, 95),
                "DirectionalAcc_val": directional_accuracy(yva, yhat_va, ref_prev_val),
                "DirectionalAcc_test": directional_accuracy(yte, yhat_te, ref_prev_test),
                "model_path": str(p), "run_dir": str(run_dir),
            }
            for K in tol_list:
                row[f"ToleranceAcc@{K}_val"] = tol_accuracy(yva, yhat_va, K)
                row[f"ToleranceAcc@{K}_test"] = tol_accuracy(yte, yhat_te, K)
            records.append(row)
        except Exception as e:
            print(f"[WARN] Failed to score {p.name}: {e}")

    # ---- compute lift vs naive and rank
    df_cmp = pd.DataFrame.from_records(records)
    try:
        naive_mae_val = float(df_cmp.loc[df_cmp["model"].str.lower()=="naive_last","MAE_val"].iloc[0])
        df_cmp["Lift_val_MAE"] = (naive_mae_val - df_cmp["MAE_val"]) / naive_mae_val
    except Exception:
        df_cmp["Lift_val_MAE"] = np.nan
    try:
        naive_mae_test = float(df_cmp.loc[df_cmp["model"].str.lower()=="naive_last","MAE_test"].iloc[0])
        df_cmp["Lift_test_MAE"] = (naive_mae_test - df_cmp["MAE_test"]) / naive_mae_test
    except Exception:
        df_cmp["Lift_test_MAE"] = np.nan

    if "MAE_val" in df_cmp.columns and "RMSE_val" in df_cmp.columns:
        df_cmp = df_cmp.sort_values(["MAE_val","RMSE_val"], ascending=[True,True]).reset_index(drop=True)
        df_cmp.insert(0, "rank", df_cmp.index + 1)

    out = run_dir / "model_compare.csv"
    df_cmp.to_csv(out, index=False, encoding="utf-8")
    print("Rebuilt comparison CSV:", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to existing run dir (experiments/runs/...)")
    ap.add_argument("--config", default="", help="YAML fallback if config.json is missing")
    ap.add_argument("--tolerances", nargs="*", type=int, default=None, help="Tolerance steps list, e.g. 50 100 200")
    a = ap.parse_args()
    main(a.run, a.config or None, a.tolerances)
