
# src/models/train.py
from __future__ import annotations
import joblib, gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse, time
import yaml, joblib
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm

from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer   # <-- not used; we pre-impute in pandas

from src.utils.io import read_parquet, save_model, new_run_dir, append_results_row, write_json
from src.utils.time import time_split_global, time_split_per_user, time_split_per_user_pct
from src.models.metrics import mae, rmse, mape, r2, perc_abs_error, tol_accuracy, directional_accuracy
from src.models.registry import make_model
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.utils.io import read_parquet, save_model, new_run_dir, append_results_row, write_json
from src.utils.time import (
    time_split_global,
    time_split_per_user,
    time_split_per_user_pct,   # <-- NEW: percentage-based per-user splitter
)
from src.models.metrics import mae, rmse, mape, r2, perc_abs_error, tol_accuracy, directional_accuracy
from src.models.registry import make_model


def select_xy(df: pd.DataFrame, task: str, drop_current_steps: bool = True):
    """
    Select features (X) and target (y) for training.
    - Drops the target column from features.
    - Keeps only numeric columns in X to avoid sklearn type errors.
    """
    target = "steps_t"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Did you run the features step with the right task?")

    # Avoid leakage: only keep numeric features and drop the target itself
    y = df[target].values
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    if drop_current_steps and "steps" in X.columns:
        X = X.drop(columns=["steps"])
    return X, y


def plot_preds(y: np.ndarray, yhat: np.ndarray, path: Path, n: int = 500):
    """
    Save a simple line plot comparing true vs predicted for the first n samples.
    """
    n = min(n, len(y), len(yhat))
    plt.figure()
    idx = np.arange(n)
    plt.plot(idx, y[:n], label="true")
    plt.plot(idx, yhat[:n], label="pred")
    plt.legend()
    plt.title("Predictions (first {} samples)".format(n))
    plt.grid(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
# in train.py before fit()
def make_sample_weight(y, low=50, high=400, max_w=2.0):
    w = np.ones_like(y, dtype="float32")
    # only up-weight when target is in a “useful” peak band
    mask = (y >= low) & (y <= high)
    # linear ramp from 1 → max_w across [low, high]
    w[mask] = 1.0 + (max_w - 1.0) * ((y[mask] - low) / (high - low))
    # very large spikes get weight=1 again (we don’t chase outliers)
    return w

def main(cfg_path: str, resume_from: str = "", verbose: bool = False):
    # ---- read config & data -------------------------------------------------
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    df = read_parquet(cfg["data"]["processed_path"])

    skip_on_oom = bool(cfg.get("runtime", {}).get("skip_on_oom", True))

    # allow YAML fallback too
    if not resume_from:
        resume_from = cfg.get("runtime", {}).get("resume_from", "") or ""

    # if resume_from:
    #     run_dir = Path(resume_from)
    #     print(f"[Resume] Resuming run: {run_dir}")
    # else:
    #     run_dir = Path(new_run_dir())
    #     print(f"[Run] New run dir: {run_dir}")
        
    # Optional: filter users with enough samples BEFORE splitting
    split_cfg = cfg.get("split", {})
    min_samples = split_cfg.get("min_samples_per_user")
    if min_samples:
        counts = df.groupby("user_id").size()
        keep = counts[counts >= min_samples].index
        df = df[df["user_id"].isin(keep)]

    # ---- choose split mode --------------------------------------------------
    mode = split_cfg.get("mode", "global_time")  # "global_time" | "per_user" | "per_user_pct"
    if mode == "per_user":
        tr, va, te = time_split_per_user(
            df, "ts",
            val_days=split_cfg.get("per_user", {}).get("val_days", 14),
            test_days=split_cfg.get("per_user", {}).get("test_days", 14),
        )
    elif mode == "per_user_pct":
        p = split_cfg.get("per_user_pct", {})
        tr, va, te = time_split_per_user_pct(
            df, "ts",
            val_pct=p.get("val_pct", 0.15),
            test_pct=p.get("test_pct", 0.15),
            min_days_for_any_split=p.get("min_days_for_any_split", 10),
            min_val_days=p.get("min_val_days", 1),
            min_test_days=p.get("min_test_days", 1),
        )
    else:  # "global_time"
        tr, va, te = time_split_global(
            df, "ts",
            split_cfg["train_until"],
            split_cfg["valid_until"],
        )

    print(f"[Split] train: {len(tr):,} rows | val: {len(va):,} | test: {len(te):,}")
    print(f"[Users] train: {tr['user_id'].nunique()} | val: {va['user_id'].nunique()} | test: {te['user_id'].nunique()}")

    # ---- select features/targets -------------------------------------------
    Xtr, ytr = select_xy(tr, cfg["task"])
    Xva, yva = select_xy(va, cfg["task"])
    Xte, yte = select_xy(te, cfg["task"])

    # ---- prepare run directory ---------------------------------------------
    run_dir = None
    # ---- prepare / resume run directory ------------------------------------
    if resume_from:
        run_dir = Path(resume_from)
        print(f"[Resume] Resuming run: {run_dir}")
    else:
        run_dir = Path(new_run_dir())
        print(f"[Run] New run dir: {run_dir}")
    (Path(run_dir) / "artifacts").mkdir(parents=True, exist_ok=True)
    write_json(cfg, Path(run_dir) / "config.json")

    # ---- iterate models -----------------------------------------------------
    model_list = cfg.get("models", [])
        # ---- iterate models & collect comparison rows --------------------------
    tol_k = cfg.get("eval", {}).get("tolerance_steps", 100)  # default 100 steps
    records = []

    # We'll keep the naive MAE (val/test) to compute lift later
    naive_mae_val = None
    naive_mae_test = None

    for model_spec in model_list:
        name = model_spec["name"] if isinstance(model_spec, dict) else str(model_spec)
        print(f"\n== Training: {name} ==")

        if "Xtr_f" not in locals():
            Xtr_f = Xtr.fillna(0).astype("float32")
            Xva_f = Xva.fillna(0).astype("float32")
            Xte_f = Xte.fillna(0).astype("float32")
            
        # Prepare references for directional accuracy (if available)
        ref_prev_val = va["steps_lag1"].to_numpy() if "steps_lag1" in va.columns else None
        ref_prev_test = te["steps_lag1"].to_numpy() if "steps_lag1" in te.columns else None

        if name.lower() == "naive_last":
            if "steps_lag1" not in va.columns or "steps_lag1" not in te.columns:
                raise AssertionError("Naive baseline requires 'steps_lag1'. Add lags in features step.")
            # Predict (no fitting)
            yhat_tr = tr["steps_lag1"].to_numpy() if "steps_lag1" in tr.columns else None
            yhat_va = va["steps_lag1"].to_numpy()
            yhat_te = te["steps_lag1"].to_numpy()
            model = None
            model_path = None
        else:
            model_path = Path(run_dir) / "artifacts" / f"{name}.joblib"
            yhat_tr = None
            dropped_train = False  # track if we drop Xtr/ytr to free RAM

            if model_path.exists():
                print(f"[Resume] {name}: artifact exists — loading and scoring.")
                try:
                    # first try: mem-map
                    pipe = joblib.load(model_path, mmap_mode="r")
                except Exception as e1:
                    print(f"[WARN] Loading {name} with mmap failed ({e1}). Freeing train matrices and retrying...")
                    # free the biggest stuff first
                    try:
                        del Xtr, ytr
                    except NameError:
                        pass
                    gc.collect()
                    try:
                        pipe = joblib.load(model_path, mmap_mode="r")
                    except Exception as e2:
                        msg = f"[OOM] Could not load {name} ({e2})."
                        print(msg)
                        if skip_on_oom:
                            # record a SKIPPED row so CSV still gets written and the run continues
                            record = {
                                "model": name,
                                "status": "SKIPPED_OOM",
                                "split_mode": mode, "task": cfg["task"], "horizon": cfg.get("horizon"),
                                "MAE_train": np.nan,
                                "MAE_val": np.nan, "RMSE_val": np.nan, "MAPE_val": np.nan, "R2_val": np.nan,
                                "MAE_test": np.nan, "RMSE_test": np.nan, "MAPE_test": np.nan, "R2_test": np.nan,
                                "P50_abs_err_val": np.nan, "P75_abs_err_val": np.nan,
                                "P90_abs_err_val": np.nan, "P95_abs_err_val": np.nan,
                                "P50_abs_err_test": np.nan, "P75_abs_err_test": np.nan,
                                "P90_abs_err_test": np.nan, "P95_abs_err_test": np.nan,
                                f"ToleranceAcc@{tol_k}_val": np.nan, f"ToleranceAcc@{tol_k}_test": np.nan,
                                "DirectionalAcc_val": np.nan, "DirectionalAcc_test": np.nan,
                                "model_path": str(model_path), "run_dir": str(run_dir),
                                "note": "Skipped during resume due to MemoryError",
                            }
                            records.append(record)
                            # go to next model
                            continue
                        else:
                            # if you set skip_on_oom: false, crash loudly so you notice
                            raise

            else:
                # Build model and fit
                model = make_model(
                    name,
                    random_state=cfg["runtime"].get("random_state", 42),
                    n_jobs=cfg["runtime"].get("n_jobs", -1),
                )
                pipe = Pipeline([
                    ("impute", SimpleImputer()),
                    ("model", model),
                ])
                # # Optional memory saver: float32
                # Xtr_f = Xtr.astype("float32")
                # Xva_f = Xva.astype("float32")
                # Xte_f = Xte.astype("float32")
                if name == "hgb_mae":
                    w_tr = make_sample_weight(ytr, low=80, high=400, max_w=1.5)
                    pipe.fit(Xtr_f, ytr, model__sample_weight=w_tr)
                    del w_tr
                else:
                    pipe.fit(Xtr_f, ytr)
                yhat_tr = pipe.predict(Xtr_f)

                # Save model
                model_path = Path(run_dir) / "artifacts" / f"{name}.joblib"
                save_model(pipe, model_path)
            yhat_va = pipe.predict(Xva_f)
            yhat_te = pipe.predict(Xte_f)

        # ---- metrics (core) -------------------------------------------------
        mae_tr = mae(ytr, yhat_tr) if yhat_tr is not None else np.nan
        mae_va = mae(yva, yhat_va)
        rmse_va = rmse(yva, yhat_va)
        mape_va = mape(yva, yhat_va)
        r2_va = r2(yva, yhat_va)

        mae_te = mae(yte, yhat_te)
        rmse_te = rmse(yte, yhat_te)
        mape_te = mape(yte, yhat_te)
        r2_te = r2(yte, yhat_te)

        # ---- extra stats ----------------------------------------------------
        p50_va = perc_abs_error(yva, yhat_va, 50)
        p75_va = perc_abs_error(yva, yhat_va, 75)
        p90_va = perc_abs_error(yva, yhat_va, 90)
        p95_va = perc_abs_error(yva, yhat_va, 95)

        p50_te = perc_abs_error(yte, yhat_te, 50)
        p75_te = perc_abs_error(yte, yhat_te, 75)
        p90_te = perc_abs_error(yte, yhat_te, 90)
        p95_te = perc_abs_error(yte, yhat_te, 95)

        tolacc_va = tol_accuracy(yva, yhat_va, tol_k)
        tolacc_te = tol_accuracy(yte, yhat_te, tol_k)

        diracc_va = directional_accuracy(yva, yhat_va, ref_prev_val)
        diracc_te = directional_accuracy(yte, yhat_te, ref_prev_test)

        # Keep naive MAEs for lift
        if name.lower() == "naive_last":
            naive_mae_val = mae_va
            naive_mae_test = mae_te

        # print tolerance accuracy for val/test
        print(f"  Val   - MAE: {mae_va:.1f} | RMSE: {rmse_va:.1f} | MAPE: {mape_va:.1f}% | R2: {r2_va:.3f} | Tol@{tol_k}: {tolacc_va:.3f} | DirAcc: {diracc_va:.3f}")
        print(f"  Test  - MAE: {mae_te:.1f} | RMSE: {rmse_te:.1f} | MAPE: {mape_te:.1f}% | R2: {r2_te:.3f} | Tol@{tol_k}: {tolacc_te:.3f} | DirAcc: {diracc_te:.3f}")
        # We’ll compute lift after loop when we know naive
        record = {
            "model": name,
            f"ToleranceAcc@{tol_k}_val": tolacc_va,
            f"ToleranceAcc@{tol_k}_test": tolacc_te,
            "split_mode": mode,
            "task": cfg["task"],
            "horizon": cfg.get("horizon"),

            # core metrics
            "MAE_train": mae_tr,
            "MAE_val": mae_va,
            "RMSE_val": rmse_va,
            "MAPE_val": mape_va,
            "R2_val": r2_va,

            "MAE_test": mae_te,
            "RMSE_test": rmse_te,
            "MAPE_test": mape_te,
            "R2_test": r2_te,

            # extra
            "P50_abs_err_val": p50_va,
            "P75_abs_err_val": p75_va,
            "P90_abs_err_val": p90_va,
            "P95_abs_err_val": p95_va,

            "P50_abs_err_test": p50_te,
            "P75_abs_err_test": p75_te,
            "P90_abs_err_test": p90_te,
            "P95_abs_err_test": p95_te,


            "DirectionalAcc_val": diracc_va,
            "DirectionalAcc_test": diracc_te,

            "model_path": str(model_path) if model_path else "",
            "run_dir": str(run_dir),
            "note": cfg.get("runtime", {}).get("run_note", ""),
        }
        records.append(record)

        # quick plot (test)
        plot_preds(yte, yhat_te, Path(run_dir) / f"{name}_test_preds.png", n=500)

        # also append the simple summary row to the global experiments/results.csv
        append_results_row({
            "run_dir": str(run_dir),
            "model": name,
            "task": cfg["task"],
            "split_mode": mode,
            "horizon": cfg.get("horizon"),
            "MAE_val": mae_va,
            "RMSE_val": rmse_va,
            "MAPE_val": mape_va,
            "R2_val": r2_va,
            "MAE_test": mae_te,
            "RMSE_test": rmse_te,
            "MAPE_test": mape_te,
            "R2_test": r2_te,
            "note": cfg.get("runtime", {}).get("run_note", ""),
        })

    # ---- compute lift vs naive and save comparison CSV ---------------------
    df_cmp = pd.DataFrame.from_records(records)

    if naive_mae_val is not None:
        df_cmp["Lift_val_MAE"] = (naive_mae_val - df_cmp["MAE_val"]) / naive_mae_val
    else:
        df_cmp["Lift_val_MAE"] = np.nan

    if naive_mae_test is not None:
        df_cmp["Lift_test_MAE"] = (naive_mae_test - df_cmp["MAE_test"]) / naive_mae_test
    else:
        df_cmp["Lift_test_MAE"] = np.nan

    # Rank by validation MAE (lower is better)
    df_cmp = df_cmp.sort_values(["MAE_val", "RMSE_val"], ascending=[True, True]).reset_index(drop=True)
    df_cmp.insert(0, "rank", df_cmp.index + 1)

    # Save CSV inside the run folder
    cmp_path = Path(run_dir) / "model_compare.csv"
    df_cmp.to_csv(cmp_path, index=False, encoding="utf-8")
    print("Saved comparison CSV:", cmp_path)

    print("\nRun saved to:", run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume_from", default="", help="Existing run dir to resume")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args.config, resume_from=args.resume_from, verbose=args.verbose)

