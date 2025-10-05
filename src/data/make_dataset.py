import argparse, pandas as pd
from pathlib import Path
from src.utils.io import write_parquet
from src.data.mat_loader import load_tidy_from_compat

def main(raw_mat_path, varname, out_path):
    df = load_tidy_from_compat(raw_mat_path, varname)
    # basic cleaning
    df = df.sort_values(["user_id","ts"]).reset_index(drop=True)
    write_parquet(df, out_path)
    print("Saved:", out_path, "rows:", len(df))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--varname", default="DataComplete")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    main(a.mat, a.varname, a.out)
