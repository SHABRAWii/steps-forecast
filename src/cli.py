import argparse, yaml, subprocess, sys

def main():
    ap = argparse.ArgumentParser(description="Steps Forecast CLI")
    ap.add_argument("cmd", choices=["build", "features", "train"])
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config,"r",encoding="utf-8"))
    if a.cmd == "build":
        cmd = [
          "python","-m","src.data.make_dataset",
          "--mat", cfg["data"]["raw_mat_path"],
          "--varname", cfg["data"].get("varname","DataComplete"),
          "--out", cfg["data"]["processed_path"].replace(".parquet", "_raw.parquet")
        ]
    elif a.cmd == "features":
        inp = cfg["data"]["processed_path"].replace(".parquet", "_raw.parquet")
        out = cfg["data"]["processed_path"]

        args = [
        sys.executable, "-m", "src.features.build_features",
        "--in", inp, "--out", out,
        "--task", cfg["task"], "--horizon", str(cfg["horizon"]),
        ]
        if cfg["features"]["use_hr"]:
            args.append("--use_hr")
        args += ["--lags", *map(str, cfg["features"]["lags"])]
        args += ["--rolls", *map(str, cfg["features"]["roll_windows"])]
        args += ["--calendar", *cfg["features"]["calendar"]]
        args += ["--uid_enc", cfg["features"]["user_id_encoding"]]

        # Optional extras if you enabled them in your config
        if cfg["features"].get("add_slot_day"):
            args.append("--add_slot_day")
        if cfg["features"].get("add_gap"):
            args.append("--add_gap")
        if cfg["features"].get("strict_15min"):
            args.append("--strict_15min")

        cmd = args

    else:  # train
        cmd = ["python","-m","src.models.train","--config", a.config]

    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
