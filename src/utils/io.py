import json, os, hashlib, time, csv
from pathlib import Path
from joblib import dump, load
from pathlib import Path
import joblib, os, tempfile

def ensure_dir(p: str|Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_parquet(df, path):
    ensure_dir(Path(path).parent)
    df.to_parquet(path, index=False)

def read_parquet(path):
    return __import__("pandas").read_parquet(path)

# def save_model(obj, path):
#     ensure_dir(Path(path).parent)
#     dump(obj, path)

def load_model(path):
    return load(path)

def sha1_of_string(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]

def new_run_dir(root="experiments/runs", prefix="run"):
    ts = time.strftime("%Y%m%d-%H%M%S")
    rd = Path(root) / f"{prefix}-{ts}"
    ensure_dir(rd)
    return rd

def append_results_row(row: dict, csv_path="experiments/results.csv"):
    ensure_dir(Path(csv_path).parent)
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists: w.writeheader()
        w.writerow(row)

def write_json(d, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, default=str)
def save_model(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # atomic write: write to temp then replace
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        joblib.dump(obj, tmp_path)
        os.replace(tmp_path, path)  # atomic on same filesystem
    finally:
        if tmp_path.exists():
            try: tmp_path.unlink()
            except OSError:
                pass