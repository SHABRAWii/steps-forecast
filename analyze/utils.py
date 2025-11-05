# analyze/utils.py
from __future__ import annotations
import os, json, datetime
from pathlib import Path

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def new_run_dir(base: str) -> str:
    d = Path(base) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    return str(d)

def write_json(obj, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
