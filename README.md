# Steps Forecast – Time‑Series ML Project

A clean, reproducible pipeline to predict **steps** from wearable data.  
You can try multiple models, track metrics, and compare results across runs.

---

## Repository Structure

```
steps-forecast/
├─ data/
│  ├─ raw/              # original MATLAB *_compat.mat (read-only)
│  ├─ interim/          # optional temp conversions
│  └─ processed/        # tidy parquet after build/features
├─ experiments/
│  ├─ results.csv       # append-only registry of runs (metrics summary)
│  └─ runs/             # artifacts per run (models, plots, config)
├─ configs/
│  ├─ base.yaml         # default experiment configuration
│  └─ custom-*.yaml     # your variants per experiment
├─ notebooks/           # optional exploratory notebooks
├─ src/
│  ├─ data/
│  │  ├─ mat_loader.py      # load *_compat.mat → tidy DataFrame
│  │  └─ make_dataset.py    # CLI: save tidy parquet to data/processed
│  ├─ features/
│  │  └─ build_features.py  # lags/rolling/calendar features & targets
│  ├─ models/
│  │  ├─ metrics.py         # MAE, RMSE, MAPE, R²
│  │  ├─ registry.py        # model factory (add your models here)
│  │  ├─ train.py           # read config → split → train → eval → save
│  │  └─ evaluate.py        # (placeholder for future standalone eval)
│  ├─ utils/
│  │  ├─ io.py              # parquet I/O, model save/load, run dirs, logger
│  │  └─ time.py            # time-based train/val/test split
│  └─ cli.py                # single CLI entrypoint: build/features/train
├─ requirements.txt
└─ README.md                # this file
```

> **Why this layout?** It cleanly separates *data ingestion*, *feature engineering*, and *modeling*.  
> Results are versioned by run folders, and each run stores the exact `config.json` for reproducibility.

---

## Prerequisites

- Python 3.10+ recommended
- Install dependencies:
  ```bash
  python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
  pip install -r requirements.txt
  ```
- MATLAB side (already done on your end): convert MCOS `datetime` fields to basic types (**datenum** or ISO strings) and save as `*_compat.mat`.  
  Example (done once in MATLAB):
  ```matlab
  load DataStepsOmar_complete.mat
  for i = 1:numel(DataComplete)
      c = DataComplete{i};
      if isfield(c,'Date'),     c.Date_dnum     = datenum(c.Date);     else, c.Date_dnum = []; end
      if isfield(c,'DateTime'), c.DateTime_dnum = datenum(c.DateTime); else, c.DateTime_dnum = []; end
      DataComplete{i} = c;
  end
  save('DataStepsOmar_complete_compat.mat','DataComplete','-v7');
  ```

Place your converted file here:
```
data/raw/DataStepsOmar_complete_compat.mat
```

---

## Configuration

All knobs live in `configs/base.yaml`. Example:

```yaml
task: "interval"            # "interval" = next slot; "daily_total" = next day sum
horizon: 1                  # steps ahead for interval task
agg_slot_minutes: 15        # reference (64 slots/day)

data:
  raw_mat_path: "data/raw/DataStepsOmar_complete_compat.mat"
  varname: "DataComplete"
  processed_path: "data/processed/steps_interval.parquet"

split:
  train_until: "2024-06-30T23:59:59"
  valid_until: "2024-08-31T23:59:59"
  min_samples_per_user: 500

features:
  use_hr: true
  lags: [1,2,3,4,8,16,32,64]
  roll_windows: [4,16,64]
  calendar: ["dow","hour","month"]
  user_id_encoding: "onehot"
  target_col: "steps_t"

models:
  - name: "naive_last"      # baseline: y_hat = steps_lag1
  - name: "linreg"
  - name: "ridge"
  - name: "rf100"

eval:
  metrics: ["MAE","RMSE","MAPE","R2"]
  top_k_plots: 3
  save_preds: true

runtime:
  random_state: 42
  n_jobs: -1
  run_note: "baseline interval t+1"
```

**What to edit most often:**
- `split.*` dates for your data range
- `features.*` (lags, windows, calendar, HR usage)
- `models` list (add/remove)
- `task/horizon` to switch between next-slot vs next-day targets

---

## Data Flow

1. **Build (ingest)**: read MAT → tidy long table with `ts`
2. **Features**: add lags, rolling stats, calendar → `steps_t` target
3. **Train**: time-based split, train models, log metrics, save artifacts

```
MAT *_compat.mat --(build)--> data/processed/*_raw.parquet
                    |
                    v
              --(features)--> data/processed/*.parquet (features + target)
                    |
                    v
               --(train)--> experiments/runs/<run-id>/*  &  experiments/results.csv
```

---

## Commands (End-to-End)

> Use the shared CLI (`src/cli.py`) so all steps read the same config.

### 1) Build dataset
```bash
python -m src.cli build --config configs/base.yaml
```
- Loads `data/raw/DataStepsOmar_complete_compat.mat`
- Writes `data/processed/steps_interval_raw.parquet`

### 2) Feature engineering
```bash
python -m src.cli features --config configs/base.yaml
```
- Reads `*_raw.parquet`
- Adds lags/rollings/calendar & target (`steps_t`)
- Writes `data/processed/steps_interval.parquet`

### 3) Train & evaluate
```bash
python -m src.cli train --config configs/base.yaml
```
- Splits by time (train/val/test)
- Trains all models in `models:`
- Saves:
  - `experiments/runs/<run-id>/*.joblib` (fitted models)
  - `experiments/runs/<run-id>/*_test_preds.png` (quick plot)
  - `experiments/runs/<run-id>/config.json` (snapshot)
  - Appends a row to `experiments/results.csv` (metrics summary)

---

## Dataset Columns

After **build** (tidy long table):
- `user_id` – subject identifier
- `row_idx`, `col_idx` – matrix indices from original (64×268)
- `date` – day (converted from `Date_dnum`)
- `ts` – exact timestamp per slot (from `DateTime_dnum`)
- `steps` – numeric steps
- `hr` – numeric heart rate

After **features** (adds example engineered columns):
- `steps_lag{L}` – lagged steps (e.g., 1,2,4,8,16,32,64)
- `steps_mean_{W}`, `steps_std_{W}` – rolling stats
- `hr_mean_{W}`, `hr_std_{W}` – optional if `use_hr: true`
- `dow`, `hour`, `month` – calendar features
- One‑hot user columns if `user_id_encoding: onehot`
- **Target**: `steps_t` (shifted by `horizon` for interval task; next‑day total for daily task)

---

## Models

Defined in `src/models/registry.py`. Default options:
- **naive_last** – baseline (`steps_lag1`)
- **linreg** – LinearRegression with standardization
- **ridge** – RidgeCV with alpha grid
- **rf100** – RandomForestRegressor (n=100)

> Add more by extending `registry.py` (e.g., XGBoost, LightGBM, CatBoost, Lasso, SVR).

Metrics in `src/models/metrics.py`:
- MAE, RMSE, MAPE, R² (extend as needed)

---

## Results & Reproducibility

- Each run gets its own timestamped folder under `experiments/runs/`
- We store a `config.json` copy used for that run
- `experiments/results.csv` acts as a central leaderboard across runs

---

## Common Tweaks

- **Predict further ahead**: set `horizon: 4` for 1‑hour ahead (15‑min slots)
- **Daily target**: set `task: "daily_total"` (switches aggregation & target build)
- **Per‑user models**: modify `train.py` to loop users and train separate models
- **More features**: edit `build_features.py` (e.g., EMA, min/max, HR deltas)
- **Rolling CV**: replace simple split with `TimeSeriesSplit` in `train.py`

---

## Troubleshooting

- **Empty splits**: check `split.train_until` / `valid_until` windows vs your data range
- **NaNs in features**: the training pipeline includes an imputer; still, verify lags/rolls
- **Performance flat**: confirm the **naive_last** baseline is included for sanity
- **Large memory**: reduce lags/rolls or work per user to profile

---

## Next Steps (optional)

- Integrate MLflow or Weights & Biases for richer experiment tracking
- Add XGBoost/LightGBM to the `registry.py`
- Add an `inference.py` to predict the next slot/day for a given user on fresh data
- Dockerize the pipeline for consistent environments

---

## License

Internal project (add your preferred license if needed).
