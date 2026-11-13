# Experiment tracking (MLflow, local file store)

Runs, metrics, artifacts and the model registry live in `./mlruns/` (git-ignored).

## Setup
```powershell
.\scripts\setup.ps1                       # creates .venv, installs requirements-dev.txt
.\.venv\Scripts\Activate.ps1
```

## Populate the store from the stored results (idempotent)
```powershell
python -m scripts.backfill_mlflow         # E0-E3, phase3_test, registers defect-detector (alias: production)
```
Needs `reports/phase2b/raw/phase2b_outputs/` and `models/best.pt` locally (both git-ignored).
Each run carries a `git_commit` tag: the commit that recorded those results (not the Kaggle training code state).

## Launch the UI
```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"     # MLflow >= 3.18 requires this opt-in for the file store
mlflow ui --backend-store-uri ./mlruns    # http://127.0.0.1:5000
```
Registered model: `defect-detector`, alias `production` -> E0 (YOLOv8s, 640) weights.
Load by alias: `models:/defect-detector@production` (weights file is `best.pt`).

## Logging a new run
```python
from src import tracking as T
rid = T.log_run("my_run", params={...}, metrics={...}, artifacts=["path/file.json"], tags={"git_commit": T.git_commit()})
```

## Notes
- MLflow has put the filesystem backend in maintenance mode; migrate with `mlflow migrate-filestore` / a SQLite URI when needed.
- Dataset: NEU surface-defect data, licence unknown; it is not committed and CI never downloads it.
