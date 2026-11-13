# Serving the model (FastAPI + Docker)

## Endpoints
| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `{"status":"ok","model_loaded":bool}`; works even without weights |
| GET | `/model-info` | model name, version (`sha256:<12 hex>` of the weights file), optional registry alias, frozen policy |
| POST | `/predict` | multipart field `files` (1..8 images: JPEG/PNG/BMP, <=10 MB each). Optional query param `include_ignored=true` (default false omits detections whose action is `ignore`). Returns detections (class, confidence, box, severity, action) and a per-image `decision`; every response carries `model_version` |

Errors: 415 wrong content type, 400 undecodable/unsupported/empty file or too many files, 413 file too large, 422 missing field, 503 weights not loaded.

Decisions use `src/decision.py` unchanged with the frozen `primary` policy in `reports/phase3/frozen.json` and `configs/cost.yaml`. The per-image `decision` is the most urgent detection action (`stop_now` > `schedule_service` > `human_review` > `monitor` > `ignore`); that roll-up order is a serving-layer choice.

## Configuration (environment variables)
`MODEL_PATH` (default `models/best.pt`; `/models/best.pt` in the image), `FROZEN_PATH`, `COST_CONFIG`, `POLICY_KEY` (default `primary`), `MAX_UPLOAD_BYTES` (10485760), `MAX_FILES` (8), optional `MLFLOW_TRACKING_URI` + `MODEL_ALIAS` to report the registry alias (needs mlflow installed; not in the image).

## Run locally
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --port 8000
```

## Docker
Weights are not baked in; mount them read-only.
```bash
docker build -t defect-inspection .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/models:ro" defect-inspection
# PowerShell: -v "${PWD}/models:/models:ro"
curl http://localhost:8000/health
curl -F "files=@sample.jpg;type=image/jpeg" http://localhost:8000/predict
# To include ignored detections:
curl -F "files=@sample.jpg;type=image/jpeg" "http://localhost:8000/predict?include_ignored=true"
```
Different weights file: `-e MODEL_PATH=/models/other.pt`.

The image is `python:3.13-slim`, CPU torch, headless OpenCV, non-root user (uid 10001), with a `HEALTHCHECK` on `/health`. `constraints.txt` is a `pip freeze` of the working venv with Windows-only packages (such as `pywin32`) removed. Regenerate it after changing requirements.

## CI
The `docker-build` job builds the image (no push), starts it without weights, and checks `/health`, `/model-info`, `/predict` -> 503 and that the process is not root.
