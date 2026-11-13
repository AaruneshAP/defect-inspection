"""MLflow helpers: log params/metrics/artifacts to a local file store and register models."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

# MLflow >=3.18 refuses the local file store unless explicitly allowed (it is in maintenance mode).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_EXPERIMENT = "defect-inspection"


def git_commit(path: str | None = None, cwd: str | Path = ".") -> str | None:
    """Return the full hash of the last commit touching `path` (or HEAD if None); None if unavailable."""
    cmd = ["git", "log", "-1", "--format=%H"] + (["--", path] if path else [])
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


def flatten(d: Mapping[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dicts into dotted keys."""
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            flat.update(flatten(v, key, sep))
        else:
            flat[key] = v
    return flat


def clean_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Flatten and keep only finite numeric values (MLflow metrics must be floats)."""
    out: dict[str, float] = {}
    for k, v in flatten(metrics).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if math.isfinite(v):
            out[k] = float(v)
    return out


def setup(tracking_uri: str = DEFAULT_TRACKING_URI, experiment: str = DEFAULT_EXPERIMENT) -> MlflowClient:
    """Point MLflow at a store, select the experiment, and return a client."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return MlflowClient(tracking_uri)


def find_run(run_name: str, experiment: str = DEFAULT_EXPERIMENT) -> str | None:
    """Return the id of an existing active run with this name, or None."""
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return None
    runs = mlflow.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'", output_format="list")
    return runs[0].info.run_id if runs else None


def log_run(
    run_name: str,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    artifacts: list[str | Path] | None = None,
    tags: Mapping[str, str] | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment: str = DEFAULT_EXPERIMENT,
    skip_existing: bool = True,
) -> str:
    """Log one run; returns its id. With skip_existing, an existing run of the same name is reused."""
    setup(tracking_uri, experiment)
    if skip_existing:
        existing = find_run(run_name, experiment)
        if existing:
            return existing
    with mlflow.start_run(run_name=run_name) as run:
        for k, v in flatten(params or {}).items():
            mlflow.log_param(k, str(v)[:500])
        mlflow.log_metrics(clean_metrics(metrics or {}))
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})
        for a in artifacts or []:
            p = Path(a)
            if p.is_file():
                mlflow.log_artifact(str(p))
            else:
                raise FileNotFoundError(f"artifact not found: {p}")
        return run.info.run_id


def register_model(
    run_id: str,
    weights: str | Path,
    name: str,
    alias: str = "production",
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> str:
    """Upload `weights` to the run, register a new version of model `name`, point `alias` at it. Returns version."""
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    client = setup(tracking_uri)
    client.log_artifact(run_id, str(weights), artifact_path="model")
    try:
        client.create_registered_model(name)
    except mlflow.exceptions.MlflowException:
        pass  # already exists
    source = f"{client.get_run(run_id).info.artifact_uri}/model/{weights.name}"
    mv = client.create_model_version(name, source, run_id)
    client.set_registered_model_alias(name, alias, mv.version)
    return str(mv.version)
