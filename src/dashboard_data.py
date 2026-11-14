"""Data loading and aggregation for the monitoring page (pure pandas; no Streamlit or plotting imports).

Everything here is SIMULATED data from src/simulator.py. The x-axis is event order, not the simulated clock.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.inspection_log import STAT_COLUMNS, ParquetSink, SQLiteSink, normalize

SAMPLE_PATH = "data/sample/inspection_sample.parquet"
ACTIONS = ["stop_now", "schedule_service", "human_review", "monitor", "ignore"]
PERTURBATION_ORDER = ["none", "blur", "noise", "contrast"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Simulated rows only, in time order, with `event_index` (0, 1, 2, ...) as the event-order axis."""
    df = normalize(df)
    df = df[df["simulated"]].sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    df["event_index"] = df.index
    return df


def load_local(path: str) -> pd.DataFrame:
    """Read a Parquet file or part directory, or a SQLite file, and prepare it."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if p.suffix in (".sqlite", ".db"):
        sink = SQLiteSink(str(p))
        try:
            return prepare(sink.read())
        finally:
            sink.close()
    return prepare(ParquetSink(str(p)).read() if p.is_dir() else pd.read_parquet(p))


def load_delta(sink: Optional[object] = None) -> pd.DataFrame:
    """Read the Delta table through DeltaSink.read (credentials from the environment or .env) and prepare it."""
    if sink is None:
        from src.inspection_log import DeltaSink

        sink = DeltaSink()
    try:
        return prepare(sink.read())
    finally:
        sink.close()


def drift_start(df: pd.DataFrame) -> Optional[int]:
    """Event index of the first perturbed event, or None if the run has no drift."""
    perturbed = df.loc[df["perturbation"] != "none", "event_index"]
    return int(perturbed.min()) if len(perturbed) else None


def add_bucket(df: pd.DataFrame, size: int) -> pd.DataFrame:
    """Add `bucket`: the event_index of the first event in each group of `size` consecutive events."""
    if size < 1:
        raise ValueError("bucket size must be >= 1")
    return df.assign(bucket=df["event_index"] // size * size)


def _by_bucket(df: pd.DataFrame, size: int):
    return add_bucket(df, size).groupby("bucket")


def mean_max_confidence(df: pd.DataFrame, size: int) -> pd.DataFrame:
    """Mean max_confidence per bucket."""
    return _by_bucket(df, size)["max_confidence"].mean().rename("mean_max_confidence").reset_index()


def detection_count(df: pd.DataFrame, size: int) -> pd.DataFrame:
    """Mean detections per image per bucket."""
    return _by_bucket(df, size)["n_detections"].mean().rename("mean_detections_per_image").reset_index()


def latency_percentiles(df: pd.DataFrame, size: int) -> pd.DataFrame:
    """p50 and p95 of latency_ms per bucket, excluding the first event (model warm-up)."""
    g = _by_bucket(df[df["event_index"] > 0], size)["latency_ms"]
    return pd.DataFrame({"p50_ms": g.quantile(0.50), "p95_ms": g.quantile(0.95)}).reset_index()


def input_statistics(df: pd.DataFrame, size: int) -> pd.DataFrame:
    """Mean of each input statistic per bucket (schema v2+ rows only; v1 rows have NaN statistics)."""
    return _by_bucket(df.dropna(subset=STAT_COLUMNS), size)[STAT_COLUMNS].mean().reset_index()


def action_counts(df: pd.DataFrame, size: int, include_ignore: bool = False) -> pd.DataFrame:
    """Long format (bucket, action, detections): detections per action per bucket; 'ignore' only if asked."""
    actions = ACTIONS if include_ignore else [a for a in ACTIONS if a != "ignore"]
    wide = _by_bucket(df, size)[[f"n_{a}" for a in actions]].sum()
    wide.columns = actions
    return wide.reset_index().melt(id_vars="bucket", var_name="action", value_name="detections")


def perturbation_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per perturbation type: event count and mean of max_confidence and each input statistic."""
    cols = ["max_confidence"] + STAT_COLUMNS
    out = df.groupby("perturbation")[cols].mean()
    out.insert(0, "events", df.groupby("perturbation").size())
    known = [p for p in PERTURBATION_ORDER if p in out.index]
    other: List[str] = sorted(set(out.index) - set(PERTURBATION_ORDER))
    return out.reindex(known + other).rename_axis("perturbation").reset_index()
