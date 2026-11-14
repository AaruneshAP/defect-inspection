"""Inspection event log: one row per inspected image, written through a small Sink interface.

Callers only use `Sink.write / read / close`. Sinks: ParquetSink, SQLiteSink (local) and DeltaSink (Databricks).
Schema v2 (Phase 6b) adds input image statistics and `schema_version`; v1 logs load with NaN statistics.

  python -m src.inspection_log load data/inspection/sim.parquet   # upload a local log to Delta (idempotent)
"""

import argparse
import os
import re
import sqlite3
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

SCHEMA_VERSION = 2
DELTA_TABLE = "defect_inspection.inspection_events"
STAT_COLUMNS = ["brightness_mean", "contrast_std", "sharpness", "noise_est"]
ACTION_PRIORITY: List[str] = ["stop_now", "schedule_service", "human_review", "monitor", "ignore"]
_NAN = float("nan")


@dataclass(frozen=True)
class InspectionEvent:
    """One inspected image. Counts cover all detections, including those the policy maps to 'ignore'.

    The four statistics describe the (grayscale, 0-255) image as the service received it, after any perturbation.
    """

    event_id: str
    timestamp: datetime  # tz-aware UTC
    image_id: str
    model_version: str
    n_detections: int
    worst_decision: str  # most urgent action; 'ignore' if no detections
    max_confidence: float  # 0.0 if no detections
    n_stop_now: int
    n_schedule_service: int
    n_human_review: int
    n_monitor: int
    n_ignore: int
    latency_ms: float
    perturbation: str = "none"  # 'none', 'blur', 'noise', 'contrast', ...
    severity: int = 0  # severity index from src.perturb (0 = unperturbed)
    simulated: bool = False  # True for synthetic traffic (the simulator)
    brightness_mean: float = _NAN
    contrast_std: float = _NAN
    sharpness: float = _NAN  # variance of the Laplacian
    noise_est: float = _NAN  # 1.4826 * MAD of the Laplacian
    schema_version: int = SCHEMA_VERSION


COLUMNS: List[str] = [f.name for f in fields(InspectionEvent)]
_INT_COLS = ["n_detections", "n_stop_now", "n_schedule_service", "n_human_review", "n_monitor", "n_ignore",
             "severity", "schema_version"]
_FLOAT_COLS = ["max_confidence", "latency_ms"] + STAT_COLUMNS
_MISSING_DEFAULTS = {"schema_version": 1, **{c: _NAN for c in STAT_COLUMNS}}  # columns absent from v1 logs
assert [f"n_{a}" for a in ACTION_PRIORITY] == [c for c in COLUMNS if c.startswith("n_") and c != "n_detections"]


def image_stats(img: np.ndarray) -> Dict[str, float]:
    """brightness_mean, contrast_std, sharpness, noise_est of a uint8 RGB (H, W, 3) or gray (H, W) array."""
    import cv2

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float64)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    mad = float(np.median(np.abs(lap - np.median(lap))))
    return {"brightness_mean": float(gray.mean()), "contrast_std": float(gray.std()),
            "sharpness": float(lap.var()), "noise_est": 1.4826 * mad}


def event_from_result(result: Any, *, event_id: str, timestamp: datetime, model_version: str,
                      latency_ms: float, perturbation: str = "none", severity: int = 0,
                      simulated: bool = False, stats: Optional[Dict[str, float]] = None) -> InspectionEvent:
    """Build an event from an ImageResult (call the service with include_ignored=True so counts are complete)."""
    counts = {f"n_{a}": 0 for a in ACTION_PRIORITY}
    for d in result.detections:
        counts[f"n_{d.action}"] += 1
    return InspectionEvent(event_id=event_id, timestamp=timestamp, image_id=result.filename or "",
                           model_version=model_version, n_detections=len(result.detections),
                           worst_decision=result.decision,
                           max_confidence=max((d.confidence for d in result.detections), default=0.0),
                           latency_ms=latency_ms, perturbation=perturbation, severity=severity,
                           simulated=simulated, **counts, **(stats or {}))


def events_frame(events: Sequence[InspectionEvent]) -> pd.DataFrame:
    """Events as a DataFrame with the canonical columns and dtypes."""
    df = pd.DataFrame([{c: getattr(e, c) for c in COLUMNS} for e in events], columns=COLUMNS)
    return normalize(df)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a frame read from any sink to the canonical dtypes (columns missing from v1 logs get defaults)."""
    df = df.copy()
    for c, default in _MISSING_DEFAULTS.items():
        if c not in df:
            df[c] = default
    df = df[COLUMNS]
    ts = df["timestamp"]
    df["timestamp"] = pd.to_datetime(ts, utc=True, format="ISO8601" if ts.dtype == object else None).astype(
        "datetime64[ns, UTC]")
    for c in _INT_COLS:
        df[c] = df[c].astype("int64")
    for c in _FLOAT_COLS:
        df[c] = df[c].astype("float64")
    df["simulated"] = df["simulated"].astype(bool)
    return df.reset_index(drop=True)


def _py(v):
    """numpy/pandas scalar -> plain Python value (what database drivers bind)."""
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v.item() if hasattr(v, "item") else v


@runtime_checkable
class Sink(Protocol):
    """Where events go. Implementations: ParquetSink, SQLiteSink, DeltaSink."""

    def write(self, events: Sequence[InspectionEvent]) -> None:
        """Append events."""

    def read(self) -> pd.DataFrame:
        """All events written so far with the canonical columns and dtypes."""

    def close(self) -> None:
        """Release resources (a no-op for sinks that hold none)."""


class ParquetSink:
    """A directory of Parquet part files, one per write() call."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _parts(self) -> List[Path]:
        return sorted(self.path.glob("part-*.parquet"))

    def write(self, events):
        if events:
            events_frame(events).to_parquet(self.path / f"part-{len(self._parts()):05d}.parquet", index=False)

    def read(self):
        parts = self._parts()
        return normalize(pd.concat([pd.read_parquet(p) for p in parts])) if parts else events_frame([])

    def close(self):
        pass


class SQLiteSink:
    """A single SQLite table `inspection_events`."""

    _SQL_TYPES = {"int64": "INTEGER", "float64": "REAL", "bool": "INTEGER"}

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        cols = ", ".join(f"{c} {self._SQL_TYPES.get(str(t), 'TEXT')}"
                         for c, t in events_frame([]).dtypes.items())
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS inspection_events ({cols})")
        self.conn.commit()

    def write(self, events):
        if not events:
            return
        df = events_frame(events)
        df["timestamp"] = df["timestamp"].map(lambda t: t.isoformat())
        rows = df.astype(object).itertuples(index=False, name=None)
        self.conn.executemany(f"INSERT INTO inspection_events ({','.join(COLUMNS)}) "
                              f"VALUES ({','.join('?' * len(COLUMNS))})", [tuple(_py(v) for v in r) for r in rows])
        self.conn.commit()

    def read(self):
        df = pd.read_sql_query("SELECT * FROM inspection_events ORDER BY rowid", self.conn)
        return normalize(df) if len(df) else events_frame([])

    def close(self):
        self.conn.close()


# ---- Databricks -------------------------------------------------------------------------------------------------

DATABRICKS_VARS = ("DATABRICKS_SERVER_HOSTNAME", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN")
_SQL_TYPE_OF = {"int64": "BIGINT", "float64": "DOUBLE", "bool": "BOOLEAN", "datetime64[ns, UTC]": "TIMESTAMP"}
_IDENT = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*){0,2}$")


def load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE parser; real environment variables win over the file."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


def missing_databricks_vars() -> List[str]:
    """Names of unset Databricks variables, after loading a git-ignored .env if present."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    return [k for k in DATABRICKS_VARS if not os.environ.get(k)]


def connect_from_env():
    """Open a databricks-sql-connector connection from the environment. Errors never contain the token."""
    missing = missing_databricks_vars()
    if missing:
        raise RuntimeError(f"missing Databricks settings: {', '.join(missing)}")
    token = os.environ["DATABRICKS_TOKEN"]
    try:
        from databricks import sql

        return sql.connect(server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"].removeprefix("https://").rstrip("/"),
                           http_path=os.environ["DATABRICKS_HTTP_PATH"], access_token=token)
    except Exception as exc:
        raise RuntimeError(f"Databricks connection failed: {type(exc).__name__}: "
                           f"{str(exc).replace(token, '***')[:300]}") from None


class DeltaSink:
    """Delta table via databricks-sql-connector. write() is idempotent by event_id (MERGE ... WHEN NOT MATCHED)
    and sends `batch_size` rows per parameterized statement. `connection` may be injected (tests use a fake)."""

    def __init__(self, connection=None, table: str = DELTA_TABLE, batch_size: int = 100):
        if not _IDENT.match(table):
            raise ValueError(f"invalid table name {table!r}")
        self.conn, self.table, self.batch_size = connection or connect_from_env(), table, batch_size
        schema_cols = ", ".join(f"{c} {_SQL_TYPE_OF.get(str(t), 'STRING')}"
                                for c, t in events_frame([]).dtypes.items())
        if "." in table:
            self._execute(f"CREATE SCHEMA IF NOT EXISTS {table.rsplit('.', 1)[0]}")
        self._execute(f"CREATE TABLE IF NOT EXISTS {table} ({schema_cols}) USING DELTA")
        self._casts = {c: _SQL_TYPE_OF.get(str(t), "STRING") for c, t in events_frame([]).dtypes.items()}

    def _execute(self, sql: str, params: Optional[dict] = None, fetch: bool = False):
        with self.conn.cursor() as cur:
            cur.execute(sql, params) if params else cur.execute(sql)
            if fetch:
                return [tuple(r) for r in cur.fetchall()], [d[0] for d in cur.description]

    def write(self, events):
        unique = list({e.event_id: e for e in events}.values())  # a MERGE source must not repeat a key
        for start in range(0, len(unique), self.batch_size):
            records = events_frame(unique[start:start + self.batch_size]).to_dict("records")
            params, tuples = {}, []
            for r, rec in enumerate(records):
                for c in COLUMNS:
                    params[f"{c}_{r}"] = _py(rec[c])
                tuples.append("(" + ", ".join(f"CAST(:{c}_{r} AS {self._casts[c]})" for c in COLUMNS) + ")")
            cols = ", ".join(COLUMNS)
            self._execute(
                f"MERGE INTO {self.table} AS t USING (SELECT * FROM VALUES {', '.join(tuples)} AS v({cols})) AS s "
                f"ON t.event_id = s.event_id WHEN NOT MATCHED THEN INSERT ({cols}) "
                f"VALUES ({', '.join('s.' + c for c in COLUMNS)})", params)

    def read(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Events ordered by timestamp, optionally only the first `limit`."""
        q = f"SELECT {', '.join(COLUMNS)} FROM {self.table} ORDER BY timestamp, event_id"
        if limit is not None:
            q += f" LIMIT {int(limit)}"
        rows, names = self._execute(q, fetch=True)
        return normalize(pd.DataFrame(rows, columns=names)) if rows else events_frame([])

    def count(self) -> int:
        """Number of rows in the table."""
        return int(self._execute(f"SELECT COUNT(*) FROM {self.table}", fetch=True)[0][0][0])

    def close(self):
        self.conn.close()


def df_to_events(df: pd.DataFrame) -> List[InspectionEvent]:
    """Rows of a (possibly v1) frame as events."""
    return [InspectionEvent(**{c: _py(v) for c, v in rec.items()}) for rec in normalize(df).to_dict("records")]


def load_log(source: Sink, dest: Sink, batch_size: int = 100) -> int:
    """Copy every event from one sink to another in batches; returns the number of events read."""
    events = df_to_events(source.read())
    for i in range(0, len(events), batch_size):
        dest.write(events[i:i + batch_size])
    return len(events)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ld = sub.add_parser("load", help="upload a local Parquet log to Delta (re-running adds no duplicates)")
    ld.add_argument("path")
    ld.add_argument("--table", default=DELTA_TABLE)
    ld.add_argument("--batch-size", type=int, default=100)
    a = ap.parse_args()
    dest = DeltaSink(table=a.table, batch_size=a.batch_size)
    before = dest.count()
    n = load_log(ParquetSink(a.path), dest, a.batch_size)
    after = dest.count()
    dest.close()
    print(f"read {n} events from {a.path}; {a.table}: {before} -> {after} rows ({after - before} new)")


if __name__ == "__main__":
    main()
