"""SIMULATED inspection traffic (Phase 6a): replays val images through the inference service into an event Sink.

Every event this module writes is synthetic (`simulated=True`): timestamps are a simulated clock, images are sampled
with replacement from the val split, and drift is injected on purpose. It is not production data.
Val images only: the test split is spent, so image directories named 'test' are refused.

  python -m src.simulator --n-events 300 --drift-start 150 --seed 0 --out data/inspection/sim.parquet
"""

import argparse
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from app.schemas import ImageResult
from app.service import ACTION_PRIORITY, InspectionService, Settings
from src.inspection_log import DELTA_TABLE, STAT_COLUMNS, Sink, event_from_result, image_stats
from src.perturb import LEVELS, apply_perturbation, image_seed

DEFAULT_KINDS = ("blur", "noise", "contrast")
START_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)
_NAMESPACE = uuid.UUID("6f0c1c6e-5a52-4d1b-9a53-0d3a5f9f1a11")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class Backend(Protocol):
    """Runs one image through the inspection service; returns (result, model_version)."""

    def infer(self, image: Image.Image, name: str) -> Tuple[ImageResult, str]: ...


class ServiceBackend:
    """Calls InspectionService directly (default)."""

    def __init__(self, service: InspectionService):
        self.service = service

    def infer(self, image, name):
        return self.service.run(image, name, include_ignored=True), self.service.version or "unknown"


class HttpBackend:
    """Posts PNGs to /predict. `client` is an httpx.Client (or a FastAPI TestClient) with its base_url set."""

    def __init__(self, client):
        self.client = client

    def infer(self, image, name):
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")  # lossless, so results match the direct path
        r = self.client.post("/predict", params={"include_ignored": "true"},
                             files=[("files", (name, buf.getvalue(), "image/png"))])
        r.raise_for_status()
        body = r.json()
        return ImageResult.model_validate(body["results"][0]), body["model_version"]


@dataclass(frozen=True)
class DriftSchedule:
    """From event `start`, perturb every image; severity rises from the mildest level to the strongest over `ramp`
    events. With several kinds, each event draws one at random."""

    start: int
    ramp: int = 100
    kinds: Tuple[str, ...] = DEFAULT_KINDS

    def __post_init__(self):
        if self.start < 0 or self.ramp < 1 or not self.kinds:
            raise ValueError("need start >= 0, ramp >= 1 and at least one kind")
        for k in self.kinds:
            if k not in LEVELS:
                raise ValueError(f"unknown perturbation {k!r}; choose from {sorted(LEVELS)}")

    def at(self, i: int, rng: np.random.Generator) -> Tuple[str, int]:
        """(perturbation, severity index) for event i; ('none', 0) before the drift starts."""
        if i < self.start:
            return "none", 0
        kind = self.kinds[int(rng.integers(len(self.kinds)))]
        top = len(LEVELS[kind])
        progress = min(1.0, (i - self.start) / self.ramp)
        return kind, min(top, 1 + int(progress * top))


@dataclass(frozen=True)
class SimConfig:
    """Simulation settings. `rate_hz` sets the simulated clock; `realtime=True` also sleeps to match it."""

    n_events: int = 300
    seed: int = 0
    rate_hz: float = 5.0
    drift: Optional[DriftSchedule] = None
    start_time: datetime = START_TIME
    realtime: bool = False
    batch_size: int = 100


def make_event_id(seed: int, i: int) -> str:
    """Deterministic event id for (seed, event index)."""
    return str(uuid.uuid5(_NAMESPACE, f"{seed}:{i}"))


def list_images(image_dir: str) -> list:
    """Sorted image paths in a val directory (refuses the spent test split)."""
    d = Path(image_dir)
    if d.name.lower() == "test":
        raise ValueError("the test split is spent; the simulator uses val images only")
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"no images found in {d}")
    return paths


def simulate(backend: Backend, image_dir: str, sink: Sink, cfg: SimConfig) -> int:
    """Generate cfg.n_events SIMULATED events into the sink; returns the number written.

    Deterministic in (seed, image set, backend output) apart from latency_ms, which is measured.
    """
    paths = list_images(image_dir)
    batch, written = [], 0
    for i in range(cfg.n_events):
        rng = np.random.default_rng([cfg.seed, i])
        path = paths[int(rng.integers(len(paths)))]
        kind, sev = cfg.drift.at(i, rng) if cfg.drift else ("none", 0)
        with Image.open(path) as im:
            img = im.convert("RGB")
        if sev:
            arr = apply_perturbation(np.asarray(img), kind, sev, seed=image_seed(path.name, cfg.seed + i))
            img = Image.fromarray(arr)
        stats = image_stats(np.asarray(img))  # on the image exactly as the service receives it
        t0 = time.perf_counter()
        result, version = backend.infer(img, path.name)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        batch.append(event_from_result(
            result, event_id=make_event_id(cfg.seed, i),
            timestamp=cfg.start_time + timedelta(seconds=i / cfg.rate_hz), model_version=version,
            latency_ms=latency_ms, perturbation=kind, severity=sev, simulated=True, stats=stats))
        if cfg.realtime:
            time.sleep(max(0.0, 1.0 / cfg.rate_hz - latency_ms / 1000.0))
        if len(batch) >= cfg.batch_size:
            sink.write(batch)
            written, batch = written + len(batch), []
    if batch:
        sink.write(batch)
        written += len(batch)
    return written


def summarize(df: pd.DataFrame, drift_start: Optional[int]) -> pd.DataFrame:
    """Worst-decision counts per phase ('before'/'after' drift) for a simulated run, in event order."""
    phase = pd.Series("before", index=df.index)
    if drift_start is not None:
        phase[df.index >= drift_start] = "after"
    table = pd.crosstab(phase, df["worst_decision"]).reindex(columns=ACTION_PRIORITY, fill_value=0)
    return table.reindex([p for p in ("before", "after") if p in table.index]).rename_axis(index="phase", columns=None)


def stats_summary(df: pd.DataFrame, drift_start: Optional[int]) -> pd.DataFrame:
    """Mean of each input statistic before drift and, per perturbation kind, after it."""
    group = pd.Series("before", index=df.index)
    if drift_start is not None:
        group[df.index >= drift_start] = "after:" + df["perturbation"][df.index >= drift_start]
    return df.groupby(group)[STAT_COLUMNS].mean().rename_axis("group").round(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-events", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=5.0, help="simulated events per second")
    ap.add_argument("--realtime", action="store_true", help="also sleep to match --rate")
    ap.add_argument("--drift-start", type=int, default=None, help="event index where drift begins (default: none)")
    ap.add_argument("--drift-ramp", type=int, default=100, help="events over which severity ramps up")
    ap.add_argument("--kinds", default=",".join(DEFAULT_KINDS), help="comma-separated perturbations")
    ap.add_argument("--image-dir", default="data/yolo/images/val")
    ap.add_argument("--sink", choices=["parquet", "sqlite", "delta"], default="parquet")
    ap.add_argument("--out", default=None, help="local sinks; default: data/inspection/sim.parquet or sim.sqlite")
    ap.add_argument("--table", default=DELTA_TABLE, help="Delta table (--sink delta)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--api-url", default=None, help="use the HTTP API at this URL instead of the in-process service")
    a = ap.parse_args()

    from src.inspection_log import DeltaSink, ParquetSink, SQLiteSink

    if a.sink == "delta":
        out = a.table  # Delta writes are idempotent by event_id, so re-running the same seed adds nothing
        sink = DeltaSink(table=a.table, batch_size=100)
    else:
        out = Path(a.out or f"data/inspection/sim.{'parquet' if a.sink == 'parquet' else 'sqlite'}")
        if out.exists() and any(out.iterdir() if out.is_dir() else [out]):
            if not a.overwrite:
                ap.error(f"{out} already exists; pass --overwrite to replace it")
            import shutil

            shutil.rmtree(out) if out.is_dir() else out.unlink()
        sink = ParquetSink(str(out)) if a.sink == "parquet" else SQLiteSink(str(out))

    if a.api_url:
        import httpx

        backend = HttpBackend(httpx.Client(base_url=a.api_url, timeout=60.0))
    else:
        service = InspectionService(Settings.from_env())
        if not service.loaded:
            ap.error(f"no weights at {service.settings.model_path}")
        backend = ServiceBackend(service)
    drift = (DriftSchedule(a.drift_start, a.drift_ramp, tuple(a.kinds.split(",")))
             if a.drift_start is not None else None)
    cfg = SimConfig(a.n_events, a.seed, a.rate, drift, realtime=a.realtime)

    print("*** SIMULATED traffic: val images replayed, drift injected on purpose; not production data ***")
    n = simulate(backend, a.image_dir, sink, cfg)
    df = sink.read()
    df = df[df.event_id.isin({make_event_id(a.seed, i) for i in range(a.n_events)})]  # this run only
    df = df.sort_values("timestamp").reset_index(drop=True)
    sink.close()
    print(f"wrote {n} simulated events to {out} (seed={a.seed}, backend={'api' if a.api_url else 'service'})")
    if drift:
        print(f"drift: from event {drift.start}, ramp {drift.ramp}, kinds {list(drift.kinds)}")
    print("worst decision counts (SIMULATED):")
    print(summarize(df, a.drift_start).to_string())
    print("mean input statistics (SIMULATED):")
    print(stats_summary(df, a.drift_start).to_string())
    print(f"median latency {df.latency_ms.median():.1f} ms")


if __name__ == "__main__":
    main()
