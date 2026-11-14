"""Inspection event schema and sink tests (no weights needed)."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.schemas import Detection, ImageResult
from src.inspection_log import (COLUMNS, SCHEMA_VERSION, DeltaSink, ParquetSink, SQLiteSink, Sink, event_from_result,
                                events_frame, image_stats)
from tests.fakes import FakeConnection

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def det(action, conf):
    return Detection(class_id=0, class_name="crazing", confidence=conf, box=(0, 0, 1, 1), severity=0.5, action=action)


STATS = {"brightness_mean": 100.0, "contrast_std": 20.0, "sharpness": 50.0, "noise_est": 3.0}


def make_event(i=0, ts=T0, **kw):
    res = ImageResult(filename=f"img{i}.jpg", width=10, height=10, decision="stop_now",
                      detections=[det("stop_now", 0.9), det("ignore", 0.05), det("ignore", 0.02)])
    return event_from_result(res, event_id=f"e{i}", timestamp=ts, model_version="v1", latency_ms=12.5,
                             stats=kw.pop("stats", STATS), **kw)


def test_schema_columns_and_values():
    assert COLUMNS == ["event_id", "timestamp", "image_id", "model_version", "n_detections", "worst_decision",
                       "max_confidence", "n_stop_now", "n_schedule_service", "n_human_review", "n_monitor",
                       "n_ignore", "latency_ms", "perturbation", "severity", "simulated", "brightness_mean",
                       "contrast_std", "sharpness", "noise_est", "schema_version"]
    e = make_event(perturbation="blur", severity=2, simulated=True)
    assert (e.image_id, e.n_detections, e.worst_decision, e.max_confidence) == ("img0.jpg", 3, "stop_now", 0.9)
    assert (e.n_stop_now, e.n_ignore, e.n_monitor) == (1, 2, 0)
    assert (e.perturbation, e.severity, e.simulated) == ("blur", 2, True)
    assert (e.sharpness, e.noise_est, e.schema_version) == (50.0, 3.0, SCHEMA_VERSION)


def test_defaults_and_empty_result():
    res = ImageResult(filename="a.jpg", width=1, height=1, decision="ignore", detections=[])
    e = event_from_result(res, event_id="x", timestamp=T0, model_version="v", latency_ms=1.0)
    assert (e.n_detections, e.max_confidence, e.perturbation, e.severity, e.simulated) == (0, 0.0, "none", 0, False)
    assert np.isnan(e.sharpness) and e.schema_version == SCHEMA_VERSION


def test_events_frame_dtypes():
    df = events_frame([make_event()])
    assert list(df.columns) == COLUMNS
    assert str(df.timestamp.dtype) == "datetime64[ns, UTC]" and df.n_detections.dtype == "int64"
    assert list(events_frame([]).columns) == COLUMNS


@pytest.fixture(params=["parquet", "sqlite", "delta-fake"])
def sink(request, tmp_path):
    s = {"parquet": lambda: ParquetSink(str(tmp_path / "log")),
         "sqlite": lambda: SQLiteSink(str(tmp_path / "log.sqlite")),
         "delta-fake": lambda: DeltaSink(FakeConnection())}[request.param]()
    yield s
    s.close()


def test_sink_interface_roundtrip(sink):
    assert isinstance(sink, Sink)
    assert sink.read().empty and list(sink.read().columns) == COLUMNS
    first = [make_event(0), make_event(1, ts=T0 + pd.Timedelta(seconds=1), perturbation="noise", severity=1,
                                       simulated=True)]
    third = make_event(2, ts=T0 + pd.Timedelta(seconds=2), stats={})  # no statistics -> NaN must survive
    sink.write(first)
    sink.write([third])
    sink.write([])
    pd.testing.assert_frame_equal(sink.read(), events_frame(first + [third]))


def test_sinks_agree(tmp_path):
    evs = [make_event(i) for i in range(5)]
    p, s, d = ParquetSink(str(tmp_path / "p")), SQLiteSink(str(tmp_path / "s.sqlite")), DeltaSink(FakeConnection())
    for k in (p, s, d):
        k.write(evs)
    pd.testing.assert_frame_equal(p.read(), s.read())
    pd.testing.assert_frame_equal(p.read(), d.read())
    s.close()


def test_parquet_sink_appends_across_instances(tmp_path):
    ParquetSink(str(tmp_path / "log")).write([make_event(0)])
    ParquetSink(str(tmp_path / "log")).write([make_event(1)])
    assert list(ParquetSink(str(tmp_path / "log")).read().event_id) == ["e0", "e1"]


def test_image_stats_values_and_direction():
    flat = np.full((32, 32, 3), 100, np.uint8)
    st = image_stats(flat)
    assert st == {"brightness_mean": 100.0, "contrast_std": 0.0, "sharpness": 0.0, "noise_est": 0.0}
    rng = np.random.default_rng(0)
    noisy = np.clip(flat + rng.normal(0, 10, flat.shape), 0, 255).astype(np.uint8)
    n = image_stats(noisy)
    assert n["noise_est"] > 5 and n["sharpness"] > 100 and n["contrast_std"] > 5
    assert image_stats(flat[..., 0])["brightness_mean"] == 100.0  # gray input works
