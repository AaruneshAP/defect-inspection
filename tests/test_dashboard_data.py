"""Dashboard data loading and aggregation, on fabricated events and a fake Delta connection (no UI, no network)."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src import dashboard_data as dd
from src.inspection_log import DeltaSink, InspectionEvent, ParquetSink, SQLiteSink, events_frame
from tests.fakes import FakeConnection

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_events(n=20, drift_at=10, simulated=True):
    kinds = ["blur", "noise", "contrast"]
    out = []
    for i in range(n):
        kind = "none" if i < drift_at else kinds[i % 3]
        out.append(InspectionEvent(
            event_id=f"e{i:03d}", timestamp=T0 + timedelta(seconds=i), image_id=f"img{i}.jpg", model_version="v",
            n_detections=i % 4, worst_decision="monitor", max_confidence=0.5 + 0.01 * i, n_stop_now=1,
            n_schedule_service=0, n_human_review=0, n_monitor=1, n_ignore=i % 4, latency_ms=100.0 + i,
            perturbation=kind, severity=0 if kind == "none" else 1, simulated=simulated, brightness_mean=100.0,
            contrast_std=20.0 + i, sharpness=50.0 + i, noise_est=3.0))
    return out


def test_prepare_filters_simulated_sorts_and_indexes():
    mixed = make_events(5) + [replace(make_events(1)[0], event_id="real", simulated=False)]
    df = dd.prepare(events_frame(list(reversed(mixed))))
    assert len(df) == 5 and df["simulated"].all()
    assert list(df["event_index"]) == [0, 1, 2, 3, 4]
    assert df["timestamp"].is_monotonic_increasing


def test_drift_start():
    assert dd.drift_start(dd.prepare(events_frame(make_events(20, drift_at=10)))) == 10
    assert dd.drift_start(dd.prepare(events_frame(make_events(20, drift_at=99)))) is None


def test_bucket_aggregations():
    df = dd.prepare(events_frame(make_events(20, drift_at=10)))
    conf = dd.mean_max_confidence(df, 10)
    assert conf["mean_max_confidence"].iloc[0] == pytest.approx(0.5 + 0.01 * 4.5)
    lat = dd.latency_percentiles(df, 20)
    assert lat["p50_ms"].iloc[0] == pytest.approx(110.0)  # events 1..19 only: event 0 (warm-up) is excluded
    assert lat["p95_ms"].iloc[0] > lat["p50_ms"].iloc[0]
    assert dd.detection_count(df, 20)["mean_detections_per_image"].iloc[0] == pytest.approx(sum(i % 4 for i in range(20)) / 20)
    with pytest.raises(ValueError):
        dd.add_bucket(df, 0)


def test_action_counts_long_format():
    df = dd.prepare(events_frame(make_events(10)))
    act = dd.action_counts(df, 5)
    assert "ignore" not in set(act["action"]) and len(act) == 2 * (len(dd.ACTIONS) - 1)
    assert act[act["action"] == "stop_now"]["detections"].tolist() == [5, 5]
    full = dd.action_counts(df, 5, include_ignore=True)
    assert set(full["action"]) == set(dd.ACTIONS) and len(full) == 2 * len(dd.ACTIONS)
    assert full[full["action"] == "ignore"]["detections"].sum() == sum(i % 4 for i in range(10))


def test_input_statistics_skip_missing_stats():
    df = dd.prepare(events_frame(make_events(10)))
    df.loc[:4, "sharpness"] = float("nan")  # as in v1 rows
    stats = dd.input_statistics(df, 5)
    assert list(stats["bucket"]) == [5]


def test_perturbation_breakdown_order_and_means():
    df = dd.prepare(events_frame(make_events(20, drift_at=10)))
    bd = dd.perturbation_breakdown(df)
    assert list(bd["perturbation"]) == ["none", "blur", "noise", "contrast"]
    assert bd["events"].sum() == 20
    assert bd.loc[bd["perturbation"] == "none", "noise_est"].iloc[0] == pytest.approx(3.0)


def test_load_local_parquet_file_dir_and_sqlite(tmp_path):
    events = make_events(12)
    frame = events_frame(events)
    frame.to_parquet(tmp_path / "one.parquet", index=False)
    pq = ParquetSink(str(tmp_path / "dir"))
    pq.write(events[:6])
    pq.write(events[6:])
    sq = SQLiteSink(str(tmp_path / "log.sqlite"))
    sq.write(events)
    sq.close()
    for path in ("one.parquet", "dir", "log.sqlite"):
        df = dd.load_local(str(tmp_path / path))
        assert len(df) == 12 and list(df["event_index"]) == list(range(12))
    with pytest.raises(FileNotFoundError):
        dd.load_local(str(tmp_path / "missing.parquet"))


def test_load_delta_with_fake_connection():
    conn = FakeConnection()
    sink = DeltaSink(connection=conn, table="defect_inspection.inspection_events")
    sink.write(make_events(8))
    df = dd.load_delta(DeltaSink(connection=conn))
    assert len(df) == 8 and conn.closed and list(df["event_index"]) == list(range(8))


def test_committed_sample_snapshot_is_simulated_only():
    path = Path(dd.SAMPLE_PATH)
    if not path.exists():
        pytest.skip("sample snapshot not present")
    raw = pd.read_parquet(path)
    assert len(raw) == 300 and raw["simulated"].all()
    assert len(dd.load_local(str(path))) == 300


def test_dashboard_imports_without_torch_ultralytics_or_app_service(tmp_path):
    import subprocess
    import sys
    sample = "data/sample/inspection_sample.parquet"
    if not Path(sample).exists():
        p = tmp_path / "tiny.parquet"
        events_frame(make_events(2)).to_parquet(p, index=False)
        sample = str(p)

    cmd = [
        sys.executable, "-c",
        "import sys; "
        "sys.modules['torch'] = None; "
        "sys.modules['ultralytics'] = None; "
        "sys.modules['app.service'] = None; "
        "from src import dashboard_data as dd; "
        f"df = dd.load_local({sample!r}); "
        "assert len(df) >= 1"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Import or sample load failed: {res.stderr}"
