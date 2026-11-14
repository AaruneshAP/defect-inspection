"""DeltaSink tests against a fake connection (no network); the live test needs DATABRICKS_* variables."""

import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.inspection_log import (COLUMNS, DeltaSink, ParquetSink, connect_from_env, events_frame, load_log,
                                missing_databricks_vars)
from tests.fakes import FakeConnection
from tests.test_inspection_log import make_event

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def events(n, start=0):
    return [make_event(i, ts=T0 + timedelta(seconds=i)) for i in range(start, start + n)]


def test_creates_schema_and_table_with_matching_types():
    conn = FakeConnection()
    DeltaSink(conn)
    create = [s for s, _ in conn.statements if s.startswith("CREATE TABLE")][0]
    assert conn.statements[0][0] == "CREATE SCHEMA IF NOT EXISTS defect_inspection"
    assert "defect_inspection.inspection_events" in create and "USING DELTA" in create
    for frag in ("event_id STRING", "timestamp TIMESTAMP", "n_detections BIGINT", "sharpness DOUBLE",
                 "simulated BOOLEAN", "schema_version BIGINT"):
        assert frag in create


def test_batches_rows_into_few_parameterized_statements():
    conn = FakeConnection()
    sink = DeltaSink(conn, batch_size=100)
    sink.write(events(250))
    merges = conn.merges()
    assert len(merges) == 3  # 100 + 100 + 50, never row by row
    assert [len(p) // len(COLUMNS) for _, p in merges] == [100, 100, 50]
    assert all("img" not in sql and "v1" not in sql for sql, _ in merges)  # values travel as parameters only
    assert len(conn.rows) == 250


def test_write_is_idempotent_by_event_id():
    conn = FakeConnection()
    sink = DeltaSink(conn)
    evs = events(30)
    sink.write(evs)
    sink.write(evs)
    sink.write(evs[10:] + events(5, start=30))  # overlap plus new rows
    assert sink.count() == 35 and sink.read().event_id.is_unique


def test_duplicate_ids_within_one_write_are_collapsed():
    conn = FakeConnection()
    sink = DeltaSink(conn)
    sink.write(events(3) + events(3))
    assert sink.count() == 3


def test_read_returns_canonical_frame_and_honours_limit():
    sink = DeltaSink(FakeConnection())
    evs = events(7)
    sink.write(list(reversed(evs)))
    pd.testing.assert_frame_equal(sink.read(), events_frame(evs))
    assert list(sink.read(limit=2).event_id) == ["e0", "e1"]
    assert sink.read().shape[1] == len(COLUMNS)


def test_empty_write_and_empty_read():
    conn = FakeConnection()
    sink = DeltaSink(conn)
    sink.write([])
    assert conn.merges() == [] and sink.read().empty and list(sink.read().columns) == COLUMNS


def test_table_name_is_validated_and_close_closes():
    with pytest.raises(ValueError):
        DeltaSink(FakeConnection(), table="x; DROP TABLE y")
    conn = FakeConnection()
    DeltaSink(conn).close()
    assert conn.closed


def test_load_log_uploads_local_parquet_idempotently(tmp_path):
    local = ParquetSink(str(tmp_path / "log"))
    local.write(events(120))
    sink = DeltaSink(FakeConnection())
    assert load_log(local, sink) == 120 and sink.count() == 120
    assert load_log(local, sink) == 120 and sink.count() == 120


def test_v1_log_without_statistics_loads_with_nan(tmp_path):
    old = events_frame(events(3)).drop(columns=["brightness_mean", "contrast_std", "sharpness", "noise_est",
                                                "schema_version"])
    (tmp_path / "log").mkdir()
    old.to_parquet(tmp_path / "log" / "part-00000.parquet", index=False)
    df = ParquetSink(str(tmp_path / "log")).read()
    assert (df.schema_version == 1).all() and df.sharpness.isna().all()
    sink = DeltaSink(FakeConnection())
    load_log(ParquetSink(str(tmp_path / "log")), sink)
    assert (sink.read().schema_version == 1).all()


def test_missing_variables_are_named_without_values(monkeypatch):
    for k in ("DATABRICKS_SERVER_HOSTNAME", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("src.inspection_log.load_dotenv", lambda path: None)
    assert missing_databricks_vars() == ["DATABRICKS_SERVER_HOSTNAME", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN"]
    with pytest.raises(RuntimeError, match="DATABRICKS_TOKEN"):
        connect_from_env()


def test_token_is_scrubbed_from_connection_errors(monkeypatch):
    secret = "dapi-SECRET-123"
    monkeypatch.setenv("DATABRICKS_SERVER_HOSTNAME", "https://example.cloud.databricks.com/")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/x")
    monkeypatch.setenv("DATABRICKS_TOKEN", secret)
    databricks_sql = pytest.importorskip("databricks.sql")

    def boom(**kw):
        raise ValueError(f"auth failed for {kw['access_token']}")

    monkeypatch.setattr(databricks_sql, "connect", boom)
    with pytest.raises(RuntimeError) as e:
        connect_from_env()
    assert secret not in str(e.value) and "***" in str(e.value)


@pytest.mark.requires_databricks
def test_live_roundtrip_and_idempotency():
    table = f"defect_inspection.test_events_{uuid.uuid4().hex[:10]}"
    sink = DeltaSink(table=table, batch_size=3)
    try:
        evs = events(5)
        sink.write(evs)
        sink.write(evs)
        assert sink.count() == 5
        pd.testing.assert_frame_equal(sink.read(), events_frame(evs), check_exact=False)
    finally:
        with sink.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        sink.close()
