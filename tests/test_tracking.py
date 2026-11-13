"""Tests for src.tracking using a throwaway local MLflow file store."""

import math

import pytest

pytest.importorskip("mlflow")

from mlflow.tracking import MlflowClient  # noqa: E402

from src import tracking as T  # noqa: E402


@pytest.fixture
def uri(tmp_path):
    return (tmp_path / "mlruns").as_uri()


def test_flatten_nested():
    assert T.flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_clean_metrics_keeps_only_finite_numbers():
    m = T.clean_metrics({"x": 1, "y": {"z": 2.5}, "bad": math.nan, "inf": math.inf, "s": "a", "b": True})
    assert m == {"x": 1.0, "y.z": 2.5}


def test_git_commit_returns_hash_or_none():
    h = T.git_commit()
    assert h is None or len(h) == 40
    assert T.git_commit(cwd="/nonexistent-dir-xyz") is None


def test_log_run_records_params_metrics_tags_and_artifacts(uri, tmp_path):
    art = tmp_path / "res.json"
    art.write_text("{}")
    rid = T.log_run("r1", {"cfg": {"lr": 0.1}, "seed": 42}, {"mAP": 0.5, "pc": {"a": 0.2}}, [art],
                    {"git_commit": "abc"}, tracking_uri=uri)
    c = MlflowClient(uri)
    run = c.get_run(rid)
    assert run.data.params == {"cfg.lr": "0.1", "seed": "42"}
    assert run.data.metrics == {"mAP": 0.5, "pc.a": 0.2}
    assert run.data.tags["git_commit"] == "abc"
    assert [a.path for a in c.list_artifacts(rid)] == ["res.json"]


def test_log_run_is_idempotent_by_name(uri):
    a = T.log_run("same", {"x": 1}, tracking_uri=uri)
    b = T.log_run("same", {"x": 1}, tracking_uri=uri)
    c = T.log_run("same", {"x": 1}, tracking_uri=uri, skip_existing=False)
    assert a == b != c


def test_log_run_missing_artifact_raises(uri, tmp_path):
    with pytest.raises(FileNotFoundError):
        T.log_run("r", artifacts=[tmp_path / "nope"], tracking_uri=uri)


def test_register_model_sets_alias(uri, tmp_path):
    w = tmp_path / "best.pt"
    w.write_bytes(b"weights")
    rid = T.log_run("m", tracking_uri=uri)
    v = T.register_model(rid, w, "det", "production", uri)
    mv = MlflowClient(uri).get_model_version_by_alias("det", "production")
    assert str(mv.version) == v and mv.run_id == rid
    v2 = T.register_model(rid, w, "det", "production", uri)  # re-register: new version, alias moves
    assert v2 != v
    assert str(MlflowClient(uri).get_model_version_by_alias("det", "production").version) == v2


def test_register_model_missing_weights(uri, tmp_path):
    with pytest.raises(FileNotFoundError):
        T.register_model("x", tmp_path / "none.pt", "det", tracking_uri=uri)
