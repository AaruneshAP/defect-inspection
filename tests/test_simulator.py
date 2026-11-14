"""Simulator tests with a stub detector (no weights needed); the real-weights test is marked requires_weights."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.service import InspectionService, Settings
from src.inspection_log import STAT_COLUMNS, ParquetSink, SQLiteSink, image_stats
from src.simulator import DriftSchedule, HttpBackend, ServiceBackend, SimConfig, list_images, simulate, stats_summary, summarize

SCRATCHES = 5


class RecordingDetector:
    """One detection whose confidence tracks image sharpness; records the pixels it was given."""

    def __init__(self):
        self.seen = []

    def detect(self, image):
        self.seen.append(np.asarray(image).copy())
        g = np.asarray(image.convert("L"), dtype=float)
        sharp = float(np.abs(np.diff(g, axis=1)).mean())
        return [((0, 0, 100, 100), min(0.99, 0.05 + sharp / 400), SCRATCHES)]


@pytest.fixture
def image_dir(tmp_path):
    d = tmp_path / "val"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(d / f"v{i}.png")
    return str(d)


def run(image_dir, sink, cfg, det=None):
    det = det or RecordingDetector()
    svc = InspectionService(Settings(model_path="does/not/exist.pt"), det, "sha256:stub")
    simulate(ServiceBackend(svc), image_dir, sink, cfg)
    return sink.read(), det


def test_events_are_labelled_simulated_and_complete(image_dir, tmp_path):
    df, _ = run(image_dir, ParquetSink(str(tmp_path / "o")), SimConfig(n_events=12, seed=1, rate_hz=2.0))
    assert len(df) == 12 and df.simulated.all() and df.event_id.is_unique
    assert set(df.model_version) == {"sha256:stub"} and set(df.perturbation) == {"none"}
    assert (df.n_detections == 1).all() and (df.latency_ms > 0).all()
    assert (df.timestamp.diff().dropna() == np.timedelta64(500, "ms")).all()


def test_same_seed_is_deterministic_and_different_seed_differs(image_dir, tmp_path):
    def cfg(s):
        return SimConfig(n_events=40, seed=s, drift=DriftSchedule(start=10, ramp=20))

    a, _ = run(image_dir, ParquetSink(str(tmp_path / "a")), cfg(7))
    b, _ = run(image_dir, SQLiteSink(str(tmp_path / "b.sqlite")), cfg(7))
    c, _ = run(image_dir, ParquetSink(str(tmp_path / "c")), cfg(8))
    keep = [x for x in a.columns if x != "latency_ms"]  # latency is measured, not seeded
    assert a[keep].equals(b[keep])
    assert not a[keep].equals(c[keep])


def test_drift_schedule_values():
    rng = np.random.default_rng(0)
    d = DriftSchedule(start=5, ramp=10, kinds=("blur",))
    assert d.at(4, rng) == ("none", 0)
    assert d.at(5, rng) == ("blur", 1)
    assert d.at(9, rng)[1] == 2 and d.at(15, rng) == ("blur", 3) and d.at(500, rng) == ("blur", 3)
    assert DriftSchedule(0, 4, ("contrast",)).at(100, rng) == ("contrast", 2)  # capped at that kind's top level
    with pytest.raises(ValueError):
        DriftSchedule(0, kinds=("sepia",))


def test_drift_applies_after_start_with_rising_severity(image_dir, tmp_path):
    cfg = SimConfig(n_events=60, seed=3, drift=DriftSchedule(start=20, ramp=30))
    df, det = run(image_dir, ParquetSink(str(tmp_path / "o")), cfg)
    before, after = df.iloc[:20], df.iloc[20:]
    assert (before.perturbation == "none").all() and (before.severity == 0).all()
    assert (after.perturbation != "none").all() and (after.severity >= 1).all()
    assert set(after.perturbation) <= {"blur", "noise", "contrast"} and len(set(after.perturbation)) > 1
    assert after.severity.iloc[:10].mean() < after.severity.iloc[-10:].mean()
    # the detector saw altered pixels exactly when severity > 0
    originals = {p.name: np.asarray(Image.open(p).convert("RGB")) for p in list_images(image_dir)}
    for row, seen in zip(df.itertuples(), det.seen):
        assert np.array_equal(seen, originals[row.image_id]) == (row.severity == 0)


def test_stats_are_computed_on_the_image_the_service_received(image_dir, tmp_path):
    cfg = SimConfig(n_events=30, seed=5, drift=DriftSchedule(start=10, ramp=10))
    df, det = run(image_dir, ParquetSink(str(tmp_path / "o")), cfg)
    assert (df.schema_version == 2).all() and df[STAT_COLUMNS].notna().all().all()
    for row, seen in zip(df.itertuples(), det.seen):
        for c, v in image_stats(seen).items():
            assert getattr(row, c) == pytest.approx(v)
    clean, blur = df[df.severity == 0], df[df.perturbation == "blur"]
    assert blur.sharpness.mean() < clean.sharpness.mean()
    assert stats_summary(df, 10).index[0].startswith("after") and "before" in stats_summary(df, 10).index


def test_no_drift_means_no_perturbation(image_dir, tmp_path):
    df, _ = run(image_dir, ParquetSink(str(tmp_path / "o")), SimConfig(n_events=15))
    assert (df.severity == 0).all()


def test_summarize_splits_before_and_after(image_dir, tmp_path):
    df, _ = run(image_dir, ParquetSink(str(tmp_path / "o")), SimConfig(n_events=30, drift=DriftSchedule(10, 10)))
    t = summarize(df, 10)
    assert list(t.index) == ["before", "after"]
    assert t.loc["before"].sum() == 10 and t.loc["after"].sum() == 20
    assert list(summarize(df, None).index) == ["before"]


def test_test_split_is_refused(tmp_path):
    (tmp_path / "test").mkdir()
    with pytest.raises(ValueError, match="test split"):
        list_images(str(tmp_path / "test"))


def test_api_path_matches_service_path(image_dir, tmp_path):
    cfg = SimConfig(n_events=15, seed=2, drift=DriftSchedule(start=5, ramp=5))
    direct, _ = run(image_dir, ParquetSink(str(tmp_path / "d")), cfg)
    app = create_app(Settings(model_path="does/not/exist.pt"), detector=RecordingDetector(), version="sha256:stub")
    with TestClient(app) as c:
        via_api = ParquetSink(str(tmp_path / "a"))
        simulate(HttpBackend(c), image_dir, via_api, cfg)
    api = via_api.read()
    keep = [x for x in direct.columns if x != "latency_ms"]
    assert api[keep].equals(direct[keep])
    assert (api.n_detections == 1).all()  # include_ignored=True keeps every detection in the counts


@pytest.mark.requires_weights
@pytest.mark.requires_data
def test_real_weights_small_run(tmp_path):
    svc = InspectionService(Settings.from_env())
    sink = ParquetSink(str(tmp_path / "o"))
    simulate(ServiceBackend(svc), "data/yolo/images/val", sink,
             SimConfig(n_events=6, seed=0, drift=DriftSchedule(start=3, ramp=3)))
    df = sink.read()
    assert len(df) == 6 and df.model_version.str.startswith("sha256:").all()
    assert (df.severity.iloc[:3] == 0).all() and (df.severity.iloc[3:] > 0).all()
