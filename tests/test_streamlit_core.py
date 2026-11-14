"""Hosted-demo helpers with stubs: asset fetching and fallbacks, service wiring, and (if installed) the pages."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.service import InspectionService, Settings
from streamlit_app import core

# class ids: 0 crazing, 1 inclusion, 2 patches, 3 pitted_surface, 4 rolled-in_scale, 5 scratches
BIG_PATCH = ((10.0, 10.0, 90.0, 90.0), 0.9, 2)  # large box, confident -> alert-level action
LOW_SCRATCH = ((0.0, 0.0, 20.0, 20.0), 0.01, 5)  # far below every band -> ignore


class Stub:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, image):
        return list(self.dets)


def service(dets):
    return InspectionService(Settings(), Stub(dets), "stub")


def img(size=(100, 100)):
    return Image.fromarray(np.full(size[::-1] + (3,), 128, np.uint8))


class StubDetector:
    def __init__(self, *a, **k):
        pass

    def detect(self, image):
        return [((10.0, 10.0, 90.0, 90.0), 0.9, 2)]


def fake_download(files):
    def download(repo, name):
        if name not in files:
            raise FileNotFoundError(name)
        return f"/cache/{repo}/{name}"

    return download


def test_fetch_assets_gets_model_and_up_to_six_examples(monkeypatch):
    monkeypatch.delenv("ONNX_PATH", raising=False)
    monkeypatch.delenv("EXAMPLES_DIR", raising=False)
    names = [f"examples/c{i}.jpg" for i in range(8)] + ["README.md", "examples/notes.txt", "best.onnx"]
    a = core.fetch_assets("u/r", fake_download(set(names)), lambda repo: names)
    assert a.onnx_path == "/cache/u/r/best.onnx" and a.error is None and a.example_note is None
    assert len(a.examples) == 6 and all("examples/c" in p for p in a.examples)


def test_model_download_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.delenv("ONNX_PATH", raising=False)
    a = core.fetch_assets("u/r", fake_download(set()), lambda repo: [])
    assert a.onnx_path is None and "Could not download the model" in a.error and "FileNotFoundError" in a.error


def test_missing_examples_fall_back_to_upload_only(monkeypatch):
    monkeypatch.delenv("ONNX_PATH", raising=False)
    monkeypatch.delenv("EXAMPLES_DIR", raising=False)
    none = core.fetch_assets("u/r", fake_download({"best.onnx"}), lambda repo: ["best.onnx"])
    assert none.onnx_path and none.examples == [] and "upload your own" in none.example_note
    boom = core.fetch_assets("u/r", fake_download({"best.onnx"}), lambda repo: (_ for _ in ()).throw(OSError("down")))
    assert boom.onnx_path and boom.examples == [] and "OSError" in boom.example_note


def test_env_overrides_skip_the_download(monkeypatch, tmp_path):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"x")
    ex = tmp_path / "val"
    ex.mkdir()
    (ex / "a_1.jpg").write_bytes(b"x")
    (ex / "readme.txt").write_bytes(b"x")
    monkeypatch.setenv("ONNX_PATH", str(onnx))
    monkeypatch.setenv("EXAMPLES_DIR", str(ex))

    def never(*a):
        raise AssertionError("must not download")

    a = core.fetch_assets("u/r", never, never)
    assert a.onnx_path == str(onnx) and [Path(p).name for p in a.examples] == ["a_1.jpg"]


def test_examples_dir_refuses_the_test_split(tmp_path):
    t = tmp_path / "test"
    t.mkdir()
    with pytest.raises(ValueError):
        core.local_examples(str(t))
    assert core.local_examples(str(tmp_path / "missing")) == []


def test_build_service_uses_the_onnx_backend_and_frozen_policy(monkeypatch):
    monkeypatch.setattr(core, "OnnxDetector", StubDetector)
    svc = core.build_service("models/best.onnx")
    assert svc.loaded and svc.settings.backend == "onnx" and svc.version == "onnx:best.onnx"
    assert svc.policy.name == "primary" and svc.bands["crazing"] == (0.1, 0.25)
    out, rows, text = core.analyze(Image.fromarray(np.zeros((100, 100, 3), np.uint8)), False, svc)
    assert rows and out.size == (100, 100) and "Image decision" in text


def test_analyze_returns_annotated_image_rows_and_decision():
    out, rows, text = core.analyze(img(), False, service([BIG_PATCH, LOW_SCRATCH]))
    assert out.size == (100, 100) and out.getpixel((10, 50)) != (128, 128, 128)  # box edge was drawn
    assert [r[0] for r in rows] == ["patches"] and rows[0][3] in {"stop_now", "schedule_service", "monitor"}
    assert f"`{rows[0][3]}`" in text and "hidden" in text


def test_ignored_detections_are_hidden_unless_requested():
    s = service([BIG_PATCH, LOW_SCRATCH])
    _, rows, text = core.analyze(img(), True, s)
    assert [r[0] for r in rows] == ["patches", "scratches"] and rows[1][3] == "ignore" and "hidden" not in text
    _, rows, _ = core.analyze(img(), False, s)
    assert len(rows) == 1


def test_no_detections_gives_decision_ignore():
    out, rows, text = core.analyze(img(), False, service([]))
    assert rows == [] and "`ignore`" in text and out.size == (100, 100)


def test_missing_image_or_model_is_reported_not_raised():
    assert core.analyze(None, False, service([]))[2].startswith("Upload")
    no_model = InspectionService(Settings(model_path="does/not/exist.pt"))
    assert not no_model.loaded and "not loaded" in core.analyze(img(), False, no_model)[2]


def test_large_images_are_downscaled_and_grayscale_is_accepted():
    big = Image.fromarray(np.zeros((3200, 2000), np.uint8))
    assert max(core.prepare(big).size) == core.MAX_SIDE and core.prepare(big).mode == "RGB"
    out, _, _ = core.analyze(big, False, service([]))
    assert max(out.size) == core.MAX_SIDE


def test_pick_examples_one_per_class_from_val_only(tmp_path):
    d = tmp_path / "val"
    d.mkdir()
    for name in ("scratches_2", "crazing_9", "crazing_1", "patches_5", "inclusion_3", "pitted_surface_1",
                 "rolled-in_scale_4", "patches_1", "notes"):
        (d / f"{name}.jpg" if name != "notes" else d / "notes.txt").write_bytes(b"x")
    picks = core.pick_examples(str(d), 6)
    assert [p.stem for p in picks] == ["crazing_1", "inclusion_3", "patches_1", "pitted_surface_1", "rolled-in_scale_4",
                                       "scratches_2"]
    assert len(core.pick_examples(str(d), 3)) == 3
    t = tmp_path / "test"
    t.mkdir()
    with pytest.raises(ValueError):
        core.pick_examples(str(t))


def test_banners_state_demo_and_simulated_data():
    assert "Portfolio demo" in core.BANNER and "NEU-DET" in core.BANNER
    assert "SIMULATED" in core.SIM_BANNER


def test_requirements_are_light_and_pinned():
    lines = [ln for ln in Path("streamlit_app/requirements.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    names = {ln.split("==")[0].lower() for ln in lines}
    assert all("==" in ln for ln in lines)
    assert not names & {"torch", "torchvision", "ultralytics", "mlflow"}
    assert {"streamlit", "plotly", "pandas", "pyarrow", "numpy", "pillow", "onnxruntime",
            "opencv-python-headless", "huggingface_hub"} <= names


@pytest.mark.skipif(not Path("models/best.onnx").exists(), reason="models/best.onnx not exported")
def test_pages_render_with_the_onnx_backend(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    pytest.importorskip("onnxruntime")
    from streamlit.testing.v1 import AppTest

    ex = tmp_path / "val"
    ex.mkdir()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)).save(ex / "crazing_1.jpg")
    monkeypatch.setenv("ONNX_PATH", str(Path("models/best.onnx").resolve()))
    monkeypatch.setenv("EXAMPLES_DIR", str(ex))
    at = AppTest.from_file(str(Path("streamlit_app/try_it.py").resolve()), default_timeout=120).run()
    assert not at.exception and at.selectbox[0].options == ["(none)", "crazing_1"]
    at.selectbox[0].select("crazing_1").run()
    assert not at.exception and any("Image decision" in m.value for m in at.markdown)
    mon = AppTest.from_file(str(Path("streamlit_app/monitoring.py").resolve()), default_timeout=120).run()
    assert not mon.exception and any("SIMULATED DATA" in e.value for e in mon.error)
