"""API tests with a stub detector (no weights needed) plus one real-weights test."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.service import Settings

CRAZING, SCRATCHES = 0, 5  # class ids; crazing band is (0.1, 0.25) in the frozen primary policy


class StubDetector:
    """Returns fixed detections regardless of the image."""

    def __init__(self, dets):
        self.dets = dets

    def detect(self, image):
        return self.dets


def png_bytes(size=(100, 100), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buf, format=fmt)
    return buf.getvalue()


def client(dets=(), **settings_kw):
    settings = Settings(model_path="does/not/exist.pt", **settings_kw)
    app = create_app(settings, detector=StubDetector(list(dets)), version="sha256:stub")
    return TestClient(app)


def post(c, content, name="a.png", ctype="image/png"):
    return c.post("/predict", files=[("files", (name, content, ctype))])


def test_health_and_model_info():
    with client() as c:
        assert c.get("/health").json() == {"status": "ok", "model_loaded": True}
        info = c.get("/model-info").json()
    assert info["model_name"] == "defect-detector" and info["model_version"] == "sha256:stub"
    assert info["policy"]["name"] == "primary" and info["policy"]["reviewer_accuracy"] == 0.9
    assert info["policy"]["bands"]["crazing"] == [0.1, 0.25]


def test_predict_applies_frozen_policy_per_detection_and_rolls_up():
    dets = [((0, 0, 100, 100), 0.9, SCRATCHES),   # confident, whole-image box -> stop_now
            ((0, 0, 10, 10), 0.2, CRAZING),       # inside crazing review band -> human_review
            ((0, 0, 10, 10), 0.05, CRAZING)]      # below review band -> ignore
    with client(dets) as c:
        r = post(c, png_bytes())
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "sha256:stub"
    res = body["results"][0]
    assert (res["filename"], res["width"], res["height"]) == ("a.png", 100, 100)
    assert [d["action"] for d in res["detections"]] == ["stop_now", "human_review"]
    assert res["detections"][0]["class_name"] == "scratches" and res["detections"][0]["box"] == [0, 0, 100, 100]
    assert res["decision"] == "stop_now"


def test_predict_include_ignored_flag():
    dets = [((0, 0, 100, 100), 0.9, SCRATCHES),   # stop_now
            ((0, 0, 10, 10), 0.2, CRAZING),       # human_review
            ((0, 0, 10, 10), 0.05, CRAZING)]      # ignore
    with client(dets) as c:
        r_default = c.post("/predict", files=[("files", ("a.png", png_bytes(), "image/png"))])
        res_default = r_default.json()["results"][0]
        assert [d["action"] for d in res_default["detections"]] == ["stop_now", "human_review"]

        r_inc = c.post("/predict?include_ignored=true", files=[("files", ("a.png", png_bytes(), "image/png"))])
        res_inc = r_inc.json()["results"][0]
        assert [d["action"] for d in res_inc["detections"]] == ["stop_now", "human_review", "ignore"]

        assert res_default["decision"] == res_inc["decision"] == "stop_now"


def test_predict_all_ignored_decision_identical():
    dets = [((0, 0, 10, 10), 0.05, CRAZING)]  # ignore
    with client(dets) as c:
        r_default = c.post("/predict", files=[("files", ("a.png", png_bytes(), "image/png"))])
        res_default = r_default.json()["results"][0]
        assert res_default["detections"] == []
        assert res_default["decision"] == "ignore"

        r_inc = c.post("/predict?include_ignored=true", files=[("files", ("a.png", png_bytes(), "image/png"))])
        res_inc = r_inc.json()["results"][0]
        assert len(res_inc["detections"]) == 1
        assert res_inc["detections"][0]["action"] == "ignore"
        assert res_inc["decision"] == res_default["decision"] == "ignore"


def test_predict_no_detections_is_ignore():
    with client([]) as c:
        res = post(c, png_bytes()).json()["results"][0]
    assert res["detections"] == [] and res["decision"] == "ignore"


def test_predict_multiple_files_and_jpeg():
    with client([]) as c:
        r = c.post("/predict", files=[("files", ("a.png", png_bytes(), "image/png")),
                                      ("files", ("b.jpg", png_bytes(fmt="JPEG"), "image/jpeg"))])
    assert r.status_code == 200 and [x["filename"] for x in r.json()["results"]] == ["a.png", "b.jpg"]


def test_bad_content_type_is_415():
    with client() as c:
        r = post(c, b"hello", "a.txt", "text/plain")
    assert r.status_code == 415 and "not allowed" in r.json()["detail"]


def test_undecodable_bytes_with_image_content_type_is_400():
    with client() as c:
        r = post(c, b"not an image at all")
    assert r.status_code == 400 and "decodable" in r.json()["detail"]


def test_disallowed_image_format_is_400():
    with client() as c:
        r = post(c, png_bytes(fmt="GIF"), "a.png", "image/png")
    assert r.status_code == 400 and "unsupported image format" in r.json()["detail"]


def test_empty_file_is_400():
    with client() as c:
        assert post(c, b"").status_code == 400


def test_oversize_is_413():
    with client(max_upload_bytes=100) as c:
        assert post(c, png_bytes((200, 200))).status_code == 413


def test_missing_files_field_is_422():
    with client() as c:
        assert c.post("/predict").status_code == 422


def test_too_many_files_is_400():
    with client(max_files=1) as c:
        r = c.post("/predict", files=[("files", ("a.png", png_bytes(), "image/png"))] * 2)
    assert r.status_code == 400


def test_missing_weights_health_ok_predict_503():
    app = create_app(Settings(model_path="does/not/exist.pt"))
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok", "model_loaded": False}
        info = c.get("/model-info").json()
        assert info["model_loaded"] is False and info["model_version"] is None
        assert post(c, png_bytes()).status_code == 503


def test_model_path_read_from_environment(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "somewhere/else.pt")
    assert Settings.from_env().model_path == "somewhere/else.pt"


@pytest.mark.requires_weights
def test_real_weights_end_to_end():
    with TestClient(create_app(Settings(model_path="models/best.pt"))) as c:
        assert c.get("/health").json()["model_loaded"] is True
        assert c.get("/model-info").json()["model_version"].startswith("sha256:")
        imgs = sorted(Path("data/yolo/images/val").glob("*.jpg"))[:1] if Path("data/yolo/images/val").exists() else []
        content = imgs[0].read_bytes() if imgs else png_bytes((200, 200), "JPEG")
        r = post(c, content, "x.jpg", "image/jpeg")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"].startswith("sha256:")
    assert body["results"][0]["decision"] in {"stop_now", "schedule_service", "human_review", "monitor", "ignore"}
