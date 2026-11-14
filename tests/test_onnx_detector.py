"""ONNX detector: letterbox, decode, NMS and parity logic on synthetic data; real parity needs weights + data."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.service import InspectionService, Settings
from src.onnx_detector import (OnnxDetector, box_iou, compare_detections, decode, letterbox, nms, summarize_parity,
                               to_tensor)


def raw(rows, nc=3, n=20):
    """YOLOv8-style output (1, 4 + nc, n): rows = [(cx, cy, w, h, class, score)], rest zero."""
    out = np.zeros((1, 4 + nc, n), np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(rows):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + c, i] = s
    return out


def test_letterbox_square_input_is_scaled_without_padding():
    img = np.full((200, 200, 3), 50, np.uint8)
    out, r, pad = letterbox(img, 640)
    assert out.shape == (640, 640, 3) and r == 3.2 and pad == (0, 0) and (out == 50).all()


def test_letterbox_centres_and_pads_with_114():
    img = np.zeros((100, 200, 3), np.uint8)  # wide: ratio 3.2, new 640x320, 160 px padding top and bottom
    out, r, pad = letterbox(img, 640)
    assert r == 3.2 and pad == (0, 160) and out.shape == (640, 640, 3)
    assert (out[:160] == 114).all() and (out[480:] == 114).all() and (out[160:480] == 0).all()


def test_letterbox_odd_padding_splits_like_ultralytics():
    img = np.zeros((99, 200, 3), np.uint8)
    out, _, pad = letterbox(img, 640)  # new_h = round(316.8) = 317, dh = 161.5 -> top 161, bottom 162
    assert pad == (0, 161) and out.shape == (640, 640, 3) and (out[:161] == 114).all() and (out[479:] == 114).all()


def test_to_tensor_layout_and_range():
    t = to_tensor(np.full((4, 4, 3), 255, np.uint8))
    assert t.shape == (1, 3, 4, 4) and t.dtype == np.float32 and t.max() == 1.0


def test_box_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert np.allclose(box_iou(a, b), [[1.0, 1 / 3, 0.0]])


def test_nms_is_class_wise_and_uses_strict_threshold():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10], [5, 0, 15, 10]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    classes = np.array([0, 0, 1, 0])
    keep = nms(boxes, scores, classes, 0.5)
    assert keep.tolist() == [0, 2, 3]  # dup of box 0 dropped; other class kept; IoU 1/3 <= 0.5 kept
    assert nms(boxes, scores, classes, 0.3).tolist() == [0, 2]  # IoU 1/3 > 0.3 now suppresses box 3
    assert nms(boxes[:1], scores[:1], classes[:1]).tolist() == [0]


def test_decode_maps_boxes_back_to_the_original_image():
    out = raw([(320, 320, 200, 100, 1, 0.8)])  # centred box in a 640 letterbox of a 200x200 image (ratio 3.2)
    dets = decode(out, 3.2, (0, 0), (200, 200))
    (box, conf, cls), = dets
    assert cls == 1 and conf == pytest.approx(0.8) and box == pytest.approx((68.75, 84.375, 131.25, 115.625))


def test_decode_removes_padding_clips_and_applies_floor_and_nms():
    out = raw([(320, 320, 700, 100, 0, 0.9),      # wider than the image -> clipped
               (320, 322, 700, 100, 0, 0.8),      # near-duplicate, same class -> suppressed
               (100, 100, 20, 20, 2, 0.0005)])    # below the 0.001 floor -> dropped
    dets = decode(out, 3.2, (0, 160), (100, 200))
    assert len(dets) == 1
    box = dets[0][0]
    assert box[0] == 0.0 and box[2] == 200.0 and box[1] == pytest.approx((270 - 160) / 3.2)
    assert decode(raw([]), 3.2, (0, 0), (200, 200)) == []


def test_decode_limits_detections():
    rows = [(10 + 30 * i, 50, 10, 10, 0, 0.5) for i in range(15)]
    assert len(decode(raw(rows, n=40), 1.0, (0, 0), (1000, 100), max_det=5)) == 5


class FakeSession:
    def __init__(self, out):
        self.out, self.seen = out, None

    def get_inputs(self):
        return [type("I", (), {"name": "images"})()]

    def run(self, _, feed):
        self.seen = feed["images"]
        return [self.out]


def test_onnx_detector_runs_letterbox_then_decode():
    s = FakeSession(raw([(320, 320, 200, 100, 1, 0.8)]))
    det = OnnxDetector("unused.onnx", session=s)
    dets = det.detect(Image.fromarray(np.zeros((200, 200, 3), np.uint8)))
    assert s.seen.shape == (1, 3, 640, 640) and dets[0][2] == 1 and dets[0][0][0] == pytest.approx(68.75)


def test_service_accepts_either_backend_and_validates():
    assert Settings().backend == "torch"
    with pytest.raises(ValueError):
        InspectionService(Settings(backend="tensorrt"))
    svc = InspectionService(Settings(backend="onnx", onnx_path="does/not/exist.onnx"))
    assert not svc.loaded  # missing weights: not loaded, no crash
    stub = type("S", (), {"detect": lambda self, im: [((0.0, 0.0, 50.0, 50.0), 0.9, 2)]})()
    assert InspectionService(Settings(backend="onnx"), stub, "v").run(Image.new("RGB", (100, 100)), "x").decision


def test_settings_from_env_reads_backend(monkeypatch):
    monkeypatch.setenv("DETECTOR_BACKEND", "onnx")
    monkeypatch.setenv("ONNX_PATH", "m.onnx")
    s = Settings.from_env()
    assert (s.backend, s.onnx_path) == ("onnx", "m.onnx")


def det(x, c=0.5, k=0, w=10):
    return ((x, 0.0, x + w, 10.0), c, k)


def test_compare_detections_matches_by_class_and_iou():
    a = [det(0, 0.9), det(50, 0.8, 1), det(100, 0.7)]
    b = [det(0, 0.9001), det(50, 0.8, 2), det(100.2, 0.65)]  # class differs for the 2nd; 3rd shifted slightly
    r = compare_detections(a, b)
    assert (r["n_a"], r["n_b"], r["matched"], r["unmatched_a"], r["unmatched_b"]) == (3, 3, 2, 1, 1)
    assert r["conf_abs_diff"] == pytest.approx([0.0001, 0.05]) and min(r["iou"]) > 0.9
    assert compare_detections([], [det(0)])["unmatched_b"] == 1
    assert compare_detections([det(0)], [])["unmatched_a"] == 1
    assert compare_detections([det(0)], [det(5)])["matched"] == 0  # IoU 1/3 < 0.9


def test_summarize_parity_counts_agreement():
    ok = compare_detections([det(0)], [det(0)])
    bad = compare_detections([det(0)], [det(0), det(40)])
    s = summarize_parity([ok, bad, compare_detections([], [])])
    assert (s["n_images"], s["images_equal_count"], s["images_all_matched"]) == (3, 2, 2)
    assert (s["detections_a"], s["detections_b"], s["matched"], s["unmatched_b"]) == (2, 3, 2, 1)
    assert s["iou_min"] == pytest.approx(1.0) and s["conf_abs_diff_max"] == 0.0
    assert summarize_parity([])["iou_min"] is None


@pytest.mark.requires_weights
@pytest.mark.requires_data
@pytest.mark.skipif(not Path("models/best.onnx").exists(), reason="models/best.onnx not exported")
def test_parity_with_torch_on_val_images():
    from app.service import YoloDetector
    from src.simulator import list_images

    t, o = YoloDetector("models/best.pt"), OnnxDetector("models/best.onnx")
    results = []
    for p in list_images("data/yolo/images/val")[:36]:
        im = Image.open(p).convert("RGB")
        a = [d for d in t.detect(im) if d[1] >= 0.127]
        b = [d for d in o.detect(im) if d[1] >= 0.127]
        results.append(compare_detections(a, b))
    s = summarize_parity(results)
    assert s["n_images"] == 36 and s["images_equal_count"] == 36 and s["unmatched_a"] == s["unmatched_b"] == 0
    assert s["iou_min"] > 0.99 and s["conf_abs_diff_max"] < 1e-3
