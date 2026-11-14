"""ONNX Runtime detector with the same interface and post-processing as the Ultralytics torch path (CPU).

Letterbox (centred, pad 114, linear resize), YOLOv8 output decoding, class-wise NMS at IoU 0.5, confidence floor 0.001
and max 300 detections match `ultralytics` predict defaults. Pure numpy/cv2 helpers are separate from the session so they
can be tested without a model. Fixed 640x640 input: non-square images are padded to a square, whereas the torch path
pads to the nearest multiple of 32; NEU-DET images are square, so the two agree there.

  python -m src.onnx_detector export            # models/best.pt -> models/best.onnx
  python -m src.onnx_detector parity            # torch vs ONNX on every val image -> reports/phase7/onnx_parity.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from src.predict import CONF_FLOOR

IOU_THRES = 0.5  # Ultralytics 8.4 predict default (val uses 0.7)
MAX_DET = 300
MAX_NMS = 30000
PAD_VALUE = 114
Det = Tuple[Tuple[float, float, float, float], float, int]


def letterbox(img: np.ndarray, size: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize an RGB uint8 (H, W, 3) image into size x size, centred, padded with 114.

    Returns (padded image, scale ratio, (pad_left, pad_top)).
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom, left, right = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return out, r, (left, top)


def to_tensor(padded: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (1, 3, H, W) float32 in [0, 1]."""
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes, shape (len(a), len(b))."""
    a, b = np.atleast_2d(a).astype(np.float64), np.atleast_2d(b).astype(np.float64)
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thres: float = IOU_THRES) -> np.ndarray:
    """Class-wise greedy NMS (a box is dropped when IoU > iou_thres with a higher-scoring kept box of the same class).

    Returns kept indices, highest score first.
    """
    keep: List[int] = []
    order = np.argsort(-scores, kind="stable")
    for c in np.unique(classes):
        idx = order[classes[order] == c]
        while len(idx):
            i = idx[0]
            keep.append(int(i))
            if len(idx) == 1:
                break
            idx = idx[1:][box_iou(boxes[i:i + 1], boxes[idx[1:]])[0] <= iou_thres]
    keep.sort(key=lambda i: -scores[i])
    return np.array(keep, dtype=int)


def decode(output: np.ndarray, ratio: float, pad: Tuple[int, int], orig_hw: Tuple[int, int], conf_thres: float = CONF_FLOOR,
           iou_thres: float = IOU_THRES, max_det: int = MAX_DET) -> List[Det]:
    """YOLOv8 raw output (1, 4 + nc, N) with xywh boxes in letterbox pixels -> [(xyxy in original pixels, conf, class)]."""
    pred = np.asarray(output)[0].T  # (N, 4 + nc)
    scores_all = pred[:, 4:]
    conf, cls = scores_all.max(1), scores_all.argmax(1)
    keep = conf > conf_thres
    pred, conf, cls = pred[keep], conf[keep], cls[keep]
    if not len(pred):
        return []
    if len(pred) > MAX_NMS:
        top = np.argsort(-conf)[:MAX_NMS]
        pred, conf, cls = pred[top], conf[top], cls[top]
    xywh = pred[:, :4]
    xyxy = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], axis=1)
    kept = nms(xyxy, conf, cls, iou_thres)[:max_det]
    h, w = orig_hw
    boxes = xyxy[kept].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return [(tuple(float(v) for v in b), float(c), int(k)) for b, c, k in zip(boxes, conf[kept], cls[kept])]


class OnnxDetector:
    """Detector backend over onnxruntime (CPU); same `detect(image)` contract as app.service.YoloDetector."""

    def __init__(self, path: str, imgsz: int = 640, conf: float = CONF_FLOOR, iou: float = IOU_THRES, session=None):
        if session is None:
            import onnxruntime as ort

            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.session, self.imgsz, self.conf, self.iou = session, imgsz, conf, iou
        self.input_name = session.get_inputs()[0].name

    def detect(self, image) -> List[Det]:
        """PIL image -> [(xyxy box, confidence, class_id)] in original-image pixels."""
        rgb = np.asarray(image.convert("RGB"))
        padded, ratio, pad = letterbox(rgb, self.imgsz)
        out = self.session.run(None, {self.input_name: to_tensor(padded)})[0]
        return decode(out, ratio, pad, rgb.shape[:2], self.conf, self.iou)


# ---- parity ---------------------------------------------------------------------------------------------------------

def compare_detections(a: Sequence[Det], b: Sequence[Det], iou_match: float = 0.9) -> Dict[str, object]:
    """Compare two detection lists for one image.

    Each detection in `a` is matched greedily (highest confidence first) to the unused same-class detection in `b`
    with the highest IoU, counted when IoU >= iou_match.
    """
    matched_iou, conf_diff = [], []
    if len(a) and len(b):
        iou = box_iou(np.array([d[0] for d in a]), np.array([d[0] for d in b]))
        iou[np.array([d[2] for d in a])[:, None] != np.array([d[2] for d in b])[None, :]] = 0.0
        used = np.zeros(len(b), bool)
        for i in np.argsort([-d[1] for d in a], kind="stable"):
            row = np.where(used, 0.0, iou[i])
            j = int(row.argmax())
            if row[j] >= iou_match:
                used[j] = True
                matched_iou.append(float(row[j]))
                conf_diff.append(abs(a[i][1] - b[j][1]))
    return {"n_a": len(a), "n_b": len(b), "matched": len(matched_iou), "unmatched_a": len(a) - len(matched_iou),
            "unmatched_b": len(b) - len(matched_iou), "iou": matched_iou, "conf_abs_diff": conf_diff}


def summarize_parity(per_image: Sequence[Dict[str, object]], above: float = 0.25) -> Dict[str, object]:
    """Aggregate compare_detections results over images (count agreement, IoU and confidence differences)."""
    ious = [v for p in per_image for v in p["iou"]]
    diffs = [v for p in per_image for v in p["conf_abs_diff"]]
    q = (lambda x, p: float(np.percentile(x, p)) if len(x) else None)
    return {"n_images": len(per_image),
            "images_equal_count": sum(p["n_a"] == p["n_b"] for p in per_image),
            "images_all_matched": sum(p["unmatched_a"] == 0 and p["unmatched_b"] == 0 for p in per_image),
            "detections_a": sum(p["n_a"] for p in per_image), "detections_b": sum(p["n_b"] for p in per_image),
            "matched": len(ious), "unmatched_a": sum(p["unmatched_a"] for p in per_image),
            "unmatched_b": sum(p["unmatched_b"] for p in per_image),
            "iou_min": min(ious) if ious else None, "iou_median": q(ious, 50),
            "conf_abs_diff_median": q(diffs, 50), "conf_abs_diff_p99": q(diffs, 99),
            "conf_abs_diff_max": max(diffs) if diffs else None}


def _above(dets: Sequence[Det], thr: float) -> List[Det]:
    return [d for d in dets if d[1] >= thr]


def run_parity(model_pt: str, model_onnx: str, image_dir: str, out: str, thresholds=(CONF_FLOOR, 0.127, 0.25)) -> dict:
    """Torch vs ONNX on every image in a val directory: agreement at several confidence cut-offs, plus CPU latency."""
    from PIL import Image

    from app.service import YoloDetector
    from src.simulator import list_images  # refuses a 'test' directory

    paths = list_images(image_dir)
    torch_det, onnx_det = YoloDetector(model_pt), OnnxDetector(model_onnx)
    im0 = Image.open(paths[0]).convert("RGB")
    for _ in range(3):  # warm-up
        torch_det.detect(im0)
        onnx_det.detect(im0)
    pairs, t_torch, t_onnx = [], [], []
    for p in paths:
        with Image.open(p) as im:
            img = im.convert("RGB")
        t0 = time.perf_counter()
        a = torch_det.detect(img)
        t1 = time.perf_counter()
        b = onnx_det.detect(img)
        t2 = time.perf_counter()
        t_torch.append((t1 - t0) * 1000)
        t_onnx.append((t2 - t1) * 1000)
        pairs.append((a, b))
    result = {"n_images": len(paths), "image_dir": image_dir, "split": "val",
              "latency_ms": {"torch_median": statistics.median(t_torch), "torch_p95": float(np.percentile(t_torch, 95)),
                             "onnx_median": statistics.median(t_onnx), "onnx_p95": float(np.percentile(t_onnx, 95))},
              "by_conf_cutoff": {}}
    for thr in thresholds:
        result["by_conf_cutoff"][str(thr)] = summarize_parity(
            [compare_detections(_above(a, thr), _above(b, thr)) for a, b in pairs])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--weights", default="models/best.pt")
    e.add_argument("--imgsz", type=int, default=640)
    p = sub.add_parser("parity")
    p.add_argument("--weights", default="models/best.pt")
    p.add_argument("--onnx", default="models/best.onnx")
    p.add_argument("--image-dir", default="data/yolo/images/val")
    p.add_argument("--out", default="reports/phase7/onnx_parity.json")
    a = ap.parse_args()
    if a.cmd == "export":
        from ultralytics import YOLO

        print(YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, device="cpu", simplify=False, dynamic=False, opset=17))
    else:
        r = run_parity(a.weights, a.onnx, a.image_dir, a.out)
        print(json.dumps({"latency_ms": r["latency_ms"], **{k: v for k, v in r["by_conf_cutoff"].items()}}, indent=1))


if __name__ == "__main__":
    main()

