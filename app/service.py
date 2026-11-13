"""Model loading, frozen-policy loading and per-image decision logic (decision rules live in src.decision)."""

import hashlib
import io
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

from PIL import Image

from app.schemas import Detection, ImageResult, ModelInfo, PolicyInfo
from src.data import CLASS_NAMES
from src.decision import decide_action, load_cost_config, severity
from src.predict import CONF_FLOOR

MODEL_NAME = "defect-detector"
# Most urgent first. Serving-layer choice for the per-image roll-up; src.decision only maps single detections.
ACTION_PRIORITY = ["stop_now", "schedule_service", "human_review", "monitor", "ignore"]
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp"}


class Detector(Protocol):
    """Anything that turns a PIL image into raw (box, confidence, class_id) triples."""

    def detect(self, image: Image.Image) -> List[Tuple[Tuple[float, float, float, float], float, int]]: ...


class YoloDetector:
    """Ultralytics YOLO wrapper (CPU); the model is loaded once and inference is serialised."""

    def __init__(self, path: str, imgsz: int = 640):
        from ultralytics import YOLO

        self.model, self.imgsz, self._lock = YOLO(path), imgsz, threading.Lock()

    def detect(self, image):
        """Run inference at the same confidence floor used to build the Phase 3 caches."""
        with self._lock:
            res = self.model.predict(image, conf=CONF_FLOOR, imgsz=self.imgsz, device="cpu", verbose=False)[0]
        return [(tuple(b), float(c), int(k)) for b, c, k in
                zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())]


@dataclass
class Settings:
    """Runtime configuration (environment variables with repo-relative defaults)."""

    model_path: str = "models/best.pt"
    frozen_path: str = "reports/phase3/frozen.json"
    cost_path: str = "configs/cost.yaml"
    policy_key: str = "primary"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_files: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from MODEL_PATH, FROZEN_PATH, COST_CONFIG, POLICY_KEY, MAX_UPLOAD_BYTES, MAX_FILES."""
        e, d = os.environ.get, cls()
        return cls(e("MODEL_PATH", d.model_path), e("FROZEN_PATH", d.frozen_path), e("COST_CONFIG", d.cost_path),
                   e("POLICY_KEY", d.policy_key), int(e("MAX_UPLOAD_BYTES", d.max_upload_bytes)),
                   int(e("MAX_FILES", d.max_files)))


def file_sha256(path: str) -> str:
    """Full sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def registry_label() -> Optional[str]:
    """Registry alias/version if mlflow is installed and MLFLOW_TRACKING_URI is set; else None."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    try:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        from mlflow.tracking import MlflowClient

        alias = os.environ.get("MODEL_ALIAS", "production")
        mv = MlflowClient(uri).get_model_version_by_alias(MODEL_NAME, alias)
        return f"{MODEL_NAME}@{alias} (v{mv.version})"
    except Exception:
        return None


class InspectionService:
    """Holds the detector, frozen policy and cost config; created once at startup."""

    def __init__(self, settings: Settings, detector: Optional[Detector] = None, version: Optional[str] = None):
        self.settings = settings
        self.cost = load_cost_config(settings.cost_path)
        frozen = json.loads(Path(settings.frozen_path).read_text(encoding="utf-8"))
        entry = frozen[settings.policy_key]
        self.bands: Dict[str, Tuple[float, float]] = {n: (float(b[0]), float(b[1])) for n, b in entry["bands"].items()}
        missing = set(CLASS_NAMES) - set(self.bands)
        if missing:
            raise ValueError(f"frozen policy lacks bands for {sorted(missing)}")
        self.policy = PolicyInfo(name=settings.policy_key, reviewer_accuracy=entry["reviewer_accuracy"],
                                 budget=entry["budget"], bands=self.bands, source=settings.frozen_path)
        self.detector, self.version, self.registry = detector, version, None
        if detector is None and Path(settings.model_path).is_file():
            self.version = f"sha256:{file_sha256(settings.model_path)[:12]}"
            self.registry = registry_label()
            self.detector = YoloDetector(settings.model_path)

    @property
    def loaded(self) -> bool:
        """True when a detector is available."""
        return self.detector is not None

    def info(self) -> ModelInfo:
        """Model identity plus policy."""
        return ModelInfo(model_name=MODEL_NAME, model_version=self.version, registry=self.registry,
                         model_loaded=self.loaded, policy=self.policy)

    def run(self, image: Image.Image, filename: Optional[str], include_ignored: bool = False) -> ImageResult:
        """Detect on one image and apply the frozen policy to every detection."""
        w, h = image.size
        dets: List[Detection] = []
        for box, conf, cid in self.detector.detect(image):
            name = CLASS_NAMES[cid]
            lo, hi = self.bands[name]
            sev = severity(name, box, w, h, self.cost)
            dets.append(Detection(class_id=cid, class_name=name, confidence=conf, box=box, severity=sev,
                                  action=decide_action(conf, sev, lo, hi, self.cost)))
        dets.sort(key=lambda d: -d.confidence)
        decision = min((d.action for d in dets), key=ACTION_PRIORITY.index, default="ignore")
        if not include_ignored:
            dets = [d for d in dets if d.action != "ignore"]
        return ImageResult(filename=filename, width=w, height=h, detections=dets, decision=decision)


def decode_image(data: bytes) -> Image.Image:
    """Decode bytes into an RGB image; ValueError if not a JPEG/PNG/BMP."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.load()
            rgb = im.convert("RGB")
    except Exception as exc:
        raise ValueError("file is not a decodable image") from exc
    if fmt not in ALLOWED_FORMATS:
        raise ValueError(f"unsupported image format {fmt}; allowed: JPEG, PNG, BMP")
    return rgb
