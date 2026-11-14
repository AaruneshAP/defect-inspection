"""Non-UI logic for the hosted demo: fetch the ONNX model (and examples) from the HF model repo, build the service.

No streamlit import here, so it is testable with stubs. ONNX_PATH / EXAMPLES_DIR env vars override the download
(used for local runs).
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import ImageResult  # noqa: E402
from app.service import InspectionService, Settings  # noqa: E402
from src.onnx_detector import OnnxDetector  # noqa: E402

MODEL_REPO = os.environ.get("MODEL_REPO", "Aarun-1/defect-detector-neu-det")
ONNX_FILE = "best.onnx"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
BANNER = ("**Portfolio demo on a public research dataset (NEU-DET steel surface defects).** Not validated on real "
          "production data; the maintenance actions use illustrative cost assumptions.")
SIM_BANNER = ("**SIMULATED DATA.** The monitoring page shows synthetic traffic: val images replayed through the "
              "model with drift injected on purpose. It is a snapshot, not live production data.")
ACTION_COLORS = {
    "stop_now": "#d62728",
    "schedule_service": "#ff7f0e",
    "human_review": "#1f77b4",
    "monitor": "#2ca02c",
    "ignore": "#7f7f7f",
}
TABLE_HEADERS = ["class", "confidence", "severity", "action", "box (x1, y1, x2, y2)"]
MAX_SIDE = 1600


@dataclass
class Assets:
    """What startup found: local ONNX path (None if unavailable), example image paths, and messages for failures."""

    onnx_path: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    error: Optional[str] = None  # model unavailable: inference cannot run
    example_note: Optional[str] = None  # examples unavailable: upload-only mode


def _download(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename)


def _list_files(repo: str) -> List[str]:
    from huggingface_hub import list_repo_files

    return list_repo_files(repo)


def local_examples(directory: str) -> List[str]:
    """Image files in a directory, sorted (refuses a directory named 'test': val images only)."""
    d = Path(directory)
    if d.name.lower() == "test":
        raise ValueError("the test split is spent; examples come from val only")
    return sorted(str(p) for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if d.is_dir() else []


def fetch_assets(repo: str = MODEL_REPO, download: Callable[[str, str], str] = _download,
                 list_files: Callable[[str], List[str]] = _list_files) -> Assets:
    """best.onnx plus any examples/*.jpg from the model repo. A failed model download gives error set, onnx_path None."""
    assets = Assets()
    env_onnx = os.environ.get("ONNX_PATH")
    try:
        assets.onnx_path = env_onnx if env_onnx and Path(env_onnx).is_file() else download(repo, ONNX_FILE)
    except Exception as exc:  # network, missing file or permissions: fall back to upload-only
        assets.error = (f"Could not download the model from {repo} ({type(exc).__name__}); inference cannot run "
                        "without it. Reload the page to retry.")
        return assets
    if os.environ.get("EXAMPLES_DIR"):
        assets.examples = local_examples(os.environ["EXAMPLES_DIR"])
        return assets
    try:
        names = [f for f in list_files(repo) if f.startswith("examples/") and f.lower().endswith(IMAGE_SUFFIXES)]
        assets.examples = [download(repo, n) for n in sorted(names)[:6]]
    except Exception as exc:  # examples are optional: fall back to upload-only
        assets.examples = []
        assets.example_note = f"Example images could not be downloaded ({type(exc).__name__}); upload your own image."
        return assets
    if not assets.examples:
        assets.example_note = "No example images are available in the model repo; upload your own image."
    return assets


def build_service(onnx_path: str) -> InspectionService:
    """Service with the ONNX backend and the frozen primary policy (paths relative to the repo root)."""
    settings = Settings(backend="onnx", onnx_path=onnx_path, frozen_path=str(ROOT / "reports/phase3/frozen.json"),
                        cost_path=str(ROOT / "configs/cost.yaml"))
    return InspectionService(settings, OnnxDetector(onnx_path), version=f"onnx:{Path(onnx_path).name}")


def pick_examples(val_dir: str, n: int = 6) -> List[Path]:
    """First val image (alphabetical) of each of up to n classes; refuses the test split."""
    d = Path(val_dir)
    if d.name.lower() == "test":
        raise ValueError("the test split is spent; examples come from val only")
    first = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            first.setdefault(p.stem.rsplit("_", 1)[0], p)
    return list(first.values())[:n]


def prepare(image: Image.Image) -> Image.Image:
    """RGB copy, downscaled so the longest side is at most MAX_SIDE."""
    im = image.convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    return im


def annotate(image: Image.Image, result: ImageResult) -> Image.Image:
    """Copy of the image with each detection's box coloured by action and labelled 'class conf'."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for d in result.detections:
        color = ACTION_COLORS.get(d.action, "#000000")
        x1, y1, x2, y2 = d.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{d.class_name} {d.confidence:.2f}"
        tx, ty = x1 + 2, max(0.0, y1 - 11)
        draw.rectangle([tx - 1, ty, tx + 6 * len(label), ty + 11], fill=color)
        draw.text((tx, ty), label, fill="white")
    return out


def rows_for(result: ImageResult) -> List[list]:
    """Table rows, one per detection, most confident first."""
    return [[d.class_name, round(d.confidence, 3), round(d.severity, 2), d.action, [round(v) for v in d.box]]
            for d in result.detections]


def summary_text(result: ImageResult, show_ignored: bool) -> str:
    """Markdown line with the per-image decision and the detection count."""
    shown = sum(d.action != "ignore" for d in result.detections)
    note = "" if show_ignored else " (detections with action 'ignore' are hidden)"
    return (f"**Image decision: `{result.decision}`** (most urgent action over detections). "
            f"{shown} actionable detection(s){note}.")


def analyze(image: Optional[Image.Image], show_ignored: bool, service: InspectionService):
    """(annotated image, table rows, summary markdown) for one uploaded image."""
    if image is None:
        return None, [], "Upload an image or pick an example."
    if not service.loaded:
        return None, [], "Model weights are not loaded."
    im = prepare(image)
    result = service.run(im, None, include_ignored=True)
    if not show_ignored:
        result = result.model_copy(update={"detections": [d for d in result.detections if d.action != "ignore"]})
    return annotate(im, result), rows_for(result), summary_text(result, show_ignored)


def check_image_compatibility(image: Image.Image) -> Optional[str]:
    """Soft warning heuristic: returns a message if the image is color or much larger than 200x200."""
    reasons = []
    w, h = image.size
    if max(w, h) > 400:
        reasons.append(f"much larger than the 200×200 training size ({w}×{h})")
    if image.mode in ("RGB", "RGBA"):
        arr = np.array(image.convert("RGB"))
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        if int(np.max(np.abs(r.astype(int) - g.astype(int)))) > 5 or int(np.max(np.abs(r.astype(int) - b.astype(int)))) > 5:
            reasons.append("a color image (model was trained on grayscale only)")
    if reasons:
        return (
            f"Heuristic warning: This image is {' and '.join(reasons)}. "
            "The model only works on NEU-DET-style grayscale close-ups of hot-rolled steel surfaces "
            "(about 200×200); other photos give meaningless results."
        )
    return None


