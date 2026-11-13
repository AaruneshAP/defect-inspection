"""Run a trained model once at very low confidence and cache raw detections + ground truth."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

COLUMNS = ["kind", "image", "class_id", "conf", "x1", "y1", "x2", "y2", "img_w", "img_h"]
CONF_FLOOR = 0.001


def read_gt(label_path: Path, img_w: int, img_h: int, image: str) -> list:
    """Parse a YOLO label file into pixel-space ground-truth rows."""
    rows = []
    if not label_path.is_file():
        return rows
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        rows.append(("gt", image, c, np.nan, (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                     (cx + w / 2) * img_w, (cy + h / 2) * img_h, img_w, img_h))
    return rows


def collect(model_path: str, data_dir: str, split: str, imgsz: int = 640, device: str = "cpu") -> pd.DataFrame:
    """Predict every image in a split; return preds and GT rows in one DataFrame."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    img_dir = Path(data_dir) / "images" / split
    lbl_dir = Path(data_dir) / "labels" / split
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    rows = []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
        res = model.predict(str(p), conf=CONF_FLOOR, imgsz=imgsz, device=device, verbose=False)[0]
        for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            rows.append(("pred", p.name, int(k), float(c), x1, y1, x2, y2, w, h))
        rows += read_gt(lbl_dir / f"{p.stem}.txt", w, h, p.name)
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--allow-test", action="store_true", help="required to touch the reserved test split")
    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--data-dir", default="data/yolo")
    ap.add_argument("--out", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        ap.error("the test split is reserved; pass --allow-test to use it deliberately")
    out = Path(a.out or f"data/cache/{a.split}_detections.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = collect(a.model, a.data_dir, a.split, a.imgsz, a.device)
    df.to_parquet(out, index=False)
    print(f"{out}: {(df.kind == 'pred').sum()} detections, {(df.kind == 'gt').sum()} GT boxes, "
          f"{df.image.nunique()} images")


if __name__ == "__main__":
    main()
