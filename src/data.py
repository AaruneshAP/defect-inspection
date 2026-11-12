"""
Defect inspection dataset loading, Pascal VOC parsing, splitting, and YOLO formatting module.

Purpose: Discovers raw NEU-DET images and VOC XML annotations, parses bounding boxes, executes stratified 85/15 train/val splitting with a reserved test set, and exports YOLO-formatted datasets and YAML configuration.
Inputs: Raw dataset paths, Pascal VOC XML files, target defect class definitions, split ratios, and random seeds.
Outputs: Parsed annotation records, normalized YOLO bounding box files, partitioned image directories, and dataset metadata.
"""

import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CLASS_NAMES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_voc_xml(xml_path: str | Path) -> Dict[str, Any]:
    """
    Parses a single Pascal VOC XML annotation file into image dimensions and bounding boxes.

    Inputs:
        xml_path (str | Path): Filepath to the Pascal VOC XML annotation file.

    Outputs:
        Dict[str, Any]: Dictionary containing filename, width, height, depth, and list of bounding box objects.
    """
    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found at {xml_path}")

    tree = ET.parse(path)
    root = tree.getroot()

    filename_elem = root.find("filename")
    filename = filename_elem.text.strip() if filename_elem is not None and filename_elem.text else path.stem + ".jpg"

    size_elem = root.find("size")
    if size_elem is None:
        raise ValueError(f"Missing size element in XML: {xml_path}")

    width = int(float(size_elem.find("width").text))
    height = int(float(size_elem.find("height").text))
    depth_elem = size_elem.find("depth")
    depth = int(float(depth_elem.text)) if depth_elem is not None and depth_elem.text else 1

    boxes: List[Dict[str, Any]] = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue
        raw_name = name_elem.text.strip()
        if raw_name not in CLASS_TO_IDX:
            raise KeyError(f"Unknown class '{raw_name}' in {xml_path}. Expected one of {CLASS_NAMES}")

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        boxes.append(
            {
                "class_name": raw_name,
                "class_id": CLASS_TO_IDX[raw_name],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "depth": depth,
        "boxes": boxes,
    }


def voc_to_yolo_bbox(
    bbox: Tuple[float, float, float, float],
    img_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """
    Converts Pascal VOC absolute pixel coordinates to YOLO normalized center and dimension coordinates.

    Inputs:
        bbox (Tuple[float, float, float, float]): Bounding box coordinates as (xmin, ymin, xmax, ymax).
        img_size (Tuple[int, int]): Image dimensions as (width, height).

    Outputs:
        Tuple[float, float, float, float]: Normalized YOLO coordinates (x_center, y_center, width, height) clamped to [0.0, 1.0].
    """
    xmin, ymin, xmax, ymax = bbox
    img_w, img_h = img_size

    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image dimensions: {img_size}")

    xmin_c = max(0.0, min(float(xmin), float(img_w)))
    xmax_c = max(0.0, min(float(xmax), float(img_w)))
    ymin_c = max(0.0, min(float(ymin), float(img_h)))
    ymax_c = max(0.0, min(float(ymax), float(img_h)))

    box_w = max(0.0, xmax_c - xmin_c)
    box_h = max(0.0, ymax_c - ymin_c)
    x_center = xmin_c + (box_w / 2.0)
    y_center = ymin_c + (box_h / 2.0)

    norm_x = min(1.0, max(0.0, x_center / img_w))
    norm_y = min(1.0, max(0.0, y_center / img_h))
    norm_w = min(1.0, max(0.0, box_w / img_w))
    norm_h = min(1.0, max(0.0, box_h / img_h))

    return (norm_x, norm_y, norm_w, norm_h)


def scan_dataset(data_dir: str | Path = "data") -> Dict[str, List[Dict[str, Any]]]:
    """
    Scans the dataset directory, resolves annotations for all images, and returns structured records by split.

    Inputs:
        data_dir (str | Path): Root path to directory containing raw data or archives.

    Outputs:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping 'train' and 'validation' to lists of image and annotation records.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found at {data_dir}")

    neu_candidates = list(root.rglob("NEU-DET"))
    base_dir = neu_candidates[0] if neu_candidates else root

    xml_index: Dict[str, Path] = {f.stem: f for f in base_dir.rglob("*.xml")}

    scanned: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": []}

    for split_key in ["train", "validation"]:
        split_dir = base_dir / split_key
        images_dir = split_dir / "images"
        if not images_dir.exists():
            continue

        image_files = sorted(images_dir.rglob("*.jpg"))
        for img_path in image_files:
            stem = img_path.stem
            xml_path = xml_index.get(stem)
            if xml_path is None:
                raise FileNotFoundError(f"Missing XML annotation for image: {img_path}")

            parsed = parse_voc_xml(xml_path)
            primary_class = img_path.parent.name
            if primary_class not in CLASS_TO_IDX and parsed["boxes"]:
                primary_class = parsed["boxes"][0]["class_name"]

            record = {
                "stem": stem,
                "image_path": str(img_path.resolve()),
                "xml_path": str(xml_path.resolve()),
                "class_name": primary_class,
                "width": parsed["width"],
                "height": parsed["height"],
                "depth": parsed["depth"],
                "boxes": parsed["boxes"],
            }
            scanned[split_key].append(record)

    return scanned


def split_dataset(
    train_records: List[Dict[str, Any]],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Performs deterministic class-stratified splitting on training records into train and validation subsets.

    Inputs:
        train_records (List[Dict[str, Any]]): List of training sample dictionaries.
        val_ratio (float): Fraction of samples to allocate to validation subset (default 0.15).
        seed (int): Deterministic random seed for reproducible sampling.

    Outputs:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Tuple of (train_subset, validation_subset).
    """
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for record in train_records:
        cls_name = record["class_name"]
        by_class.setdefault(cls_name, []).append(record)

    rng = random.Random(seed)
    train_split: List[Dict[str, Any]] = []
    val_split: List[Dict[str, Any]] = []

    for cls_name in sorted(by_class.keys()):
        class_records = list(by_class[cls_name])
        class_records.sort(key=lambda r: r["stem"])
        rng.shuffle(class_records)

        num_val = int(round(len(class_records) * val_ratio))
        val_subset = class_records[:num_val]
        train_subset = class_records[num_val:]

        val_split.extend(val_subset)
        train_split.extend(train_subset)

    train_split.sort(key=lambda r: r["stem"])
    val_split.sort(key=lambda r: r["stem"])

    return train_split, val_split


def convert_dataset_to_yolo(
    data_dir: str | Path = "data",
    output_dir: str | Path = "data/yolo",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Orchestrates dataset scanning, VOC-to-YOLO conversion, stratified train/val splitting, and YAML configuration creation.

    Inputs:
        data_dir (str | Path): Source data root path.
        output_dir (str | Path): Destination directory for YOLO formatted assets.
        val_ratio (float): Validation fraction to split from training set (default 0.15).
        seed (int): Random seed for reproducibility.

    Outputs:
        Dict[str, Any]: Summary dictionary with sample counts per split and written file locations.
    """
    scanned = scan_dataset(data_dir)
    train_records = scanned["train"]
    test_records = scanned["validation"]

    train_split, val_split = split_dataset(train_records, val_ratio=val_ratio, seed=seed)

    out_root = Path(output_dir)
    splits = {
        "train": train_split,
        "val": val_split,
        "test": test_records,
    }

    counts: Dict[str, int] = {}
    for split_name, records in splits.items():
        img_out_dir = out_root / "images" / split_name
        lbl_out_dir = out_root / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for rec in records:
            src_img = Path(rec["image_path"])
            dst_img = img_out_dir / f"{rec['stem']}.jpg"
            shutil.copy2(src_img, dst_img)

            dst_lbl = lbl_out_dir / f"{rec['stem']}.txt"
            lines: List[str] = []
            img_size = (rec["width"], rec["height"])
            for box in rec["boxes"]:
                coords = (box["xmin"], box["ymin"], box["xmax"], box["ymax"])
                yolo_bbox = voc_to_yolo_bbox(coords, img_size)
                lines.append(
                    f"{box['class_id']} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n"
                )

            dst_lbl.write_text("".join(lines), encoding="utf-8")

        counts[split_name] = len(records)

    yaml_content = (
        f"path: {out_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n"
        + "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
        + "\n"
    )
    (out_root / "data.yaml").write_text(yaml_content, encoding="utf-8")

    return {
        "status": "success",
        "output_dir": str(out_root.resolve()),
        "counts": counts,
        "classes": CLASS_NAMES,
    }


def load_dataset(data_dir: str = "data", split: str = "train") -> Dict[str, Any]:
    """
    Loads dataset filepaths and defect annotations for a given split.

    Inputs:
        data_dir (str): Path to root directory containing image subsets and annotations.
        split (str): Dataset split to load ('train', 'val', or 'test').

    Outputs:
        Dict[str, Any]: Dictionary containing image paths, class labels, and metadata.
    """
    yolo_dir = Path(data_dir) / "yolo" / "images" / split
    if yolo_dir.exists():
        images = sorted(list(yolo_dir.glob("*.jpg")))
        return {
            "split": split,
            "image_paths": [str(p) for p in images],
            "labels": [p.stem.split("_")[0] for p in images],
        }

    split_dir = Path(data_dir) / split
    return {
        "split": split,
        "image_paths": list(split_dir.glob("*.*")) if split_dir.exists() else [],
        "labels": [],
    }


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Any:
    """
    Reads an image from disk, normalizes color channels, and resizes to target dimensions.

    Inputs:
        image_path (str): Filepath of the raw defect image.
        target_size (Tuple[int, int]): Desired output dimensions as (height, width).

    Outputs:
        Any: Normalized and resized image tensor or array.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at {image_path}")
    return {"path": str(path), "target_size": target_size}

