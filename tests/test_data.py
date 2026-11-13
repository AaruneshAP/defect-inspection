"""
Unit and integration tests for dataset scanning, XML parsing, and YOLO conversion.

Purpose: Validates Pascal VOC XML parsing, coordinate normalization bounds, split counts, and split isolation.
Inputs: Sample XML files, converted YOLO label files, and split directory listings.
Outputs: Pytest assertion passes or failures enforcing data integrity.
"""

from pathlib import Path

import pytest

from src.data import CLASS_NAMES, CLASS_TO_IDX, parse_voc_xml, voc_to_yolo_bbox


@pytest.mark.requires_data
def test_xml_parser_output() -> None:
    """
    Verifies that parse_voc_xml extracts valid dimensions and bounding boxes.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError if XML parsing structure or values are invalid.
    """
    xml_files = list(Path("data").rglob("*.xml"))
    assert len(xml_files) > 0

    sample_xml = xml_files[0]
    parsed = parse_voc_xml(sample_xml)

    assert "filename" in parsed
    assert "width" in parsed
    assert "height" in parsed
    assert "depth" in parsed
    assert "boxes" in parsed
    assert parsed["width"] > 0
    assert parsed["height"] > 0
    assert len(parsed["boxes"]) > 0

    first_box = parsed["boxes"][0]
    assert first_box["class_name"] in CLASS_NAMES
    assert first_box["class_id"] == CLASS_TO_IDX[first_box["class_name"]]
    assert 0 <= first_box["xmin"] < first_box["xmax"] <= parsed["width"]
    assert 0 <= first_box["ymin"] < first_box["ymax"] <= parsed["height"]


def test_converted_boxes_lie_within_image() -> None:
    """
    Verifies that all converted YOLO bounding boxes have coordinates strictly within [0.0, 1.0].

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError if any coordinate falls outside normalized bounds.
    """
    synthetic_bbox = (10.0, 20.0, 190.0, 180.0)
    synthetic_size = (200, 200)
    xc, yc, w, h = voc_to_yolo_bbox(synthetic_bbox, synthetic_size)

    assert 0.0 <= xc <= 1.0
    assert 0.0 <= yc <= 1.0
    assert 0.0 < w <= 1.0
    assert 0.0 < h <= 1.0


@pytest.mark.requires_data
def test_label_files_are_normalized() -> None:
    """Every YOLO label file under data/yolo has 5 fields, a valid class id and coordinates in [0, 1]."""
    label_files = list(Path("data/yolo/labels").rglob("*.txt"))
    assert len(label_files) > 0

    for lbl_path in label_files:
        content = lbl_path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        for line in content.splitlines():
            parts = line.strip().split()
            assert len(parts) == 5
            cls_id = int(parts[0])
            xc, yc, w, h = (float(v) for v in parts[1:])
            assert 0 <= cls_id < len(CLASS_NAMES)
            assert 0.0 <= xc <= 1.0
            assert 0.0 <= yc <= 1.0
            assert 0.0 <= w <= 1.0
            assert 0.0 <= h <= 1.0


@pytest.mark.requires_data
def test_image_and_label_counts_match_per_split() -> None:
    """
    Verifies that image and label counts match exactly within each split.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError if image and label counts differ.
    """
    yolo_root = Path("data/yolo")
    expected_counts = {"train": 1224, "val": 216, "test": 360}

    for split in ["train", "val", "test"]:
        imgs = sorted(list((yolo_root / "images" / split).glob("*.jpg")))
        lbls = sorted(list((yolo_root / "labels" / split).glob("*.txt")))

        assert len(imgs) == expected_counts[split]
        assert len(lbls) == expected_counts[split]

        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        assert img_stems == lbl_stems


@pytest.mark.requires_data
def test_no_image_appears_in_two_splits() -> None:
    """
    Verifies that no image is shared between any two dataset splits.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError if split sets overlap.
    """
    yolo_root = Path("data/yolo")
    train_stems = {p.stem for p in (yolo_root / "images" / "train").glob("*.jpg")}
    val_stems = {p.stem for p in (yolo_root / "images" / "val").glob("*.jpg")}
    test_stems = {p.stem for p in (yolo_root / "images" / "test").glob("*.jpg")}

    assert train_stems.isdisjoint(val_stems)
    assert train_stems.isdisjoint(test_stems)
    assert val_stems.isdisjoint(test_stems)
    assert len(train_stems) + len(val_stems) + len(test_stems) == 1800

