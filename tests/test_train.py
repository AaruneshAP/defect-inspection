"""
Unit tests for YOLO baseline training configuration and metrics parsing.

Purpose: Validates baseline configuration loading, image size constraints, metric extraction, and evaluation split protection.
Inputs: Baseline YAML configuration files, mock Ultralytics results objects, and synthetic metric dictionaries.
Outputs: Pytest assertion passes or failures enforcing training pipeline integrity.
"""

from pathlib import Path
from typing import Any
import json
import pytest
from src.data import (
    CLASS_NAMES,
    find_kaggle_dataset,
)
from src.train import (
    DEFAULT_CLASSES,
    evaluate,
    load_config,
    parse_training_metrics,
    save_results,
)


def test_load_config_valid() -> None:
    """
    Verifies that configs/baseline.yaml loads successfully with all required hyperparameters.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError if required fields are missing or misconfigured.
    """
    config = load_config("configs/baseline.yaml")
    assert isinstance(config, dict)
    assert config["model"] == "yolov8s.pt"
    assert config["seed"] == 42
    assert config["imgsz"] == 640
    assert config["imgsz"] % 32 == 0
    assert config["patience"] == 15
    assert config["epochs"] > 0
    assert config["batch"] > 0
    assert "data" in config


def test_load_config_320_variant() -> None:
    """
    Verifies that configs/baseline_320.yaml variant loads with imgsz=320 and valid settings.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError if 320 variant is misconfigured.
    """
    config = load_config("configs/baseline_320.yaml")
    assert isinstance(config, dict)
    assert config["model"] == "yolov8s.pt"
    assert config["imgsz"] == 320
    assert config["imgsz"] % 32 == 0
    assert config["seed"] == 42
    assert config["patience"] == 15


def test_load_config_missing_file() -> None:
    """
    Verifies that load_config raises FileNotFoundError when given a nonexistent path.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError if FileNotFoundError is not raised.
    """
    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent.yaml")


def test_load_config_invalid_imgsz(tmp_path: Path) -> None:
    """
    Verifies that load_config rejects image dimensions that are not multiples of 32.

    Inputs:
        tmp_path (Path): Pytest temporary directory fixture.

    Outputs:
        None: Raises AssertionError if ValueError is not raised for invalid dimensions.
    """
    bad_config = tmp_path / "bad_imgsz.yaml"
    bad_config.write_text(
        "model: yolov8s.pt\ndata: data/yolo/data.yaml\nepochs: 10\nimgsz: 200\nbatch: 16\nseed: 42\npatience: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple of 32"):
        load_config(bad_config)


def test_load_config_missing_keys(tmp_path: Path) -> None:
    """
    Verifies that load_config rejects configurations lacking mandatory hyperparameters.

    Inputs:
        tmp_path (Path): Pytest temporary directory fixture.

    Outputs:
        None: Raises AssertionError if ValueError is not raised for missing keys.
    """
    incomplete_config = tmp_path / "incomplete.yaml"
    incomplete_config.write_text("model: yolov8s.pt\nepochs: 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required configuration keys"):
        load_config(incomplete_config)


def test_parse_training_metrics_dict() -> None:
    """
    Verifies that parse_training_metrics handles dictionary formatted validation results.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError if metrics extraction is incorrect.
    """
    raw_dict = {
        "metrics/precision(B)": 0.852,
        "metrics/recall(B)": 0.794,
        "metrics/mAP50(B)": 0.831,
        "metrics/mAP50-95(B)": 0.542,
    }
    parsed = parse_training_metrics(raw_dict)
    assert parsed["precision"] == 0.852
    assert parsed["recall"] == 0.794
    assert parsed["mAP50"] == 0.831
    assert parsed["mAP50-95"] == 0.542
    assert isinstance(parsed["per_class_ap"], dict)


def test_parse_training_metrics_object() -> None:
    """
    Verifies that parse_training_metrics extracts overall and per-class AP from DetMetrics-like objects.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError if metric parsing or per-class breakdown fails.
    """
    class MockBox:
        mp = 0.885
        mr = 0.812
        map50 = 0.871
        map = 0.593
        ap50 = [0.85, 0.90, 0.82, 0.88, 0.89, 0.88]
        ap = [0.55, 0.62, 0.51, 0.60, 0.64, 0.64]
        ap_class_index = [0, 1, 2, 3, 4, 5]

    class MockResults:
        box = MockBox()

    parsed = parse_training_metrics(MockResults())
    assert parsed["precision"] == 0.885
    assert parsed["recall"] == 0.812
    assert parsed["mAP50"] == 0.871
    assert parsed["mAP50-95"] == 0.593

    for cls_name in DEFAULT_CLASSES:
        assert cls_name in parsed["per_class_ap"]
        assert "ap50" in parsed["per_class_ap"][cls_name]
        assert "ap50-95" in parsed["per_class_ap"][cls_name]
        assert parsed["per_class_ap"][cls_name]["ap50"] > 0.0


def test_evaluate_rejects_test_split() -> None:
    """
    Verifies that evaluate refuses to run on the test split during training, protecting against data leakage.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError if ValueError is not raised when split='test'.
    """
    with pytest.raises(ValueError, match="reserved"):
        evaluate(model=None, split="test")


def test_notebook_code_cells_syntax() -> None:
    """
    Verifies that all code cells in notebooks/02_yolo_baseline.ipynb parse without Python syntax errors.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError or SyntaxError if any code cell contains invalid syntax.
    """
    import ast
    import nbformat

    notebook_path = Path("notebooks/02_yolo_baseline.ipynb")
    assert notebook_path.is_file()

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            try:
                ast.parse(cell.source)
            except SyntaxError as err:
                raise AssertionError(f"SyntaxError in code cell {idx}: {err}")


def test_save_results(tmp_path: Path) -> None:
    """
    Verifies that save_results creates parent directories and writes valid results.json with per-class AP.

    Inputs:
        tmp_path (Path): Pytest temporary directory fixture.

    Outputs:
        None: Raises AssertionError if file is not created or schema is invalid.
    """
    dest = tmp_path / "deep" / "nested" / "results.json"
    payload = {
        "model": "yolov8s.pt",
        "epochs": 50,
        "imgsz": 640,
        "mAP50": 0.882,
        "mAP50-95": 0.615,
        "precision": 0.891,
        "recall": 0.824,
        "per_class_ap": {
            "crazing": {"ap50": 0.851, "ap50-95": 0.582},
            "inclusion": {"ap50": 0.912, "ap50-95": 0.634},
            "patches": {"ap50": 0.941, "ap50-95": 0.702},
            "pitted_surface": {"ap50": 0.873, "ap50-95": 0.591},
            "rolled-in_scale": {"ap50": 0.864, "ap50-95": 0.573},
            "scratches": {"ap50": 0.852, "ap50-95": 0.608},
        },
    }
    written_path = save_results(payload, dest)
    assert Path(written_path).is_file()

    with open(written_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["mAP50"] == 0.882
    assert loaded["mAP50-95"] == 0.615
    assert len(loaded["per_class_ap"]) == 6
    assert loaded["per_class_ap"]["crazing"]["ap50"] == 0.851
    assert loaded["per_class_ap"]["scratches"]["ap50-95"] == 0.608


def test_find_kaggle_dataset_missing_error(tmp_path: Path) -> None:
    """
    Verifies that find_kaggle_dataset raises FileNotFoundError with explicit guidance when absent.

    Inputs:
        tmp_path (Path): Pytest temporary directory fixture.

    Outputs:
        None: Raises AssertionError if expected FileNotFoundError is not raised.
    """
    with pytest.raises(FileNotFoundError, match="Attach the NEU Surface Defect Database under Add Input"):
        find_kaggle_dataset(tmp_path / "nonexistent")

    empty_root = tmp_path / "empty_kaggle"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match="Attach the NEU Surface Defect Database under Add Input"):
        find_kaggle_dataset(empty_root)

    valid_neu = empty_root / "neu-surface-defect-database" / "NEU-DET"
    valid_neu.mkdir(parents=True)
    found = find_kaggle_dataset(empty_root)
    assert found == valid_neu


def test_results_path_trainer_save_dir_and_project_runs(tmp_path: Path) -> None:
    """
    Verifies that baseline configurations set project to 'runs' and save_results handles nested paths.

    Inputs:
        tmp_path (Path): Pytest temporary directory fixture.

    Outputs:
        None: Raises AssertionError if project is not 'runs' or directory creation fails.
    """
    config = load_config("configs/baseline.yaml")
    assert config["project"] == "runs"
    assert config["name"] == "baseline_yolov8s_640"

    config_320 = load_config("configs/baseline_320.yaml")
    assert config_320["project"] == "runs"

    custom_trainer_save_dir = tmp_path / "custom" / "run" / "dir"
    results_path = save_results({"status": "ok"}, custom_trainer_save_dir / "results.json")
    assert Path(results_path).is_file()
    assert Path(results_path).parent == custom_trainer_save_dir



