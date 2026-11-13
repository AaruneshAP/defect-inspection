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
    build_comparison_table,
    evaluate,
    is_cache_valid,
    load_config,
    parse_training_metrics,
    plot_per_class_comparison,
    resolve_training_epochs,
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
    Verifies that all code cells in all notebooks parse without Python syntax errors.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError or SyntaxError if any code cell contains invalid syntax.
    """
    import ast
    import nbformat

    notebooks = sorted(list(Path("notebooks").glob("*.ipynb")))
    assert len(notebooks) >= 2

    for nb_path in notebooks:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                try:
                    ast.parse(cell.source)
                except SyntaxError as err:
                    raise AssertionError(f"SyntaxError in {nb_path.name} code cell {idx}: {err}")


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


def test_phase2b_experiment_configs() -> None:
    """
    Verifies that Phase 2b experiment configurations (E1, E2, E3) load with expected hyperparameters.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError if configs are missing or invalid.
    """
    e1 = load_config("configs/e1_imgsz320.yaml")
    assert e1["imgsz"] == 320
    assert e1["model"] == "yolov8s.pt"
    assert e1["project"] == "runs"

    e2 = load_config("configs/e2_yolov8m_640.yaml")
    assert e2["imgsz"] == 640
    assert e2["model"] == "yolov8m.pt"
    assert e2["project"] == "runs"

    e3 = load_config("configs/e3_augmented.yaml")
    assert e3["imgsz"] == 640
    assert e3["model"] == "yolov8s.pt"
    assert e3["fliplr"] == 0.5
    assert e3["flipud"] == 0.5
    assert e3["hsv_v"] == 0.4
    assert e3["mixup"] == 0.0
    assert e3["close_mosaic"] == 10


def test_build_comparison_table_from_dict(tmp_path: Path) -> None:
    """
    Verifies that build_comparison_table flattens experiment dictionaries and writes a valid CSV file.

    Inputs:
        tmp_path (Path): Pytest temporary directory fixture.

    Outputs:
        None: Raises AssertionError if comparison rows or CSV output are malformed.
    """
    import csv

    sample_experiments = {
        "E0_baseline": {
            "model": "yolov8s.pt",
            "imgsz": 640,
            "precision": 0.725,
            "recall": 0.713,
            "mAP50": 0.755,
            "mAP50-95": 0.430,
            "per_class_ap": {
                "crazing": {"ap50": 0.419, "ap50-95": 0.147},
                "inclusion": {"ap50": 0.813, "ap50-95": 0.446},
                "patches": {"ap50": 0.877, "ap50-95": 0.564},
                "pitted_surface": {"ap50": 0.845, "ap50-95": 0.602},
                "rolled-in_scale": {"ap50": 0.622, "ap50-95": 0.313},
                "scratches": {"ap50": 0.952, "ap50-95": 0.508},
            },
        },
        "E1_imgsz320": {
            "model": "yolov8s.pt",
            "imgsz": 320,
            "precision": 0.680,
            "recall": 0.670,
            "mAP50": 0.710,
            "mAP50-95": 0.390,
            "per_class_ap": {
                "crazing": {"ap50": 0.380, "ap50-95": 0.120},
                "inclusion": {"ap50": 0.790, "ap50-95": 0.410},
                "patches": {"ap50": 0.850, "ap50-95": 0.520},
                "pitted_surface": {"ap50": 0.810, "ap50-95": 0.570},
                "rolled-in_scale": {"ap50": 0.580, "ap50-95": 0.280},
                "scratches": {"ap50": 0.930, "ap50-95": 0.480},
            },
        },
    }

    out_csv = tmp_path / "comparison.csv"
    rows = build_comparison_table(sample_experiments, output_csv=out_csv)

    assert len(rows) == 2
    assert rows[0]["experiment"] == "E0_baseline"
    assert rows[0]["mAP50"] == 0.755
    assert rows[0]["crazing_AP50"] == 0.419
    assert rows[0]["scratches_AP50"] == 0.952
    assert rows[1]["experiment"] == "E1_imgsz320"
    assert rows[1]["imgsz"] == 320

    assert out_csv.is_file()
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 2
    assert "crazing_AP50" in reader[0]
    assert float(reader[0]["crazing_AP50"]) == 0.419


def test_build_comparison_table_from_directory(tmp_path: Path) -> None:
    """
    Verifies that build_comparison_table scans directory trees for results.json files.

    Inputs:
        tmp_path (Path): Pytest temporary directory fixture.

    Outputs:
        None: Raises AssertionError if directory scanning fails.
    """
    exp_dir1 = tmp_path / "reports" / "phase2b" / "e0_baseline"
    exp_dir2 = tmp_path / "reports" / "phase2b" / "e3_augmented"
    exp_dir1.mkdir(parents=True)
    exp_dir2.mkdir(parents=True)

    payload1 = {"model": "yolov8s.pt", "mAP50": 0.755, "per_class_ap": {"crazing": {"ap50": 0.419}}}
    payload2 = {"model": "yolov8s.pt", "mAP50": 0.785, "per_class_ap": {"crazing": {"ap50": 0.510}}}

    with open(exp_dir1 / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload1, f)
    with open(exp_dir2 / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload2, f)

    rows = build_comparison_table(tmp_path / "reports" / "phase2b")
    assert len(rows) == 2
    exp_names = {r["experiment"] for r in rows}
    assert "e0_baseline" in exp_names
    assert "e3_augmented" in exp_names


def test_plot_per_class_comparison(tmp_path: Path) -> None:
    """
    Verifies that plot_per_class_comparison produces a valid PNG bar chart from comparison rows.

    Inputs:
        tmp_path (Path): Pytest temporary directory fixture.

    Outputs:
        None: Raises AssertionError if plot generation fails.
    """
    rows = [
        {"experiment": "E0", "crazing_AP50": 0.419, "scratches_AP50": 0.952},
        {"experiment": "E3", "crazing_AP50": 0.520, "scratches_AP50": 0.960},
    ]
    out_png = tmp_path / "plot.png"
    result = plot_per_class_comparison(rows, output_png=out_png)
    assert Path(result).is_file()
    assert Path(result).stat().st_size > 1000


def test_smoke_test_flag_changes_epochs() -> None:
    """
    Verifies that the smoke_test flag overrides config epochs to 1 for quick pipeline testing,
    and preserves real configuration epochs when False.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError if epoch resolution is incorrect.
    """
    for cfg_path in [
        "configs/baseline.yaml",
        "configs/e1_imgsz320.yaml",
        "configs/e2_yolov8m_640.yaml",
        "configs/e3_augmented.yaml",
    ]:
        cfg = load_config(cfg_path)
        assert cfg["epochs"] == 50

        epochs_smoke = resolve_training_epochs(cfg, smoke_test=True)
        assert epochs_smoke == 1

        epochs_real = resolve_training_epochs(cfg, smoke_test=False)
        assert epochs_real == 50
        assert epochs_real == cfg["epochs"]


def test_experiments_notebook_smoke_test_flag() -> None:
    """
    Verifies that notebooks/03_experiments.ipynb defines SMOKE_TEST = True at the top and forwards it.

    Inputs:
        None.

    Outputs:
        None: Raises AssertionError if SMOKE_TEST is not configured in the notebook.
    """
    import nbformat

    with open("notebooks/03_experiments.ipynb", "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    found_smoke_declaration = False
    found_smoke_usage = False

    for cell in nb.cells:
        if cell.cell_type == "code":
            if "SMOKE_TEST = True" in cell.source:
                found_smoke_declaration = True
            if "smoke_test=SMOKE_TEST" in cell.source:
                found_smoke_usage = True

    assert found_smoke_declaration, "SMOKE_TEST = True declaration missing in notebooks/03_experiments.ipynb"
    assert found_smoke_usage, "smoke_test=SMOKE_TEST usage missing in notebooks/03_experiments.ipynb"


def test_cached_smoke_result_not_reused_when_smoke_test_false(tmp_path: Path) -> None:
    """
    Verifies that a cached 1-epoch smoke test result is NOT reused when running in real mode (SMOKE_TEST=False).

    Inputs:
        tmp_path (Path): Pytest temporary directory fixture.

    Outputs:
        None: Raises AssertionError if cache validation incorrectly accepts mismatched configs.
    """
    config = {
        "model": "yolov8s.pt",
        "imgsz": 640,
        "seed": 42,
        "epochs": 50,
    }

    # 1. Smoke test result (epochs=1, smoke_test=True)
    smoke_file = tmp_path / "smoke_results.json"
    smoke_payload = {
        "model": "yolov8s.pt",
        "imgsz": 640,
        "seed": 42,
        "epochs": 1,
        "smoke_test": True,
        "mAP50": 0.150,
    }
    with open(smoke_file, "w", encoding="utf-8") as f:
        json.dump(smoke_payload, f)

    # When SMOKE_TEST=False, the smoke-test cached result MUST NOT be reused
    assert not is_cache_valid(smoke_file, config, smoke_test=False)

    # When SMOKE_TEST=True, the smoke-test cached result CAN be reused
    assert is_cache_valid(smoke_file, config, smoke_test=True)

    # 2. Real training result (epochs=50, smoke_test=False)
    real_file = tmp_path / "real_results.json"
    real_payload = {
        "model": "yolov8s.pt",
        "imgsz": 640,
        "seed": 42,
        "epochs": 50,
        "smoke_test": False,
        "mAP50": 0.755,
    }
    with open(real_file, "w", encoding="utf-8") as f:
        json.dump(real_payload, f)

    # Real run matches real config
    assert is_cache_valid(real_file, config, smoke_test=False)

    # Real run does NOT match smoke test mode
    assert not is_cache_valid(real_file, config, smoke_test=True)

    # 3. Model, image size, or seed mismatch must also invalidate cache
    assert not is_cache_valid(dict(real_payload, model="yolov8m.pt"), config, smoke_test=False)
    assert not is_cache_valid(dict(real_payload, imgsz=320), config, smoke_test=False)
    assert not is_cache_valid(dict(real_payload, seed=99), config, smoke_test=False)






