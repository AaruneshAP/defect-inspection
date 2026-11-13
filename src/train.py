"""
Defect inspection YOLOv8 baseline training, evaluation, and metrics extraction module.

Purpose: Manages configuration loading, YOLOv8s baseline training with built-in Ultralytics losses, validation evaluation, metrics parsing (mAP50, mAP50-95, precision, recall, per-class AP), checkpoint preservation, and result artifact exportation.
Inputs: YAML configuration files, YOLO model backbones, dataset specifications, and training hyperparameters.
Outputs: Trained model weights in models/, parsed validation metrics in results.json, and evaluation curve figures.

Image Size Selection Rationale:
The primary baseline uses an input resolution of 640x640 (imgsz=640), matching the native pre-training resolution of YOLOv8s. This maximizes feature resolution in the deeper feature pyramid network (FPN) layers to capture small, fine defect structures (such as scratches and micro-crazing). A secondary configuration variant uses 320x320 (imgsz=320) to compare detection accuracy against inference throughput. Both resolutions are exact multiples of the backbone stride (32), ensuring zero fractional coordinate padding artifacts.
"""

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO

DEFAULT_CLASSES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def load_config(config_path: str | Path = "configs/baseline.yaml") -> Dict[str, Any]:
    """
    Loads and validates the baseline model training YAML configuration.

    Inputs:
        config_path (str | Path): Filepath to the YAML configuration file.

    Outputs:
        Dict[str, Any]: Dictionary containing validated hyperparameters and runtime settings.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_keys = ["model", "data", "epochs", "imgsz", "batch", "seed", "patience"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")

    if config["imgsz"] <= 0 or config["imgsz"] % 32 != 0:
        raise ValueError(f"imgsz must be a positive multiple of 32, got {config['imgsz']}")

    return config


def parse_training_metrics(
    val_results: Any,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Parses validation evaluation metrics into standardized overall and per-class AP summaries.

    Inputs:
        val_results (Any): Ultralytics validation results object or results dictionary.
        class_names (Optional[List[str]]): List of defect class names in order of index.

    Outputs:
        Dict[str, Any]: Structured dictionary with mAP50, mAP50-95, precision, recall, and per_class_ap mapping.
    """
    names = class_names or DEFAULT_CLASSES
    output: Dict[str, Any] = {
        "precision": 0.0,
        "recall": 0.0,
        "mAP50": 0.0,
        "mAP50-95": 0.0,
        "per_class_ap": {},
    }

    if hasattr(val_results, "box"):
        box = val_results.box
        output["precision"] = float(getattr(box, "mp", 0.0))
        output["recall"] = float(getattr(box, "mr", 0.0))
        output["mAP50"] = float(getattr(box, "map50", 0.0))
        output["mAP50-95"] = float(getattr(box, "map", 0.0))

        ap50_vals = getattr(box, "ap50", None)
        ap_vals = getattr(box, "ap", None)
        class_indices = getattr(box, "ap_class_index", list(range(len(names))))

        for idx_pos, cls_idx in enumerate(class_indices):
            cls_name = names[cls_idx] if cls_idx < len(names) else f"class_{cls_idx}"
            ap50 = float(ap50_vals[idx_pos]) if ap50_vals is not None and len(ap50_vals) > idx_pos else 0.0
            ap = float(ap_vals[idx_pos]) if ap_vals is not None and len(ap_vals) > idx_pos else 0.0
            output["per_class_ap"][cls_name] = {
                "ap50": round(ap50, 4),
                "ap50-95": round(ap, 4),
            }
        return output

    if isinstance(val_results, dict):
        results_dict = val_results.get("results_dict", val_results)
        output["precision"] = float(results_dict.get("metrics/precision(B)", 0.0))
        output["recall"] = float(results_dict.get("metrics/recall(B)", 0.0))
        output["mAP50"] = float(results_dict.get("metrics/mAP50(B)", 0.0))
        output["mAP50-95"] = float(results_dict.get("metrics/mAP50-95(B)", 0.0))

        if "per_class_ap" in val_results:
            output["per_class_ap"] = val_results["per_class_ap"]
        return output

    return output


def evaluate(
    model: Any,
    data_yaml: str = "data/yolo/data.yaml",
    split: str = "val",
    imgsz: int = 640,
    device: str = "auto",
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluates detection performance strictly on the validation split and extracts metric metrics.

    Inputs:
        model (Any): Initialized YOLO model instance or path to trained checkpoint.
        data_yaml (str): Path to the dataset YAML configuration.
        split (str): Split partition to evaluate ('val' only; test split is reserved).
        imgsz (int): Image dimension for evaluation.
        device (str): Compute device identifier ('auto', 'cpu', '0').
        class_names (Optional[List[str]]): Defect class name listing.

    Outputs:
        Dict[str, Any]: Summary dictionary containing overall detection metrics and per-class AP breakdown.
    """
    if split == "test":
        raise ValueError("Evaluation during baseline training must only use the 'val' split. The test split is reserved.")

    if isinstance(model, (str, Path)):
        model = YOLO(str(model))

    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    return parse_training_metrics(metrics, class_names=class_names)


def save_checkpoint(
    source_path: str | Path,
    dest_path: str | Path = "models/best.pt",
) -> str:
    """
    Copies trained model weights to the target destination path.

    Inputs:
        source_path (str | Path): Source checkpoint file path.
        dest_path (str | Path): Destination target file path.

    Outputs:
        str: Absolute filepath of the saved checkpoint artifact.
    """
    src = Path(source_path)
    dst = Path(dest_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)
    return str(dst.resolve())


def save_results(
    results_payload: Dict[str, Any],
    output_path: str | Path = "results.json",
) -> str:
    """
    Saves detection training and evaluation metrics dictionary including per-class AP to JSON.

    Inputs:
        results_payload (Dict[str, Any]): Dictionary containing training hyperparameters, overall metrics, and per-class AP.
        output_path (str | Path): Destination file path for results.json.

    Outputs:
        str: Absolute filepath of the written results.json.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    return str(target.resolve())


def resolve_training_epochs(config: Dict[str, Any], smoke_test: bool = False) -> int:
    """
    Determines training epoch count, overriding to 1 when smoke_test is enabled.

    Inputs:
        config (Dict[str, Any]): Dictionary containing configured 'epochs'.
        smoke_test (bool): Flag indicating whether to run a 1-epoch pipeline smoke test.

    Outputs:
        int: Total epochs to execute (1 if smoke_test is True, otherwise config['epochs']).
    """
    if smoke_test:
        return 1
    return int(config.get("epochs", 50))


def is_cache_valid(
    cached_data: Dict[str, Any] | str | Path,
    config: Dict[str, Any],
    smoke_test: bool = False,
) -> bool:
    """
    Validates whether a cached results payload matches the current execution configuration.

    Inputs:
        cached_data (Dict[str, Any] | str | Path): Loaded dictionary or filepath to results.json.
        config (Dict[str, Any]): Current experiment configuration dictionary.
        smoke_test (bool): Current smoke test flag setting.

    Outputs:
        bool: True if model, imgsz, seed, expected epochs, and smoke_test flag all match; otherwise False.
    """
    if isinstance(cached_data, (str, Path)):
        p = Path(cached_data)
        if not p.is_file():
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return False
    elif isinstance(cached_data, dict):
        data = cached_data
    else:
        return False

    expected_epochs = resolve_training_epochs(config, smoke_test=smoke_test)

    matches_model = str(data.get("model")) == str(config.get("model"))
    matches_imgsz = int(data.get("imgsz", -1)) == int(config.get("imgsz", -2))
    matches_seed = int(data.get("seed", -1)) == int(config.get("seed", -2))
    matches_epochs = int(data.get("epochs", -1)) == expected_epochs
    matches_smoke = ("smoke_test" in data) and (bool(data.get("smoke_test")) == bool(smoke_test))

    return matches_model and matches_imgsz and matches_seed and matches_epochs and matches_smoke


def run_training(
    config_path: str = "configs/baseline.yaml",
    smoke_test: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Coordinates baseline YOLOv8 model training, validation scoring, and artifact saving.

    Inputs:
        config_path (str): Filepath to the YAML training configuration.
        smoke_test (bool): If True, trains for 1 epoch only to validate the end-to-end pipeline rapidly.
        kwargs (Any): Runtime parameter overrides.

    Outputs:
        Dict[str, Any]: Execution summary dictionary with best checkpoint path, metrics, and artifact locations.
    """
    config = load_config(config_path)
    config.update(kwargs)

    is_smoke = smoke_test or bool(kwargs.get("smoke_test", False))
    config["epochs"] = resolve_training_epochs(config, smoke_test=is_smoke)

    model = YOLO(config["model"])

    train_args = {
        "data": config["data"],
        "epochs": config["epochs"],
        "batch": config["batch"],
        "imgsz": config["imgsz"],
        "seed": config["seed"],
        "patience": config["patience"],
        "device": config.get("device", "auto"),
        "optimizer": config.get("optimizer", "auto"),
        "lr0": config.get("lr0", 0.001),
        "lrf": config.get("lrf", 0.01),
        "project": config.get("project", "runs"),
        "name": config.get("name", "baseline_yolov8s_640"),
        "save": config.get("save", True),
        "plots": config.get("plots", True),
        "verbose": True,
    }

    augmentation_keys = [
        "fliplr",
        "flipud",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "mosaic",
        "mixup",
        "close_mosaic",
    ]
    for k in augmentation_keys:
        if k in config:
            train_args[k] = config[k]

    train_results = model.train(**train_args)

    if hasattr(model, "trainer") and model.trainer is not None and hasattr(model.trainer, "save_dir"):
        run_dir = Path(model.trainer.save_dir)
    else:
        run_dir = Path(train_args.get("project", "runs")) / "detect" / train_args.get("name", "baseline_yolov8s_640")

    run_dir.mkdir(parents=True, exist_ok=True)

    best_pt_src = getattr(model.trainer, "best", None) if hasattr(model, "trainer") and model.trainer else None
    if not best_pt_src or not Path(best_pt_src).is_file():
        best_pt_src = run_dir / "weights" / "best.pt"

    last_pt_src = getattr(model.trainer, "last", None) if hasattr(model, "trainer") and model.trainer else None
    if not last_pt_src or not Path(last_pt_src).is_file():
        last_pt_src = run_dir / "weights" / "last.pt"

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = save_checkpoint(best_pt_src, models_dir / "best.pt")
    last_checkpoint = save_checkpoint(last_pt_src, models_dir / "last.pt")

    val_metrics = evaluate(
        model=best_checkpoint if Path(best_checkpoint).is_file() else model,
        data_yaml=config["data"],
        split="val",
        imgsz=config["imgsz"],
        device=train_args["device"],
    )

    results_payload = {
        "model": config["model"],
        "epochs": config["epochs"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "seed": config["seed"],
        "patience": config["patience"],
        "smoke_test": is_smoke,
        "mAP50": val_metrics["mAP50"],
        "mAP50-95": val_metrics["mAP50-95"],
        "precision": val_metrics["precision"],
        "recall": val_metrics["recall"],
        "per_class_ap": val_metrics["per_class_ap"],
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
    }

    results_json_path = save_results(results_payload, run_dir / "results.json")
    save_results(results_payload, Path("results.json"))

    return {
        "status": "success",
        "run_dir": str(run_dir.resolve()),
        "results_json": str(results_json_path),
        "metrics": val_metrics,
        "best_checkpoint": best_checkpoint,
    }


def build_comparison_table(
    experiments_data: Dict[str, Any] | List[Dict[str, Any]] | str | Path,
    output_csv: Optional[str | Path] = None,
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Builds a structured comparison summary across multiple training experiment runs and exports to CSV.

    Inputs:
        experiments_data (Dict[str, Any] | List[Dict[str, Any]] | str | Path): Results mapping, list of results, or directory containing results.json files.
        output_csv (Optional[str | Path]): Filepath to write the resulting comparison CSV table.
        class_names (Optional[List[str]]): List of defect class names.

    Outputs:
        List[Dict[str, Any]]: List of flattened experiment metric dictionaries suitable for tabular analysis.
    """
    classes = class_names or DEFAULT_CLASSES
    entries: List[Tuple[str, Dict[str, Any]]] = []

    if isinstance(experiments_data, (str, Path)):
        p = Path(experiments_data)
        if p.is_dir():
            for json_file in sorted(p.rglob("results.json")):
                exp_name = json_file.parent.name
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        entries.append((exp_name, json.load(f)))
                except Exception:
                    continue
        elif p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                entries.append((p.parent.name, json.load(f)))
    elif isinstance(experiments_data, dict):
        for key, val in experiments_data.items():
            if isinstance(val, (str, Path)) and Path(val).is_file():
                with open(val, "r", encoding="utf-8") as f:
                    entries.append((key, json.load(f)))
            elif isinstance(val, dict):
                entries.append((key, val))
    elif isinstance(experiments_data, list):
        for idx, item in enumerate(experiments_data):
            if isinstance(item, dict):
                exp_name = item.get("experiment", item.get("name", f"exp_{idx}"))
                entries.append((exp_name, item))

    rows: List[Dict[str, Any]] = []
    for exp_name, data in entries:
        row: Dict[str, Any] = {
            "experiment": exp_name,
            "model": data.get("model", "unknown"),
            "imgsz": data.get("imgsz", 640),
            "epochs": data.get("epochs", 50),
            "precision": round(float(data.get("precision", 0.0)), 4),
            "recall": round(float(data.get("recall", 0.0)), 4),
            "mAP50": round(float(data.get("mAP50", 0.0)), 4),
            "mAP50-95": round(float(data.get("mAP50-95", 0.0)), 4),
        }

        per_class = data.get("per_class_ap", data.get("per_class", {}))
        for cls in classes:
            cls_data = per_class.get(cls, {})
            ap50 = cls_data.get("ap50", cls_data.get("AP50", 0.0))
            row[f"{cls}_AP50"] = round(float(ap50), 4)

        rows.append(row)

    if output_csv and rows:
        target_csv = Path(output_csv)
        target_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(target_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows


def plot_per_class_comparison(
    comparison_data: List[Dict[str, Any]] | str | Path,
    output_png: str | Path = "reports/phase2b/per_class_ap50_comparison.png",
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Generates a grouped bar chart comparing per-class AP50 scores across experiments.

    Inputs:
        comparison_data (List[Dict[str, Any]] | str | Path): Comparison row list, directory, or path to comparison.csv.
        output_png (str | Path): Destination file path for the exported comparison plot.
        class_names (Optional[List[str]]): List of defect class names.

    Outputs:
        str: Absolute filepath of the saved comparison chart.
    """
    import numpy as np

    classes = class_names or DEFAULT_CLASSES
    if isinstance(comparison_data, (str, Path)):
        p = Path(comparison_data)
        if p.is_file() and p.suffix == ".csv":
            rows: List[Dict[str, Any]] = []
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        else:
            rows = build_comparison_table(p)
    else:
        rows = comparison_data

    if not rows:
        raise ValueError("No comparison data provided to plot")

    exp_names = [r["experiment"] for r in rows]
    n_exp = len(exp_names)
    n_cls = len(classes)

    x = np.arange(n_cls)
    width = 0.8 / max(n_exp, 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, r in enumerate(rows):
        scores = [float(r.get(f"{c}_AP50", 0.0)) for c in classes]
        offset = (i - (n_exp - 1) / 2) * width
        ax.bar(x + offset, scores, width, label=r["experiment"])

    ax.set_ylabel("AP@50")
    ax.set_title("Phase 2b Controlled Experiments: Per-Class AP@50 Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    target = Path(output_png)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return str(target.resolve())


if __name__ == "__main__":
    run_training()
