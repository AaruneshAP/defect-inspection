"""
Defect inspection YOLOv8 baseline training, evaluation, and metrics extraction module.

Purpose: Manages configuration loading, YOLOv8s baseline training with built-in Ultralytics losses, validation evaluation, metrics parsing (mAP50, mAP50-95, precision, recall, per-class AP), checkpoint preservation, and result artifact exportation.
Inputs: YAML configuration files, YOLO model backbones, dataset specifications, and training hyperparameters.
Outputs: Trained model weights in models/, parsed validation metrics in results.json, and evaluation curve figures.

Image Size Selection Rationale:
The primary baseline uses an input resolution of 640x640 (imgsz=640), matching the native pre-training resolution of YOLOv8s. This maximizes feature resolution in the deeper feature pyramid network (FPN) layers to capture small, fine defect structures (such as scratches and micro-crazing). A secondary configuration variant uses 320x320 (imgsz=320) to compare detection accuracy against inference throughput. Both resolutions are exact multiples of the backbone stride (32), ensuring zero fractional coordinate padding artifacts.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
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


def run_training(
    config_path: str = "configs/baseline.yaml",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Coordinates baseline YOLOv8 model training, validation scoring, and artifact saving.

    Inputs:
        config_path (str): Filepath to the YAML training configuration.
        kwargs (Any): Runtime parameter overrides.

    Outputs:
        Dict[str, Any]: Execution summary dictionary with best checkpoint path, metrics, and artifact locations.
    """
    config = load_config(config_path)
    config.update(kwargs)

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


if __name__ == "__main__":
    run_training()
