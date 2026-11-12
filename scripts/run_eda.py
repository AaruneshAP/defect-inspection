"""
Exploratory Data Analysis script and figure generator.

Purpose: Analyzes dataset distributions, generates visual plots, and saves inspection figures to reports/figures/.
Inputs: YOLO formatted dataset directories, label text files, and image files.
Outputs: Exported PNG figures in reports/figures/ covering split distributions, box sizes, sample grids, and pixel intensities.
"""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def ensure_output_dir(dir_path: str = "reports/figures") -> Path:
    """
    Ensures that the output directory for figures exists.

    Inputs:
        dir_path (str): Destination path for reports and figures.

    Outputs:
        Path: Verified Path object for the target directory.
    """
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_class_and_split_distribution(yolo_dir: str = "data/yolo", out_dir: Path = Path("reports/figures")) -> None:
    """
    Plots the count of images per class across train, val, and test splits.

    Inputs:
        yolo_dir (str): Root path to YOLO dataset.
        out_dir (Path): Output directory for saved figure.

    Outputs:
        None: Saves 01_class_split_distribution.png.
    """
    splits = ["train", "val", "test"]
    data_counts = {split: {cls: 0 for cls in CLASS_NAMES} for split in splits}

    for split in splits:
        img_dir = Path(yolo_dir) / "images" / split
        for img_p in img_dir.glob("*.jpg"):
            for cls in CLASS_NAMES:
                if img_p.name.startswith(cls + "_"):
                    data_counts[split][cls] += 1
                    break

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    ax.bar(x - width, [data_counts["train"][c] for c in CLASS_NAMES], width, label="Train (85%)", color="#2b5c8f")
    ax.bar(x, [data_counts["val"][c] for c in CLASS_NAMES], width, label="Val (15%)", color="#e76f51")
    ax.bar(x + width, [data_counts["test"][c] for c in CLASS_NAMES], width, label="Test (Reserved)", color="#2a9d8f")

    ax.set_ylabel("Number of Images")
    ax.set_title("NEU-DET Defect Dataset: Class Distribution Across Splits")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=15)
    ax.set_ylim(0, 275)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "01_class_split_distribution.png", dpi=300)
    plt.close(fig)


def plot_boxes_per_image(yolo_dir: str = "data/yolo", out_dir: Path = Path("reports/figures")) -> None:
    """
    Plots the distribution of bounding box counts per image.

    Inputs:
        yolo_dir (str): Root path to YOLO dataset.
        out_dir (Path): Output directory for saved figure.

    Outputs:
        None: Saves 02_boxes_per_image_distribution.png.
    """
    counts: List[int] = []
    lbl_dir = Path(yolo_dir) / "labels"
    for lbl_p in lbl_dir.rglob("*.txt"):
        lines = [line.strip() for line in lbl_p.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts.append(len(lines))

    counter = Counter(counts)
    x = sorted(counter.keys())
    y = [counter[k] for k in x]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, y, color="#457b9d", edgecolor="#1d3557", width=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Number of Bounding Boxes per Image")
    ax.set_ylabel("Image Count")
    ax.set_title(f"Distribution of Defect Instances per Image (Mean: {np.mean(counts):.2f})")
    ax.set_xticks(x)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "02_boxes_per_image_distribution.png", dpi=300)
    plt.close(fig)


def plot_box_sizes_and_aspect_ratios(yolo_dir: str = "data/yolo", out_dir: Path = Path("reports/figures")) -> None:
    """
    Plots box width, height, and aspect ratio distributions grouped by defect class.

    Inputs:
        yolo_dir (str): Root path to YOLO dataset.
        out_dir (Path): Output directory for saved figure.

    Outputs:
        None: Saves 03_box_sizes_and_aspect_ratios.png.
    """
    box_w = {i: [] for i in range(len(CLASS_NAMES))}
    box_h = {i: [] for i in range(len(CLASS_NAMES))}
    aspects = {i: [] for i in range(len(CLASS_NAMES))}

    lbl_dir = Path(yolo_dir) / "labels"
    for lbl_p in lbl_dir.rglob("*.txt"):
        lines = [l.strip() for l in lbl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        for l in lines:
            cls_id, xc, yc, w, h = (float(v) for v in l.split())
            cls_idx = int(cls_id)
            box_w[cls_idx].append(w * 200)
            box_h[cls_idx].append(h * 200)
            aspects[cls_idx].append(w / max(h, 1e-5))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.boxplot([box_w[i] for i in range(len(CLASS_NAMES))], tick_labels=CLASS_NAMES, patch_artist=True)
    ax1.set_title("Bounding Box Width Distribution (Pixels)")
    ax1.set_ylabel("Width (px)")
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.boxplot([aspects[i] for i in range(len(CLASS_NAMES))], tick_labels=CLASS_NAMES, patch_artist=True)
    ax2.set_title("Aspect Ratio (Width / Height) Distribution")
    ax2.set_ylabel("Aspect Ratio")
    ax2.set_yscale("log")
    ax2.tick_params(axis="x", rotation=25)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "03_box_sizes_and_aspect_ratios.png", dpi=300)
    plt.close(fig)


def plot_sample_defects_grid(yolo_dir: str = "data/yolo", out_dir: Path = Path("reports/figures")) -> None:
    """
    Generates a 6x3 visual grid of sample images with overlayed ground truth bounding boxes.

    Inputs:
        yolo_dir (str): Root path to YOLO dataset.
        out_dir (Path): Output directory for saved figure.

    Outputs:
        None: Saves 04_sample_defects_grid.png.
    """
    fig, axes = plt.subplots(len(CLASS_NAMES), 3, figsize=(10, 18))

    for row_idx, cls_name in enumerate(CLASS_NAMES):
        img_candidates = sorted(list((Path(yolo_dir) / "images" / "train").glob(f"{cls_name}_*.jpg")))[:3]
        for col_idx, img_p in enumerate(img_candidates):
            ax = axes[row_idx, col_idx]
            img = Image.open(img_p).convert("RGB")
            ax.imshow(img)

            lbl_p = Path(yolo_dir) / "labels" / "train" / f"{img_p.stem}.txt"
            if lbl_p.exists():
                lines = [l.strip() for l in lbl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
                for l in lines:
                    cls_id, xc, yc, w, h = (float(v) for v in l.split())
                    box_w = w * 200
                    box_h = h * 200
                    xmin = (xc * 200) - (box_w / 2.0)
                    ymin = (yc * 200) - (box_h / 2.0)
                    rect = patches.Rectangle((xmin, ymin), box_w, box_h, linewidth=2,
                                             edgecolor=CLASS_COLORS[int(cls_id)], facecolor="none")
                    ax.add_patch(rect)

            ax.set_title(f"{img_p.name}", fontsize=9)
            ax.axis("off")

    fig.suptitle("Ground-Truth Defect Annotations (3 Samples per Class)", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(out_dir / "04_sample_defects_grid.png", dpi=300)
    plt.close(fig)


def plot_pixel_intensity_distribution(yolo_dir: str = "data/yolo", out_dir: Path = Path("reports/figures")) -> None:
    """
    Plots mean and standard deviation of grayscale pixel intensities across defect classes.

    Inputs:
        yolo_dir (str): Root path to YOLO dataset.
        out_dir (Path): Output directory for saved figure.

    Outputs:
        None: Saves 05_pixel_intensity_distribution.png.
    """
    means = []
    stds = []
    img_dir = Path(yolo_dir) / "images" / "train"

    for cls in CLASS_NAMES:
        cls_imgs = list(img_dir.glob(f"{cls}_*.jpg"))
        cls_means = []
        cls_stds = []
        for p in cls_imgs:
            arr = np.array(Image.open(p).convert("L"), dtype=np.float32)
            cls_means.append(arr.mean())
            cls_stds.append(arr.std())
        means.append(np.mean(cls_means))
        stds.append(np.mean(cls_stds))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    ax.bar(x - width/2, means, width, label="Mean Pixel Intensity", color="#2a9d8f")
    ax.bar(x + width/2, stds, width, label="Mean Contrast / Std Dev", color="#e76f51")

    ax.set_ylabel("Pixel Value (0-255)")
    ax.set_title("Pixel Intensity and Local Contrast by Defect Class")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=20)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "05_pixel_intensity_distribution.png", dpi=300)
    plt.close(fig)


def run_all_eda() -> None:
    """
    Orchestrates figure generation and verification.

    Inputs:
        None.

    Outputs:
        None: Generates all EDA figures in reports/figures/.
    """
    out_dir = ensure_output_dir()
    plot_class_and_split_distribution("data/yolo", out_dir)
    plot_boxes_per_image("data/yolo", out_dir)
    plot_box_sizes_and_aspect_ratios("data/yolo", out_dir)
    plot_sample_defects_grid("data/yolo", out_dir)
    plot_pixel_intensity_distribution("data/yolo", out_dir)


if __name__ == "__main__":
    run_all_eda()
