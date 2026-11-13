# Defect Inspection System

An automated computer vision inspection system designed to detect, classify, and localize manufacturing surface defects on production lines.

---

## Project Overview

In industrial manufacturing, manual visual inspection is slow, fatigue-prone, and inconsistent. The Defect Inspection System provides an end-to-end pipeline that takes raw camera images of manufactured components, detects visual flaws (scratches, dents, cracks, voids, discolorations), classifies defect types, and exposes inspection results through a low-latency API service.

---

## Dataset

- **Source & Authors**: The dataset used is the **NEU Surface Defect Database (NEU-DET)**, introduced by Kechen Song and Yunhui Yan (Northeastern University, Shenyang, China) in their research on hot-rolled steel strip surface defect detection.
- **Classes**: 6 balanced surface defect categories (300 images per class, 1,800 images total at 200×200 resolution):
  1. `crazing`
  2. `inclusion`
  3. `patches`
  4. `pitted_surface`
  5. `rolled-in_scale`
  6. `scratches`
- **Licensing Notice**: On Kaggle, the dataset license is listed as **"Unknown"**. Consequently, **the raw data is not redistributed in this repository** and is strictly excluded via `.gitignore`.
- **Downloading the Data**:
  Download the dataset archive from Kaggle and place/extract it into `data/`:
  ```bash
  # Using Kaggle CLI:
  kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database -p data/
  # Or place archive.zip into data/ and extract to data/archive/
  ```

---

## Phase 1: Data Preparation

Phase 1 provides complete Pascal VOC XML annotation parsing, coordinate normalization and boundary clamping, class-stratified dataset splitting, and YOLO formatting.

### Split Sizes
The provided dataset is partitioned into three strictly isolated splits:
- **Train Split (85% of provided train set)**: **1,224 images & labels** (exactly 204 images per class).
- **Validation Split (15% of provided train set)**: **216 images & labels** (exactly 36 images per class).
- **Test Split (Reserved evaluation set)**: **360 images & labels** (exactly 60 images per class, derived from the original validation partition and untouched during training or hyperparameter tuning).

### Duplicate Image Detection
An MD5 cryptographic hash audit across all 1,800 images detected **one identical duplicate image pair**:
- **Duplicate Files**: `patches_101.jpg` and `patches_105.jpg` (MD5 digest: `24f39e0f91f95e4c785052ac4d3e49a6`).
- **Leakage Prevention**: Both images reside exclusively within `images/train/`, preventing data leakage into validation or test sets.

### Running the YOLO Conversion
To convert raw Pascal VOC XML annotations to normalized YOLO format and generate `data/yolo/` with `data.yaml`:
```bash
python -c "from src.data import convert_dataset_to_yolo; convert_dataset_to_yolo(data_dir='data', output_dir='data/yolo', val_ratio=0.15, seed=42)"
```

---

## Phase 2a Results

Baseline model evaluation on the validation split (**216 images**, exactly 36 images per class) using **YOLOv8s** (`imgsz=640`, `seed=42`, trained for 50 epochs on Kaggle GPU):

### Overall Validation Performance

| Metric | Standard Evaluation (`conf=0.001`) | Operational Threshold (`conf=0.25`) |
| :--- | :--- | :--- |
| **mAP@50** | **0.755** | **0.668** |
| **mAP@50-95** | **0.430** | — |
| **Precision** | **0.725** | **0.690** |
| **Recall** | **0.713** | **0.733** |

### Per-Class mAP@50 Breakdown (Standard Evaluation, `conf=0.001`)

| Defect Class | mAP@50 | Operational Note (`conf=0.25`) |
| :--- | :--- | :--- |
| `scratches` | **0.952** | Top performer; distinctive longitudinal aspect ratio |
| `patches` | **0.877** | High contrast, uniform defect geometry |
| `pitted_surface` | **0.845** | Strong texture clustering and box overlap |
| `inclusion` | **0.813** | Solid localization on small dark anomalies |
| `rolled-in_scale`| **0.622** | Weak class; confused with background rolling grain |
| `crazing` | **0.419** | Weakest class / primary bottleneck; stays weak in both (recall 0.32 at conf=0.001 and 0.38 at conf=0.25) |

### Key Observations & Limitations
- **Confidence Threshold Effect**: At conf=0.25, overall recall is slightly higher (0.733 vs 0.713) and precision lower (0.690 vs 0.725). mAP is lower (0.668 vs 0.755) because truncating low-confidence detections shortens the precision-recall curve; the conf=0.001 figure is the standard mAP. Crazing stays weak in both (recall 0.32 and 0.38).
- **Weak Classes**: `crazing` (0.419 mAP50) and `rolled-in_scale` (0.622 mAP50) represent the primary detection bottlenecks.
- **Sample Size Limitation**: The validation split contains only **36 images per class** (216 images total), meaning single misclassifications carry noticeable percentage weight.
- **Single Seed**: Results reflect a single deterministic run (`seed=42`).
- **Confusion Matrix Note**: The Ultralytics confusion matrix is **not used for conclusions** because Ultralytics computes its confusion matrix at an arbitrary fixed confidence threshold rather than evaluating across the complete precision-recall curve.

---

## Phase 2b Results

Controlled ablation experiments to improve performance on weak defect classes (`crazing` and `rolled-in_scale`), evaluated strictly on the validation split (**216 images**, exactly 36 images per class, `seed=42`, 50 epochs on Kaggle GPU):

| experiment | model | imgsz | epochs | precision | recall | mAP50 | mAP50-95 | crazing_AP50 | inclusion_AP50 | patches_AP50 | pitted_surface_AP50 | rolled-in_scale_AP50 | scratches_AP50 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| e0_baseline | yolov8s.pt | 640 | 50 | 0.7248 | 0.7134 | 0.7545 | 0.4301 | 0.4189 | 0.8132 | 0.8772 | 0.8446 | 0.6215 | 0.9517 |
| e1_imgsz320 | yolov8s.pt | 320 | 50 | 0.6999 | 0.7372 | 0.7578 | 0.43 | 0.4311 | 0.8209 | 0.8884 | 0.8639 | 0.632 | 0.9104 |
| e2_yolov8m_640 | yolov8m.pt | 640 | 50 | 0.6593 | 0.7227 | 0.7404 | 0.4375 | 0.3262 | 0.8208 | 0.8896 | 0.8441 | 0.6311 | 0.9305 |
| e3_augmented | yolov8s.pt | 640 | 50 | 0.7058 | 0.6594 | 0.7261 | 0.4068 | 0.3397 | 0.8202 | 0.8642 | 0.8419 | 0.5941 | 0.8965 |

### Summary & Honest Empirical Analysis
- **No Experiment Clearly Beat the Baseline**: None of the interventions achieved a statistically significant breakthrough over the E0 baseline (0.755 mAP50).
- **E1 (320×320) Matches Baseline at Lower Cost**: E1 achieved 0.758 mAP50 with 4× smaller pixel processing cost, proving that downsampling closer to the native 200×200 source resolution preserves feature quality without upsampling blur.
- **Larger Model and Augmentation Bundle Did Not Help**: E2 (YOLOv8m, 25.9M params) dropped crazing AP50 to 0.326 due to overfitting on 1,224 training images. E3's heavy spatial and photometric augmentations (rotation, scale jitter, HSV shifts) degraded subtle texture cues (crazing AP50 0.340, rolled-in scale 0.594).
- **Crazing Bottleneck Persists (0.33–0.43 AP50)**: Crazing stayed at 0.33–0.43 in every single run, which suggests a data or labeling limit (such as faint contrast, ambiguous boundary annotations, or label noise) to be checked in error analysis.
- **Exact Reproducibility**: E0 reproduced Phase 2a exactly (0.7545 mAP50, 0.4301 mAP50-95, 0.4189 crazing AP50).
- **Statistical Context**: Results come from a single deterministic seed (`seed=42`) with only 36 images per class in validation, so small metric gaps are noise.


---

## Architecture & Directory Structure

```text
defect-inspection/
├── data/                  # Raw and YOLO formatted imagery (ignored in git)
│   ├── archive/           # Raw NEU-DET images and Pascal VOC XML annotations
│   └── yolo/              # Processed YOLO dataset (images, labels, data.yaml)
├── notebooks/             # Exploratory data analysis notebooks
│   └── 01_eda.ipynb       # Executed EDA notebook with embedded visualizations
├── reports/
│   └── figures/           # Exported high-resolution figures from EDA
├── src/                   # Production-grade core machine learning pipeline
│   ├── __init__.py        # Package exports
│   ├── data.py            # Dataset loading, VOC parsing, splitting, YOLO export
│   ├── train.py           # Training loop orchestration, loss tracking, and checkpointing
│   └── inference.py       # Model loading, single-sample and batch prediction engine
├── app/                   # Serving layer and user interface / REST API
│   ├── __init__.py        # Application package exports
│   └── main.py            # API service endpoints (/health, /inspect)
├── tests/                 # Automated test suite
│   ├── __init__.py        # Tests package exports
│   ├── test_data.py       # VOC parsing, YOLO coordinate bounds, and split tests
│   └── test_train.py      # Configuration, metric parsing, and training safety tests
├── scripts/
│   ├── bootstrap.sh       # One-click environment provisioning script
│   └── run_eda.py         # Headless EDA figure generation script
├── requirements.txt       # Project dependencies
├── .gitignore             # Ignored directories (data, models, internal learning notes)
└── README.md              # Project documentation and developer guide
```

---

## Module Breakdown

1. **`src/data.py`**
   - **Purpose**: Ingests raw images and annotations, performs normalization/resizing, parses Pascal VOC XMLs, converts to YOLO format, produces stratified splits, and discovers Kaggle datasets.
   - **Key Functions**: `parse_voc_xml`, `voc_to_yolo_bbox`, `scan_dataset`, `split_dataset`, `convert_dataset_to_yolo`, `load_dataset`, `preprocess_image`, `find_kaggle_dataset`.

2. **`src/train.py`**
   - **Purpose**: Coordinates YOLOv8s neural network training, validation metrics parsing, checkpoint preservation, and result exporting.
   - **Key Functions**: `load_config`, `parse_training_metrics`, `evaluate`, `save_checkpoint`, `save_results`, `run_training`.

3. **`src/inference.py`**
   - **Purpose**: Executes optimized model inference on unseen images to produce defect verdicts and confidence scores.
   - **Key Functions**: `load_model`, `predict`, `batch_predict`.

4. **`app/main.py`**
   - **Purpose**: Exposes HTTP endpoints for external inspection triggers and monitoring dashboards.
   - **Key Functions**: `health_check`, `inspect_image`, `create_app`.

5. **`tests/test_data.py` & `tests/test_train.py`**
   - **Purpose**: Validates coordinate clamping within $[0.0, 1.0]$, split sizes, disjoint partitions, configuration validation, metric extraction, and evaluation split protection.

---

## Quickstart Guide

### 1. Environment Setup

Using the bootstrap script:
```bash
bash scripts/bootstrap.sh
```

Or manually with Python:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
pytest tests/
```

Or with Python:
```bash
python -m pytest tests/
```

---

## Development Roadmap

- [x] **Phase 1**: Project scaffolding, interface definitions, Pascal VOC to YOLO conversion, stratified splitting, and EDA.
- [x] **Phase 2a**: YOLOv8s baseline model, Kaggle cloud training workflow, evaluation on val split, and metrics baseline.
- [x] **Phase 2b**: Controlled experiments (resolution, model capacity, augmentations) and ablation comparison.
- [ ] **Phase 3**: Inference engine, per-class confidence threshold calibration, and factory API.
- [ ] **Phase 4**: Production serving, final evaluation on reserved test split, and documentation.
