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
| `scratches` | **0.952** | Top performer; possible explanation (untested): distinctive longitudinal aspect ratio |
| `patches` | **0.877** | High performer; possible explanation (untested): high contrast and uniform geometry |
| `pitted_surface` | **0.845** | Strong performer; possible explanation (untested): strong texture clustering and box overlap |
| `inclusion` | **0.813** | Solid localization; possible explanation (untested): small dark anomalies |
| `rolled-in_scale`| **0.622** | Weak class; possible explanation (untested): confused with background rolling grain |
| `crazing` | **0.419** | Weakest class / primary bottleneck; stays weak in both (recall 0.32 at conf=0.001 and 0.38 at conf=0.25) |

### Key Observations & Limitations
- **Confidence Threshold Effect**: At conf=0.25, overall recall is slightly higher (0.733 vs 0.713) and precision lower (0.690 vs 0.725). mAP is lower (0.668 vs 0.755) because truncating low-confidence detections shortens the precision-recall curve; the conf=0.001 figure is the standard mAP. Crazing stays weak in both (recall 0.32 and 0.38).
- **Weak Classes**: `crazing` (0.419 mAP50) and `rolled-in_scale` (0.622 mAP50) represent the primary detection bottlenecks.
- **Sample Size Limitation**: The validation split contains only **36 images per class** (216 images total), meaning single misclassifications carry noticeable percentage weight.
- **Single Seed**: Results reflect a single deterministic run (`seed=42`).
- **Confusion Matrix Note**: We do not use the Ultralytics confusion matrix for conclusions.

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
- **No Clear Improvement Over Baseline**: None of the interventions showed a clear improvement over the E0 baseline (0.755 mAP50).
- **E1 (320×320) Matched the Baseline Within Noise at Lower Cost**: E1 achieved 0.758 mAP50 with 4× smaller pixel processing cost, matching the baseline within noise; a possible explanation (untested) is that downsampling closer to the native 200×200 source resolution preserves feature quality without upsampling blur.
- **Larger Model and Augmentation Bundle Did Not Help**: E2 (YOLOv8m, 25.9M params) dropped crazing AP50 to 0.326; a possible explanation (untested) is overfitting on 1,224 training images. E3 (crazing AP50 0.340, rolled-in scale 0.594) did not improve performance; a possible explanation (untested) is that heavy spatial and photometric augmentations (rotation, scale jitter, HSV shifts) may degrade subtle texture cues.
- **Crazing Bottleneck Persists (0.33–0.43 AP50)**: Crazing stayed at 0.33–0.43 in every single run; a possible explanation (untested) is an underlying data or labeling limit (such as faint contrast, ambiguous boundary annotations, or label noise) to be checked in error analysis.
- **Exact Reproducibility**: E0 reproduced Phase 2a exactly (0.7545 mAP50, 0.4301 mAP50-95, 0.4189 crazing AP50).
- **Statistical Context**: Results come from a single deterministic seed (`seed=42`) with only 36 images per class in validation, so small metric gaps are noise.


---

## Phase 3: Decision Layer

Phase 3 implements an operational decision layer that maps raw bounding box detections to factory actions through an explicit cost model, severity scoring, per-class threshold calibration, and optional human-in-the-loop review bands. The entire evaluation protocol was pre-registered prior to running inference on the reserved test split.

### Decision Layer Design

1. **Cost Model Formulation**:
   - Total industrial cost evaluates missed flaws, false alarms, and inspection overhead:
     $$\text{Total Cost} = \sum_{\text{missed defects}} (\text{cost\_miss} \times \text{severity}) + \sum_{\text{false alarms}} \text{cost\_false\_alarm} + \sum_{\text{reviewed boxes}} \text{cost\_review}$$
   - **Baseline Cost Weights** (`configs/cost.yaml`): $\text{cost\_miss} = 10$, $\text{cost\_false\_alarm} = 1$, $\text{cost\_review} = 0.3$.
   - **Matching Protocol**: Class-aware greedy matching at $\text{IoU} \ge 0.5$, sorted by confidence descending. Duplicate detections matching an already-claimed ground truth count as false alarms.
2. **Severity-to-Action Mapping**:
   - Defect severity is computed as $\text{severity} = \text{class\_weight} \times \text{area\_factor}$, where area factor scales linearly with box area fraction from $0.4$ (small defect, $\le 2\%$ image area) to $2.0$ (large defect, $\ge 20\%$ image area).
   - If detection confidence falls in a review band, action is `human_review`.
   - If detection confidence meets the alert threshold, severity triggers automated factory responses:
     - $\text{severity} < 0.5$: `ignore` (minor surface blemish)
     - $0.5 \le \text{severity} < 0.8$: `monitor` (log defect trend)
     - $0.8 \le \text{severity} < 1.3$: `schedule_service` (flag roll inspection)
     - $\text{severity} \ge 1.3$: `stop_now` (immediate line emergency halt)
3. **Optional Review Band & Workload Budget**:
   - Per-class confidence band $[\text{lower}, \text{upper})$: detections with $\text{conf} \ge \text{upper}$ alert directly; $\text{lower} \le \text{conf} < \text{upper}$ route to a human inspector; $\text{conf} < \text{lower}$ are ignored.
   - Reviewer accuracy $a$: symmetric model where a human reviewer correctly verifies defects with probability $a$ ($a=0.9$ primary, $a=1.0$ and $a=0.8$ secondary).
   - Workload constraint: average inspection budget $B \le 3$ (combined alerts + reviews per image $\le 3$ on validation set).

### Pre-Registered Primary Outcome (Not Supported on Val and Test)

Prior to running on the test set (`reports/phase3/preregistration.md`), we pre-registered the primary hypothesis: *a tuned per-class review band policy ($a=0.9, B \le 3$) delivers statistically significant cost savings over a matched flat alerting baseline ($0.127$).* The pre-registered success criterion required the 95% bootstrap confidence interval of cost saving to lie strictly above zero.

- **Validation Split (216 images)**:
  - Primary policy cost: **1610.7** vs matched flat cost: **1688.1**.
  - Cost saving: **77.4** (**4.6%**), but 95% bootstrap CI was **`[-33.7, 194.3]`** (percentage CI **`[-2.1%, 11.1%]`**).
  - *Result*: The interval crosses zero (lower bound **−33.7**), so the primary outcome was **not supported on validation**.
- **Test Split (360 images, 854 ground truth defects)**:
  - Primary policy cost: **3052.7** (cost/image **8.480** `[7.321, 9.709]`, recall **0.760** `[0.731, 0.788]`, alerts/image **2.525**, reviews/image **0.833**).
  - Matched flat (0.127) cost: **3041.9** (cost/image **8.450** `[7.248, 9.670]`, recall **0.785** `[0.756, 0.812]`, alerts/image **3.400**, reviews/image **0.000**).
  - Cost saving vs matched flat: **−10.8** (**−0.4%**, 95% CI **`[-225.7, 193.3]`**, percentage CI **`[-7.8%, 6.2%]`**).
  - *Result*: The primary outcome was **not supported on test**; the matched flat baseline achieved slightly lower total cost, and the difference is well within statistical noise.

### Test Split Results Table (360 Images)

| Policy | Total Cost | Cost / Image (95% CI) | Recall (95% CI) | Alerts / Img | Reviews / Img | Saving vs Flat 0.25 (%) | Saving vs Matched Flat (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `primary (a=0.9, B<=3)` | **3052.7** | 8.480 [7.321, 9.709] | 0.760 [0.731, 0.788] | 2.525 | 0.833 | **857.9** (21.9% [15.7, 27.8]) | **-10.8** (-0.4% [-7.8, 6.2]) |
| `matched flat (0.127)` | **3041.9** | 8.450 [7.248, 9.670] | 0.785 [0.756, 0.812] | 3.400 | 0.000 | **868.7** (22.2% [16.1, 28.2]) | **0.0** (reference) |
| `secondary a=1.0` | **2791.3** | 7.754 [6.631, 8.948] | 0.767 [0.738, 0.795] | 1.436 | 1.922 | **1119.3** (28.6% [22.1, 34.9]) | **250.6** (8.2% [1.0, 14.9]) |
| `secondary a=0.8` | **3065.7** | 8.516 [7.344, 9.754] | 0.766 [0.737, 0.795] | 3.294 | 0.064 | **844.9** (21.6% [14.7, 28.1]) | **-23.8** (-0.8% [-8.3, 6.0]) |
| `flat 0.25` | **3910.6** | 10.863 [9.363, 12.399] | 0.700 [0.665, 0.735] | 2.425 | 0.000 | **0.0** (reference) | **-868.7** (-28.6% [-39.3, -19.1]) |
| `no alerts` | **11549.4** | 32.082 [30.540, 33.607] | 0.000 [0.000, 0.000] | 0.000 | 0.000 | **-7638.8** (-195.3%) | **-8507.5** (-279.7%) |

### Main Finding

1. **Cost-Tuned Thresholds Beat Default 0.25 by ~22%**:
   - Aligning detection thresholds with asymmetric factory costs (where missing a defect costs 10× a false alarm) produced a substantial **21.9% to 22.2% cost reduction** on the test set compared to the standard `0.25` threshold (saving **857.9** to **868.7** cost units).
2. **A Single Matched Global Threshold Matches Complex Policies**:
   - A single global threshold of **`0.127`** (matched on validation alert workload) achieved a total test cost of **3041.9**, matching the primary per-class review-band policy (**3052.7**) within noise.
   - **Practical Takeaway**: In this industrial inspection setting, optimizing a single operating point captures all economic benefits without the operational friction, latency, and maintenance burden of per-class review bands.

### Secondary Result & Caveats

- **Infallible Reviewer Scenario ($a=1.0$)**:
  - If human review is assumed to be 100% accurate ($a=1.0$), the policy achieves a test cost of **2791.3**, saving **250.6** (**8.2%**, 95% CI `[28.5, 465.2]`, `[1.0%, 14.9%]`) over matched flat alerting.
- **Caveats**:
  - This benefit is brittle and depends entirely on the unrealistic assumption of perfect human inspection ($a=1.0$).
  - Once human error is modeled ($a=0.9$), the advantage drops to **−0.4%** (`[-7.8%, 6.2%]`).
  - At $a=0.8$, the policy barely utilizes review (only **0.064** reviews/image, test cost **3065.7**, saving **−0.8%**, 95% CI `[-8.3%, 6.0%]`) because human errors incur expensive false alarm and miss penalties that outweigh the review band's utility.

### Sensitivity Summary

A 24-condition grid sweep across miss-to-false-alarm cost ratios (2, 3, 5, 10, 20, 50) and review costs (0.1, 0.3, 0.6, 1.0) on validation revealed:
Review bands help significantly only when review cost is ≤ 0.3 with a miss-to-false-alarm ratio ≤ 5, or review cost 0.1 at ratio 10; at ratios ≥ 20 or review cost ≥ 0.6 the savings are not distinguishable from zero.

### Methodological Limitations

1. **Illustrative Cost Parameters**: Costs (10 miss / 1 false alarm / 0.3 review) are illustrative engineering benchmarks rather than site-specific plant accounting data.
2. **Absence of Defect-Free Images**: Every image in the NEU-DET database contains at least one defect. False alarm rates on pristine, flaw-free steel sheets cannot be measured on this dataset.
3. **Prior Validation Split Exposure**: The validation set was utilized in Phase 2 for baseline verification and experimental comparison prior to Phase 3 threshold tuning.
4. **Single Test Evaluation**: The test set (360 images) was strictly held out and evaluated exactly once using frozen pre-registered thresholds without retuning.

---

## Phase 4: Error Analysis and Robustness

Phase 4 diagnoses the primary failure modes on the validation split (e0 baseline, single global confidence threshold 0.127) through detailed error categorisation, IoU sensitivity sweeps, and synthetic perturbation stress tests.

### 1. Defect Error Analysis (Val Split, Threshold 0.127)

- **Crazing Bottleneck & Bounding Box Ambiguity**:
  - Evaluation of the 84 ground truth crazing boxes on validation: **43 matched (TP)**, **37 poorly localized** (best same-class IoU between 0.15 and 0.50, median 0.34), **4 missed** (no kept detection with IoU > 0.1), and **0 wrong-class** confusions.
  - Across IoU evaluation thresholds, crazing AP drops steeply: **AP30 is 0.688**, **AP50 is 0.425**, and **AP75 collapses to 0.04** (0.037).
  - Ground truth crazing boxes are large (median area fraction 0.20) and frequently overlap sibling annotations in the same image (33.3% overlap rate). This behavior is **consistent with box ambiguity**: the detector consistently locates the crazing areas, but bounding rectangular boundaries on sprawling, diffuse micro-crack networks are inherently subjective.
  - Annotation noise and model weakness cannot be separated with one annotator and no repeat labels.
- **Inclusion False Alarms**:
  - Out of 187 inclusion detections at threshold 0.127, 50 detections (**about 27%**, 26.7%) are true background false alarms (IoU < 0.1 with any ground truth defect). This is the highest background false alarm share of any defect class.

### 2. Synthetic Perturbation Robustness (Val Split, One Model)

We evaluated baseline model stability under 5 synthetic perturbation families (Gaussian blur, Gaussian noise, JPEG compression, brightness, and contrast) on all 216 validation images:
- **Evaluation Settings**: mAP50 uses all detections down to confidence floor 0.001. Recall and alerts per image use the operational 0.127 threshold.
- **Clean Baseline Note**: Clean mAP50 is **0.760** (0.7599) in this single-image evaluation pipeline versus **0.7545** from Ultralytics `model.val` in Phase 2a. The 0.004 gap is unexplained and small (batched val letterboxing differences vs single-image predict, plus 494 vs 495 ground truth boxes parsed), and all robustness deltas are measured against the same cached clean run.
- **Fragile to Blur, Noise, and JPEG**:
  - The model degrades under Gaussian blur ($\Delta \text{mAP50} = -0.236$ at $\sigma=1.0$, 95% CI `[-0.274, -0.195]`), additive noise ($\Delta \text{mAP50} = -0.203$ at $\sigma=0.02$, 95% CI `[-0.245, -0.154]`), and JPEG compression ($\Delta \text{mAP50} = -0.124$ at quality 50, 95% CI `[-0.154, -0.086]`). All paired bootstrap confidence intervals are strictly away from zero.
- **Robust to Brightness**:
  - The detector is robust to overall illumination shifts ($0.6\times$ to $1.4\times$ brightness), with mAP50 deltas of only $-0.003$ to $-0.014$, all with confidence intervals crossing zero.
- **Contrast Shifts & Threshold Instability**:
  - At contrast 1.4, mAP50 drop is small ($-0.025$, 95% CI `[-0.041, -0.008]`), but operational alerts increase by **+0.4 alerts per image** (3.38 vs 2.99 clean, $\Delta = +0.389$, 95% CI `[+0.264, +0.514]`) with no recall gain ($0.794$ vs $0.782$, $\Delta = +0.012$, 95% CI `[-0.009, 0.033]`, crossing zero).
  - The mAP50 drop is small but recall and alerts move, which suggests confidences crossing the fixed threshold and not worse ranking. A fixed threshold is not stable under these shifts.

### 3. Matching IoU Sensitivity (Exploratory, Val Only)

- When the matching threshold is lowered from IoU 0.5 to IoU 0.3 on validation, matched flat recall rises from **0.780** (`[0.739, 0.820]`) to **0.883** (`[0.851, 0.914]`).
- The optimal flat threshold barely moves (**0.127** at IoU 0.5 vs **0.128** at IoU 0.3) because the alert budget ($B \le 3$ alerts/image) sets it.
- The shipped policy and Phase 3 test results are unchanged.

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
- [x] **Phase 3**: Decision layer (severity, cost model, thresholds).
- [x] **Phase 4**: Robustness, explainability and error analysis.
- [ ] **Phase 5**: MLOps (MLflow, DVC, CI/CD, Docker, FastAPI).
- [ ] **Phase 6**: Data platform, simulator and live dashboard.
- [ ] **Phase 7**: Drift monitoring, model card and demo.
