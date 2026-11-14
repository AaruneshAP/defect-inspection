---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - object-detection
  - yolov8
  - surface-defect-inspection
  - steel
  - neu-det
  - portfolio
---

# Model card: steel surface defect detector (YOLOv8s on NEU-DET)

> **Portfolio project on a public research dataset. Nothing here was validated on real production data.** Read the
> limitations section before drawing any conclusion about industrial use.

## Model details

- **Architecture:** Ultralytics YOLOv8s (11M parameters), fine-tuned from the COCO-pretrained `yolov8s.pt`, input 640x640.
- **Task:** detect and classify six hot-rolled steel surface defect types in a single grayscale image:
  `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.
- **Weights file:** `best.pt` (sha256 `dff41d9a7f7be82791563f74990d04ffe12545767f6c8bd4c62d1100fcece56b`, the file the
  Phase 3 pre-registration hashed).
- **Served with:** a FastAPI service (`app/`) and the hosted demo on Streamlit Community Cloud (https://defect-inspection.streamlit.app/), both applying the frozen decision policy below.
- **Licence:** AGPL-3.0, inherited from Ultralytics YOLOv8 and its pretrained weights. The dataset licence is unclear (see Data).

## Intended use

- **Intended:** a worked example of an end-to-end inspection pipeline: detection, a cost-based action policy,
  monitoring, drift alerts and a model-promotion gate. Useful for discussion, teaching and as a starting template.
- **Not intended:** deciding whether to stop a production line, releasing or rejecting steel, or any safety- or
  money-relevant decision. It has never seen a real production image, a healthy part, or a different camera.

## Data

- **Dataset:** NEU surface defect database (NEU-DET; Song and Yan, Northeastern University): 1,800 grayscale
  200x200 images, 300 per class, with Pascal VOC boxes.
- **Licence uncertainty:** the Kaggle listing gives the licence as "Unknown". The raw data is therefore not redistributed
  in the code repository. The weights published here were trained on it; if you need a clear licence chain for the
  training data, this model does not provide one. The model repository and demo show six small validation images for demonstration only.
- **Splits (class-stratified, seed 42):** the provided 1,440-image training partition is split 85/15 into
  **train 1,224** (204 per class) and **val 216** (36 per class). The provided 360-image validation partition
  (60 per class) is the **test split**, used once for the Phase 3 cost evaluation and now spent: nothing is tuned on it.
  One exact duplicate pair (`patches_101`, `patches_105`) sits entirely in train.
- Val is small (36 images per class), so small metric differences are noise.

## Training setup

YOLOv8s, 50 epochs, `imgsz` 640, batch 16, seed 42, early-stopping patience 15, lr0 0.001, lrf 0.01, optimizer
`auto`, one Kaggle GPU run (`configs/baseline.yaml`, experiment E0). One seed only; no seed variance was measured.

## Headline metrics (val split, 216 images)

| Metric | conf 0.001 (standard mAP) | conf 0.25 (operating point) |
|---|---|---|
| mAP50 | **0.755** | 0.668 |
| mAP50-95 | 0.430 | n/a |
| Precision | 0.725 | 0.690 |
| Recall | 0.713 | 0.733 |

Phase 2b tried three alternatives (E1 imgsz 320, E2 YOLOv8m, E3 heavy augmentation). **None clearly beat E0**: mAP50
0.758, 0.740 and 0.726 against 0.755. E1 matched E0 within noise at lower cost; E2 and E3 lowered crazing AP50.
The promotion gate (below) rejects all three against E0.

Phase 3 (test split, 360 images, 854 ground-truth defects, protocol pre-registered before the test run): the frozen
per-class policy cost 3052.7 against 3041.9 for a single matched global threshold of 0.127, a saving of -0.4%
(95% CI -7.8% to +6.2%). **The pre-registered hypothesis, that a tuned per-class review policy beats a matched flat
threshold, was not supported.** Both beat the default 0.25 threshold by about 22%.

## Per-class AP (val)

| Class | AP50 (Phase 2a) | AP30 (Phase 4) | AP50 (Phase 4) | AP75 (Phase 4) |
|---|---|---|---|---|
| scratches | 0.952 | 0.980 | 0.969 | 0.436 |
| patches | 0.877 | 0.925 | 0.856 | 0.565 |
| pitted_surface | 0.845 | 0.915 | 0.870 | 0.513 |
| inclusion | 0.813 | 0.892 | 0.780 | 0.381 |
| rolled-in_scale | **0.622** | 0.855 | 0.659 | 0.309 |
| crazing | **0.419** | 0.688 | 0.425 | 0.037 |

(Phase 2a numbers come from the Ultralytics validator; Phase 4 numbers from the project's own AP code on cached
detections, hence the small differences.)

**Crazing and rolled-in scale are weak.** The Phase 4 error analysis, with hypotheses written before the run, found
that crazing's problem is mostly **localization, not detection**: at the operating threshold 37 of 41 crazing
ground-truth boxes that were not true positives (90%) were found but poorly localized (best IoU 0.15 to 0.50), and only 4
were missed outright. Crazing AP rises from 0.425 at IoU 0.5 to 0.688 at IoU 0.3. Rolled-in scale shows a similar but
weaker gap (+0.196). Crazing boxes are probably ambiguous (diffuse texture with no crisp boundary), but that
explanation is untested.

## Robustness (val, synthetic perturbations, one model)

mAP50 change from a clean 0.760 (95% bootstrap CI over images):

| Perturbation | Mild | Medium | Strong |
|---|---|---|---|
| Blur (sigma 1 / 2 / 3 px) | -0.24 [-0.27, -0.19] | -0.43 | -0.53 |
| Noise (sigma 0.02 / 0.05 / 0.1) | -0.20 [-0.25, -0.15] | -0.55 | -0.73 |
| JPEG (quality 50 / 20) | -0.12 [-0.15, -0.09] | -0.36 | n/a |
| Contrast (x0.6 / x1.4) | -0.04 [-0.07, -0.02] | -0.03 | n/a |
| Brightness (x0.6 / 0.8 / 1.2 / 1.4) | -0.006 [-0.02, +0.01] | -0.002 | -0.014 / -0.010 |

**Fragile to blur, noise and JPEG compression, robust to brightness, mildly affected by contrast.** These are
synthetic corruptions applied to val images, not real camera or process changes.

## Decision layer (cost-based, frozen policy)

Each detection is mapped to an action from an explicit cost model: miss 10, false alarm 1, human review 0.3, with
per-class weights (all 1.0) and an area-based severity factor. **All cost values are illustrative assumptions, not
plant economics.** Per-class confidence bands (lower, upper) route a detection to `human_review` (between the bands)
or to an alert action (above the upper band; `ignore`, `monitor`, `schedule_service` or `stop_now` by severity).

The policy was tuned once on the 216 val images (workload budget at most 3 alerts plus reviews per image, reviewer
accuracy 0.9) and frozen in `reports/phase3/frozen.json` before the test run. Frozen primary bands:
crazing 0.10-0.25, inclusion 0.25-0.30, patches 0.45, pitted_surface 0.05-0.10, rolled-in_scale 0.15-0.20,
scratches 0.15. As reported above, it performs no better than one global threshold of 0.127. The Space and the API
apply this policy unchanged. The per-image decision (most urgent action among detections) is a serving choice that
**has not been evaluated**.

## Monitoring and promotion gate (simulated)

- **Drift detection:** a two-sample KS test per feature (brightness, contrast, sharpness, noise estimate, max
  confidence, detection count) between a reference window and a sliding window, with Holm correction across the six
  features. Chosen on a calibration stream (seed 100) and scored on a fresh stream (seed 200). Tier 1 is an
  informational input-drift alert; tier 2 "retrain recommended" needs input and prediction drift together.
  On clean simulated streams: no tier-2 false triggers in 20 streams (about 15,400 windows). Tier 2 fires reliably on
  noise and strong blur, but **triggers on only 0% (blur 1) and 50% (blur 2) of streams, and on 30% and 13% for the two
  contrast levels, even though blur 1 and 2 cost 0.24 and 0.43 mAP50**. Brightness drift raises tier-1 alerts in about 99% of streams but never a retrain
  recommendation, and it barely changes mAP50: drift that does not hurt.
- **Promotion gate (`configs/promotion.yaml`, fixed before use):** a candidate must gain at least 0.01 mAP50, lose no
  more than 0.05 AP50 in any class, and, when per-image predictions exist, have a paired bootstrap CI excluding zero.
  Applied to the stored E1, E2 and E3 results against E0 on val, all three are rejected. The CI condition could not be
  applied because only aggregate metrics were stored.

## Limitations

- **One dataset, one source.** 200x200 grayscale images from one lab collection; no other cameras, lighting, steel
  grades or image sizes.
- **No healthy parts.** Every NEU-DET image contains a defect, so false-alarm behaviour on defect-free steel, the
  dominant case in a real line, is unknown.
- **Simulated drift only.** All drift and monitoring results come from replaying val images with injected
  perturbations. The drift tests use a pool of 216 images sampled with replacement, so reference and current windows
  share images. No real drift was ever observed.
- **Per-image roll-up unevaluated.** The per-image decision rule and its cost impact were never measured.
- **Small, single-seed evaluation.** 36 val images per class, one training seed, one test run; confidence intervals are
  wide and the cost model's inputs are assumptions.
- **Weak classes.** Crazing (AP50 about 0.42) is unreliable; the localization finding explains much of that but not all.
- **Licences.** Ultralytics YOLOv8 is AGPL-3.0, whose network-use clause can require offering source to users of a
  modified build served over a network (check your own case); the weights here are published under AGPL-3.0 accordingly. The NEU-DET
  licence is unspecified, so do not assume the weights are free of upstream data restrictions.
- **Nothing was validated on real production data.** Do not use this model to make real inspection decisions.

## Data attribution and citation

Images come from the **NEU surface defect database (NEU-DET)**, Northeastern University, Shenyang, China. The examples in this repository are the first validation-split image of each class (the validation split is an 85/15 split of the provided training set; the test split is never used). The dataset's licence is listed as "Unknown" on Kaggle, so no dataset is redistributed beyond these six small validation images.

- K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip
  surface defects," *Applied Surface Science*, vol. 285, pp. 858-864, 2013.
- Y. He, K. Song, Q. Meng and Y. Yan, "An end-to-end steel surface defect detection approach via fusing multiple
  hierarchical features," *IEEE Transactions on Instrumentation and Measurement*, vol. 69, no. 4, pp. 1493-1504, 2020.
