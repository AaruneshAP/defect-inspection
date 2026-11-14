# Steel surface defect inspection (hosted demo)

> **Portfolio demo on a public research dataset (NEU-DET).** Not validated on real production data. The maintenance actions use illustrative cost assumptions, not plant economics. Do not use it for real inspection decisions.

**Live app:** <https://defect-inspection.streamlit.app/>  
**Model repository:** <https://huggingface.co/Aarun-1/defect-detector-neu-det>

---

## App structure

The app runs on Streamlit Community Cloud without heavy PyTorch or GPU dependencies, using a CPU-optimised ONNX runtime backend (`best.onnx`) and the frozen Phase 3 cost-based decision policy:

1. **Try it (`try_it.py`):**
   - Upload any grayscale steel surface image (or select one of six NEU-DET validation-split examples).
   - Detects six defect classes: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, and `scratches`.
   - Maps each detection to a maintenance action (`stop_now`, `schedule_service`, `human_review`, `monitor`, `ignore`) based on confidence and bounding-box area.
   - Shows the per-image decision (most urgent action among detections; note this roll-up is an un-evaluated serving heuristic).

2. **Monitoring (`monitoring.py`):**
   - Live telemetry viewer inspecting simulated inspection streams with injected sensor drift.
   - Monitors input statistics (`sharpness`, `noise_est`, `contrast_std`, `brightness_mean`) and model signals (`max_confidence`, `n_detections`).
   - Displays Kolmogorov-Smirnov (KS) drift test results with Holm family-wise error rate correction and two-tier alerts.

---

## Running locally

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/streamlit_app.py
```

Optional environment variables:
- `ONNX_PATH`: path to a local `models/best.onnx` file (skips downloading from Hugging Face).
- `EXAMPLES_DIR`: path to a local directory with validation images (e.g. `data/yolo/images/val`).

---

## Data attribution and citation

Images come from the **NEU surface defect database (NEU-DET)**, Northeastern University, Shenyang, China. The examples provided in the app are the first validation-split image of each class (the validation split is an 85/15 split of the provided training set; the test split is never used). The dataset's licence is listed as "Unknown" on Kaggle, so no dataset is redistributed beyond these six small validation images.

- K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," *Applied Surface Science*, vol. 285, pp. 858-864, 2013.
- Y. He, K. Song, Q. Meng and Y. Yan, "An end-to-end steel surface defect detection approach via fusing multiple hierarchical features," *IEEE Transactions on Instrumentation and Measurement*, vol. 69, no. 4, pp. 1493-1504, 2020.

---

## Software and licence

Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0), so this demo and model weights are published under AGPL-3.0. Model weights, ONNX export, and model card: <https://huggingface.co/Aarun-1/defect-detector-neu-det>.

