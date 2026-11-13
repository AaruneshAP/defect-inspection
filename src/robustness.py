"""Phase 4b robustness: run the e0 baseline on perturbed *val* images and measure the degradation.

  python -m src.robustness run       # predict on clean + perturbed val images (cached in data/cache/perturbed/)
  python -m src.robustness report    # metrics, paired bootstrap CIs on the change vs clean, CSVs + figure

Metrics (policy threshold 0.127 is NOT retuned): per-class AP50 and mAP50 over all detections (conf >= 0.001),
recall and alerts per image at conf >= 0.127. IoU matching is greedy, class-aware, one-to-one, IoU >= 0.5.
delta = perturbed - clean (negative = worse for AP and recall). CIs are percentile 95% intervals from a paired
image bootstrap: the same resampled images are used for the clean and the perturbed run.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.analysis import CLASS_NAMES, THRESHOLD, TP_IOU, match
from src.perturb import LEVELS, all_conditions, apply_perturbation, image_seed, level_value

CACHE_DIR = Path("data/cache/perturbed")
OUT_DIR = Path("reports/phase4")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
METRICS = ["mAP50", "recall", "alerts_per_image"] + [f"AP50_{n}" for n in CLASS_NAMES]


# ---------------------------------------------------------------- inference
def tag(name: str, severity: int) -> str:
    return "clean" if severity == 0 else f"{name}_{severity}"


def collect_perturbed(model_path: str, data_dir: str, split: str, name: str, severity: int, seed: int = 0,
                      imgsz: int = 640, device: str = "cpu", model=None) -> pd.DataFrame:
    """Predict every image of `split` after perturbing it; return preds + GT rows like src.predict.collect."""
    import cv2
    from src.predict import COLUMNS, CONF_FLOOR, read_gt

    if split == "test":
        raise ValueError("the test split is spent; robustness runs on val only")
    if model is None:
        from ultralytics import YOLO
        model = YOLO(model_path)
    img_dir, lbl_dir = Path(data_dir) / "images" / split, Path(data_dir) / "labels" / split
    rows = []
    for p in sorted(q for q in img_dir.iterdir() if q.suffix.lower() in IMG_EXT):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)  # BGR, as ultralytics loads from a path
        h, w = img.shape[:2]
        img = apply_perturbation(img, name, severity, image_seed(p.name, seed))
        res = model.predict(img, conf=CONF_FLOOR, imgsz=imgsz, device=device, verbose=False)[0]
        for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            rows.append(("pred", p.name, int(k), float(c), x1, y1, x2, y2, w, h))
        rows += read_gt(lbl_dir / f"{p.stem}.txt", w, h, p.name)
    return pd.DataFrame(rows, columns=COLUMNS)


def cmd_run(a) -> None:
    from ultralytics import YOLO
    model = YOLO(a.model)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, sev in [("blur", 0)] + all_conditions():
        path = CACHE_DIR / f"{tag(name, sev)}.parquet"
        if path.exists() and not a.force:
            print(f"{path.name}: cached")
            continue
        df = collect_perturbed(a.model, a.data_dir, "val", name, sev, a.seed, a.imgsz, a.device, model)
        df.to_parquet(path, index=False)
        print(f"{path.name}: {(df.kind == 'pred').sum()} detections, {df.image.nunique()} images", flush=True)


# ---------------------------------------------------------------- per-image arrays + bootstrap
class Prepared:
    """Everything the bootstrap needs from one run, as arrays indexed by image."""

    def __init__(self, df: pd.DataFrame, images: List[str], threshold: float = THRESHOLD, iou: float = TP_IOU):
        self.n = len(images)
        pos = {im: i for i, im in enumerate(images)}
        preds = df[df.kind == "pred"].reset_index(drop=True)
        gts = df[df.kind == "gt"].reset_index(drop=True)
        m = match(preds, gts, iou)                      # det -> GT row or -1
        gt_conf = np.full(len(gts), -1.0)
        for d, g in enumerate(m):
            if g >= 0:
                gt_conf[g] = preds.conf.iloc[d]
        gimg = gts.image.map(pos).to_numpy()
        self.gt_img, self.gt_cls = gimg, gts.class_id.to_numpy()
        caught = gt_conf >= threshold
        self.caught_per_img = np.bincount(gimg, weights=caught, minlength=self.n)
        self.gt_per_img = np.bincount(gimg, minlength=self.n).astype(float)
        self.gt_per_img_cls = np.stack([np.bincount(gimg[self.gt_cls == c], minlength=self.n)
                                        for c in range(len(CLASS_NAMES))]).astype(float)
        kept = preds.conf.to_numpy() >= threshold
        self.alerts_per_img = np.bincount(preds.image.map(pos).to_numpy()[kept], minlength=self.n).astype(float)
        self.cls_dets = []                               # per class: (img index, tp flag) sorted by -conf
        for c in range(len(CLASS_NAMES)):
            sel = np.where(preds.class_id.to_numpy() == c)[0]
            sel = sel[np.argsort(-preds.conf.to_numpy()[sel], kind="stable")]
            self.cls_dets.append((preds.image.map(pos).to_numpy()[sel], (m[sel] >= 0).astype(float)))

    def metrics(self, w: np.ndarray = None) -> Dict[str, float]:
        """All metrics for image weights w (bootstrap counts); w=None means every image once."""
        w = np.ones(self.n) if w is None else w
        aps = [weighted_ap(img, tp, w, float(w @ self.gt_per_img_cls[c]))
               for c, (img, tp) in enumerate(self.cls_dets)]
        out = {"mAP50": float(np.nanmean(aps)),
               "recall": float(w @ self.caught_per_img / (w @ self.gt_per_img)),
               "alerts_per_image": float(w @ self.alerts_per_img / w.sum())}
        out.update({f"AP50_{n}": aps[c] for c, n in enumerate(CLASS_NAMES)})
        return out


def weighted_ap(det_img: np.ndarray, tp: np.ndarray, w: np.ndarray, n_gt: float) -> float:
    """101-point interpolated AP (same recipe as src.analysis.average_precision) where image i counts w[i] times.
    Detections arrive sorted by descending confidence."""
    if n_gt <= 0:
        return float("nan")
    wd = w[det_img]
    keep = wd > 0
    wd, t = wd[keep], tp[keep]
    if len(wd) == 0:
        return 0.0
    tpc, fpc = np.cumsum(wd * t), np.cumsum(wd * (1.0 - t))
    recall, precision = tpc / n_gt, tpc / (tpc + fpc)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], precision, [0.0])))))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def paired_bootstrap(clean: Prepared, pert: Prepared, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per metric: clean value, perturbed value, delta = perturbed - clean with percentile 95% CI over images."""
    assert clean.n == pert.n
    rng = np.random.default_rng(seed)
    c0, p0 = clean.metrics(), pert.metrics()
    deltas = {k: np.empty(n_boot) for k in METRICS}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, clean.n, clean.n), minlength=clean.n).astype(float)
        mc, mp = clean.metrics(w), pert.metrics(w)
        for k in METRICS:
            deltas[k][b] = mp[k] - mc[k]
    rows = [{"metric": k, "clean": c0[k], "value": p0[k], "delta": p0[k] - c0[k],
             "delta_lo": float(np.nanpercentile(deltas[k], 2.5)), "delta_hi": float(np.nanpercentile(deltas[k], 97.5))}
            for k in METRICS]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report
def plot_degradation(long: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    okabe = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
    rows = [("mAP50", "change in AP50 (mAP bold, per class thin)"), ("recall", f"change in recall @ {THRESHOLD}"),
            ("alerts_per_image", f"change in alerts / image @ {THRESHOLD}")]
    names = list(LEVELS)
    fig, axes = plt.subplots(len(rows), len(names), figsize=(3.1 * len(names), 2.9 * len(rows)), sharey="row")
    for j, name in enumerate(names):
        sub = long[long.perturbation == name]
        xs = [0] + sorted(sub.severity.unique().tolist())
        labels = ["clean"] + [f"{level_value(name, s):g}" for s in xs[1:]]
        for i, (metric, ylabel) in enumerate(rows):
            ax = axes[i, j]
            ax.axhline(0, color="#999", lw=0.8)
            if metric == "mAP50":
                for c, cn in enumerate(CLASS_NAMES):
                    d = sub[sub.metric == f"AP50_{cn}"].sort_values("severity")
                    ax.plot(xs, [0.0] + d.delta.tolist(), color=okabe[c], lw=1.1, alpha=.8,
                            label=cn if j == 0 else None)
            d = sub[sub.metric == metric].sort_values("severity")
            y = np.array([0.0] + d.delta.tolist())
            lo, hi = np.array([0.0] + d.delta_lo.tolist()), np.array([0.0] + d.delta_hi.tolist())
            ax.errorbar(xs, y, yerr=[y - lo, hi - y], color="black", lw=2.2, marker="o", ms=4, capsize=3,
                        label="mean" if (i == 0 and j == 0) else None, zorder=5)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
            if i == 0:
                ax.set_title(name, fontsize=11, loc="left")
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.suptitle("Robustness of the e0 baseline on perturbed val images (error bars: paired bootstrap 95% CI)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def cmd_report(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clean_df = pd.read_parquet(CACHE_DIR / "clean.parquet")
    images = sorted(clean_df.image.unique())
    clean = Prepared(clean_df, images)
    wide, long = [], []
    row = {"perturbation": "clean", "severity": 0, "level": np.nan, **clean.metrics()}
    wide.append(row)
    for name, sev in all_conditions():
        df = pd.read_parquet(CACHE_DIR / f"{tag(name, sev)}.parquet")
        assert sorted(df.image.unique()) == images
        pert = Prepared(df, images)
        bt = paired_bootstrap(clean, pert, a.n_boot, a.seed)
        wide.append({"perturbation": name, "severity": sev, "level": level_value(name, sev), **pert.metrics()})
        long.append(bt.assign(perturbation=name, severity=sev, level=level_value(name, sev)))
        print(f"{tag(name, sev):14s} mAP50 {bt.loc[0, 'value']:.3f} ({bt.loc[0, 'delta']:+.3f} "
              f"[{bt.loc[0, 'delta_lo']:+.3f}, {bt.loc[0, 'delta_hi']:+.3f}])", flush=True)
    wide_df = pd.DataFrame(wide).round(4)
    long_df = pd.concat(long, ignore_index=True)[["perturbation", "severity", "level", "metric", "clean", "value",
                                                  "delta", "delta_lo", "delta_hi"]].round(4)
    wide_df.to_csv(out / "robustness_metrics.csv", index=False)
    long_df.to_csv(out / "robustness_deltas.csv", index=False)
    plot_degradation(long_df, out / "robustness_degradation.png")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\nmetrics:\n", wide_df.to_string(index=False))
    show = long_df[long_df.metric.isin(["mAP50", "recall", "alerts_per_image"])].copy()
    show["delta_ci"] = show.apply(lambda r: f"{r.delta:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]", axis=1)
    print("\ndeltas vs clean:\n", show.pivot(index=["perturbation", "severity", "level"], columns="metric",
                                           values="delta_ci").to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--data-dir", default="data/yolo")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
