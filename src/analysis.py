"""Phase 4a error analysis on the cached *val* detections of the e0 baseline (test split is never opened).

Categorises every GT box and every kept detection (conf >= threshold), computes per-class AP at several
IoU thresholds, GT box statistics, and writes CSVs + figures to reports/phase4/.

GT categories (at the shipped threshold, detections of any class that are kept):
  tp                one-to-one match with a same-class detection, IoU >= 0.5
  shared            unmatched, but a same-class detection with IoU >= 0.5 exists (it was consumed by another GT)
  wrong_class       unmatched, an other-class detection has IoU >= 0.5 (and no same-class one does)
  poorly_localized  unmatched, best same-class IoU in (0.1, 0.5)
  missed            otherwise (no same-class detection with IoU > 0.1)
Detection categories:
  tp                matched to a GT (one-to-one, same class, IoU >= 0.5, greedy by confidence)
  duplicate         unmatched, same-class GT IoU >= 0.5 (that GT already has a better detection)
  wrong_class       unmatched, other-class GT IoU >= 0.5
  poorly_localized  unmatched, best same-class GT IoU in (0.1, 0.5)
  background        otherwise (no same-class GT with IoU > 0.1 and no other-class GT with IoU >= 0.5)
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
THRESHOLD = 0.127
TP_IOU = 0.5
LOC_IOU = 0.1
AP_IOUS = (0.1, 0.3, 0.5, 0.75)
CURVE_IOUS = [round(x, 2) for x in np.arange(0.10, 0.901, 0.05)]
GT_CATS = ["tp", "shared", "wrong_class", "poorly_localized", "missed"]
DET_CATS = ["tp", "duplicate", "wrong_class", "poorly_localized", "background"]
BOX = ["x1", "y1", "x2", "y2"]


# ---------------------------------------------------------------- geometry / matching
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes, shape (len(a), len(b))."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match(preds: pd.DataFrame, gts: pd.DataFrame, iou_thr: float) -> np.ndarray:
    """Greedy class-aware one-to-one matching, highest confidence first, each detection takes the unmatched
    same-class GT of highest IoU if that IoU >= iou_thr. Returns, per row of `preds`, the *row position*
    in `gts` of its GT, or -1."""
    out = np.full(len(preds), -1, int)
    gkeys = gts.reset_index(drop=True).groupby(["image", "class_id"]).indices
    p = preds.reset_index(drop=True)
    for key, pidx in p.groupby(["image", "class_id"]).indices.items():
        gidx = gkeys.get(key)
        if gidx is None:
            continue
        pidx = pidx[np.argsort(-p.conf.to_numpy()[pidx], kind="stable")]
        ious = iou_matrix(p[BOX].to_numpy()[pidx], gts[BOX].to_numpy()[gidx])
        free = np.ones(len(gidx), bool)
        for r, pi in enumerate(pidx):
            cand = np.where(free, ious[r], -1.0)
            j = int(cand.argmax())
            if cand[j] >= iou_thr:
                free[j] = False
                out[pi] = gidx[j]
    return out


def average_precision(conf: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP over the precision envelope (same recipe as ultralytics)."""
    if n_gt == 0:
        return float("nan")
    if len(conf) == 0:
        return 0.0
    order = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp[order].astype(float))
    fpc = np.cumsum(1.0 - tp[order].astype(float))
    recall = tpc / n_gt
    precision = tpc / (tpc + fpc)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def per_class_ap(preds: pd.DataFrame, gts: pd.DataFrame, iou_thr: float, n_classes: int = 6) -> List[float]:
    """AP per class at one IoU threshold using every detection in `preds`."""
    preds, gts = preds.reset_index(drop=True), gts.reset_index(drop=True)
    tp = match(preds, gts, iou_thr) >= 0
    aps = []
    for c in range(n_classes):
        m = (preds.class_id == c).to_numpy()
        aps.append(average_precision(preds.conf.to_numpy()[m], tp[m], int((gts.class_id == c).sum())))
    return aps


def ap_table(preds: pd.DataFrame, gts: pd.DataFrame, ious, names: List[str]) -> pd.DataFrame:
    """Rows = classes (+ mean), columns = IoU thresholds."""
    cols = {f"AP{int(round(t * 100))}": per_class_ap(preds, gts, t, len(names)) for t in ious}
    df = pd.DataFrame(cols, index=names)
    df.loc["mean"] = df.mean()
    df.index.name = "class"
    return df


# ---------------------------------------------------------------- categorisation
def categorise_gts(kept: pd.DataFrame, gts: pd.DataFrame, tp_iou: float = TP_IOU,
                   loc_iou: float = LOC_IOU) -> pd.DataFrame:
    """Return `gts` (index reset) with columns category, best_iou (best same-class IoU over kept detections)."""
    gts, kept = gts.reset_index(drop=True), kept.reset_index(drop=True)
    matched = set(int(i) for i in match(kept, gts, tp_iou) if i >= 0)
    cats, best = [], []
    kimg = kept.groupby("image").indices
    for gi, g in gts.iterrows():
        idx = kimg.get(g.image)
        s = o = 0.0
        if idx is not None:
            ious = iou_matrix(np.array([[g.x1, g.y1, g.x2, g.y2]]), kept[BOX].to_numpy()[idx])[0]
            same = kept.class_id.to_numpy()[idx] == g.class_id
            s = float(ious[same].max()) if same.any() else 0.0
            o = float(ious[~same].max()) if (~same).any() else 0.0
        best.append(s)
        if gi in matched:
            cats.append("tp")
        elif s >= tp_iou:
            cats.append("shared")
        elif o >= tp_iou:
            cats.append("wrong_class")
        elif s > loc_iou:
            cats.append("poorly_localized")
        else:
            cats.append("missed")
    return gts.assign(category=cats, best_iou=best)


def categorise_dets(kept: pd.DataFrame, gts: pd.DataFrame, tp_iou: float = TP_IOU,
                    loc_iou: float = LOC_IOU) -> pd.DataFrame:
    """Return `kept` (index reset) with columns category, best_iou (best same-class IoU over GT boxes)."""
    gts, kept = gts.reset_index(drop=True), kept.reset_index(drop=True)
    is_tp = match(kept, gts, tp_iou) >= 0
    gimg = gts.groupby("image").indices
    cats, best = [], []
    for di, d in kept.iterrows():
        idx = gimg.get(d.image)
        s = o = 0.0
        if idx is not None:
            ious = iou_matrix(np.array([[d.x1, d.y1, d.x2, d.y2]]), gts[BOX].to_numpy()[idx])[0]
            same = gts.class_id.to_numpy()[idx] == d.class_id
            s = float(ious[same].max()) if same.any() else 0.0
            o = float(ious[~same].max()) if (~same).any() else 0.0
        best.append(s)
        if is_tp[di]:
            cats.append("tp")
        elif s >= tp_iou:
            cats.append("duplicate")
        elif o >= tp_iou:
            cats.append("wrong_class")
        elif s > loc_iou:
            cats.append("poorly_localized")
        else:
            cats.append("background")
    return kept.assign(category=cats, best_iou=best)


def breakdown(df: pd.DataFrame, cats: List[str], names: List[str]) -> pd.DataFrame:
    """Per-class counts and shares of each category (+ total row)."""
    cnt = pd.crosstab(df.class_id, df.category).reindex(index=range(len(names)), columns=cats, fill_value=0)
    cnt.index = names
    cnt.loc["all"] = cnt.sum()
    cnt.insert(0, "n", cnt.sum(axis=1))
    share = cnt[cats].div(cnt["n"].replace(0, np.nan), axis=0).add_suffix("_share").round(4)
    cnt.index.name = "class"
    return pd.concat([cnt, share], axis=1)


# ---------------------------------------------------------------- GT box statistics
def gt_stats(gts: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    """Per-class: area fraction, aspect ratio, and how often same-class GT boxes overlap within an image."""
    g = gts.reset_index(drop=True).copy()
    w, h = g.x2 - g.x1, g.y2 - g.y1
    g["area_frac"] = w * h / (g.img_w * g.img_h)
    g["aspect"] = w / h
    g["aspect_max"] = np.maximum(w / h, h / w)
    ov_any = np.zeros(len(g), bool)
    ov_30 = np.zeros(len(g), bool)
    max_iou = np.zeros(len(g))
    for _, idx in g.groupby(["image", "class_id"]).indices.items():
        if len(idx) < 2:
            continue
        b = g[BOX].to_numpy()[idx]
        m = iou_matrix(b, b)
        np.fill_diagonal(m, 0.0)
        iw = np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0])
        ih = np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1])
        inter = (np.clip(iw, 0, None) * np.clip(ih, 0, None))
        np.fill_diagonal(inter, 0.0)
        ov_any[idx] = (inter > 0).any(axis=1)
        ov_30[idx] = (m >= 0.3).any(axis=1)
        max_iou[idx] = m.max(axis=1)
    g["overlaps_sibling"], g["overlaps_sibling_iou30"], g["max_sibling_iou"] = ov_any, ov_30, max_iou
    rows = []
    for c, n in enumerate(names):
        s = g[g.class_id == c]
        per_img = s.groupby("image").size()
        rows.append({"class": n, "n_gt": len(s), "n_images": len(per_img), "boxes_per_image": per_img.mean(),
                     "area_frac_median": s.area_frac.median(), "area_frac_p10": s.area_frac.quantile(.1),
                     "area_frac_p90": s.area_frac.quantile(.9), "aspect_wh_median": s.aspect.median(),
                     "aspect_max_median": s.aspect_max.median(), "aspect_max_p90": s.aspect_max.quantile(.9),
                     "frac_overlap_sibling": s.overlaps_sibling.mean(),
                     "frac_overlap_sibling_iou30": s.overlaps_sibling_iou30.mean(),
                     "max_sibling_iou_median": s.max_sibling_iou.median()})
    return pd.DataFrame(rows).set_index("class").round(4)


# ---------------------------------------------------------------- galleries / figures
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
CAT_COLORS = {"tp": "#009E73", "shared": "#56B4E9", "wrong_class": "#CC79A7", "poorly_localized": "#E69F00",
              "missed": "#D55E00", "duplicate": "#56B4E9", "background": "#888888"}


def plot_breakdown(gt_bd: pd.DataFrame, det_bd: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, bd, cats, title in [(axes[0], gt_bd, GT_CATS, "GT boxes"),
                                (axes[1], det_bd, DET_CATS, f"Detections at conf >= {THRESHOLD}")]:
        d = bd.drop(index="all")
        left = np.zeros(len(d))
        for c in cats:
            share = d[f"{c}_share"].fillna(0).to_numpy()
            ax.barh(range(len(d)), share, left=left, color=CAT_COLORS[c], label=c, edgecolor="white", linewidth=1.5)
            left += share
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([f"{n}  (n={int(k)})" for n, k in zip(d.index, d.n)])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of boxes")
        ax.set_title(title, loc="left", fontsize=11)
        ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Failure breakdown, val, e0 baseline", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ap_curves(curve: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = [float(c[2:]) / 100 for c in curve.columns]
    for i, (name, row) in enumerate(curve.drop(index="mean").iterrows()):
        hi = name == "crazing"
        ax.plot(xs, row.to_numpy(), color=OKABE[i], lw=3 if hi else 1.6, marker="o", ms=3.5, label=name,
                zorder=3 if hi else 2)
    ax.plot(xs, curve.loc["mean"].to_numpy(), color="#444", lw=1.4, ls="--", label="mean")
    for t in AP_IOUS:
        ax.axvline(t, color="#bbb", lw=0.8, zorder=0)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("AP")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-class AP vs IoU threshold (val, e0 baseline)", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_gallery(items: List[dict], gts: pd.DataFrame, kept: pd.DataFrame, img_dir: Path, path: Path,
                 title: str, names: List[str], cols: int = 4) -> None:
    """items: dicts with image, box (x1,y1,x2,y2), caption. Focus box is drawn thick; all GT green, kept dets orange."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    rows = int(np.ceil(max(len(items), 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.9 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, it in zip(axes.ravel(), items):
        ax.imshow(Image.open(img_dir / it["image"]).convert("L"), cmap="gray", vmin=0, vmax=255)
        for _, g in gts[gts.image == it["image"]].iterrows():
            ax.add_patch(Rectangle((g.x1, g.y1), g.x2 - g.x1, g.y2 - g.y1, fill=False, ec="#00e676", lw=1.2,
                                   ls="-"))
        for _, d in kept[kept.image == it["image"]].iterrows():
            ax.add_patch(Rectangle((d.x1, d.y1), d.x2 - d.x1, d.y2 - d.y1, fill=False, ec="#ff9100", lw=1.2,
                                   ls="--"))
            ax.text(d.x1 + 2, d.y1 + 9, f"{names[int(d.class_id)][:4]} {d.conf:.2f}", color="#ff9100", fontsize=6.5,
                    bbox=dict(fc="black", alpha=.55, pad=.5, ec="none"))
        x1, y1, x2, y2 = it["box"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#ffffff" if it["kind"] == "gt" else "#ff1744",
                               lw=2.6, ls="-" if it["kind"] == "gt" else "--"))
        ax.set_title(it["caption"], fontsize=7.5)
    fig.suptitle(title + "   (green = GT, orange dashed = detections >= %.3f, thick = the box in question)" % THRESHOLD,
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def select_misses(gt_cat: pd.DataFrame, cls: int, n: int = 12) -> pd.DataFrame:
    """Worst misses of one class: 'missed' first (largest GT area first), then poorly localized (lowest IoU first),
    then the remaining non-TP GT boxes."""
    s = gt_cat[(gt_cat.class_id == cls) & (gt_cat.category != "tp")].copy()
    s["area_frac"] = (s.x2 - s.x1) * (s.y2 - s.y1) / (s.img_w * s.img_h)
    rank = {"missed": 0, "wrong_class": 1, "poorly_localized": 2, "shared": 3}
    s["rank"] = s.category.map(rank)
    return s.sort_values(["rank", "area_frac", "best_iou"], ascending=[True, False, True]).head(n)


def select_false_alarms(det_cat: pd.DataFrame, cls: int, n: int = 12) -> pd.DataFrame:
    """Highest-confidence non-TP detections of one class."""
    s = det_cat[(det_cat.class_id == cls) & (det_cat.category != "tp")]
    return s.sort_values("conf", ascending=False).head(n)


# ---------------------------------------------------------------- driver
def run(cache: str, out: Path, img_dir: Path, threshold: float = THRESHOLD) -> Dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cache)
    preds = df[df.kind == "pred"].reset_index(drop=True)
    gts = df[df.kind == "gt"].reset_index(drop=True)
    kept = preds[preds.conf >= threshold].reset_index(drop=True)

    gt_cat = categorise_gts(kept, gts)
    det_cat = categorise_dets(kept, gts)
    gt_bd = breakdown(gt_cat, GT_CATS, CLASS_NAMES)
    det_bd = breakdown(det_cat, DET_CATS, CLASS_NAMES)
    gt_bd.to_csv(out / "failure_breakdown_gt.csv")
    det_bd.to_csv(out / "failure_breakdown_det.csv")

    ap4 = ap_table(preds, gts, AP_IOUS, CLASS_NAMES).round(4)
    ap4.to_csv(out / "ap_by_iou.csv")
    curve = ap_table(preds, gts, CURVE_IOUS, CLASS_NAMES).round(4)
    curve.to_csv(out / "ap_curve.csv")
    stats = gt_stats(gts, CLASS_NAMES)
    stats.to_csv(out / "gt_stats.csv")

    plot_breakdown(gt_bd, det_bd, out / "failure_breakdown.png")
    plot_ap_curves(curve, out / "ap_vs_iou.png")

    miss = select_misses(gt_cat, 0)
    fa = select_false_alarms(det_cat, 0)
    miss_items = [dict(image=r.image, box=(r.x1, r.y1, r.x2, r.y2), kind="gt",
                       caption=f"{r.image}  {r.category}  bestIoU={r.best_iou:.2f}  area={r.area_frac:.2f}")
                  for r in miss.itertuples()]
    fa_items = [dict(image=r.image, box=(r.x1, r.y1, r.x2, r.y2), kind="det",
                     caption=f"{r.image}  {r.category}  conf={r.conf:.2f}  bestIoU={r.best_iou:.2f}")
                for r in fa.itertuples()]
    draw_gallery(miss_items, gts, kept, img_dir, out / "gallery_crazing_misses.png",
                 "Worst crazing misses (GT white thick)", CLASS_NAMES)
    draw_gallery(fa_items, gts, kept, img_dir, out / "gallery_crazing_false_alarms.png",
                 "Top crazing false alarms (red dashed thick)", CLASS_NAMES)
    pd.concat([miss.assign(gallery="miss"), fa.assign(gallery="false_alarm")]).to_csv(
        out / "gallery_manifest.csv", index=False)
    return dict(gt_bd=gt_bd, det_bd=det_bd, ap=ap4, curve=curve, stats=stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default="data/cache/val_detections.parquet")
    ap.add_argument("--out", default="reports/phase4")
    ap.add_argument("--img-dir", default="data/yolo/images/val")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args()
    r = run(a.cache, Path(a.out), Path(a.img_dir), a.threshold)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    for k, v in r.items():
        print(f"\n== {k} ==\n{v}")


if __name__ == "__main__":
    main()
