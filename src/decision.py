"""Cost-based decision layer: match detections to GT, tune per-class thresholds, map to actions."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

HUMAN_REVIEW = "human_review"
_BOX = ["x1", "y1", "x2", "y2"]


def load_cost_config(path: str | Path = "configs/cost.yaml") -> dict:
    """Load the cost/severity YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iou(a, b) -> float:
    """IoU of two xyxy boxes."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0.0) * max(ih, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def area_factor(area_frac: float, cfg: dict) -> float:
    """Linear severity factor in box-area fraction, clamped to the configured anchors."""
    s = cfg["area_severity"]
    lo, hi = s["area_frac_small"], s["area_frac_large"]
    t = min(max((area_frac - lo) / (hi - lo), 0.0), 1.0)
    return s["factor_small"] + t * (s["factor_large"] - s["factor_small"])


def severity(class_name: str, box, img_w: float, img_h: float, cfg: dict) -> float:
    """class_weight * area factor for one xyxy box."""
    frac = (box[2] - box[0]) * (box[3] - box[1]) / (img_w * img_h)
    return cfg["class_weights"].get(class_name, 1.0) * area_factor(frac, cfg)


def match_detections(preds: pd.DataFrame, gts: pd.DataFrame, iou_thr: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Greedy class-aware matching, highest confidence first, one detection per GT.

    Returns (preds with bool 'tp', gts with 'match_conf' = conf of its matched detection or NaN).
    Greedy-by-confidence means thresholding afterwards equals matching only the kept detections.
    """
    preds = preds.copy()
    gts = gts.copy()
    preds["tp"] = False
    gts["match_conf"] = np.nan
    for (img, cls), p in preds.groupby(["image", "class_id"]):
        g = gts[(gts.image == img) & (gts.class_id == cls)]
        if g.empty:
            continue
        taken = set()
        for pi, prow in p.sort_values("conf", ascending=False).iterrows():
            best, best_iou = None, iou_thr
            for gi, grow in g.iterrows():
                if gi in taken:
                    continue
                v = iou(prow[_BOX].to_numpy(float), grow[_BOX].to_numpy(float))
                if v >= best_iou:
                    best, best_iou = gi, v
            if best is not None:
                taken.add(best)
                preds.loc[pi, "tp"] = True
                gts.loc[best, "match_conf"] = prow["conf"]
    return preds, gts


def _miss_cost(gts: pd.DataFrame, names: List[str], cfg: dict) -> pd.Series:
    """Per-GT miss cost: cost_miss * class weight * area factor."""
    return pd.Series([cfg["cost_miss"] * severity(names[int(r.class_id)], (r.x1, r.y1, r.x2, r.y2),
                                                  r.img_w, r.img_h, cfg)
                      for r in gts.itertuples()], index=gts.index, dtype=float)


OUTCOME_COLS = ["alerts", "reviews", "false_alarms", "gt", "caught", "miss_cost", "cost"]


def get_reviewer_accuracy(cfg: dict, override: Optional[float] = None) -> float:
    """Reviewer accuracy: explicit override, else cfg['reviewer_accuracy'], else 1.0. Must be in [0, 1]."""
    a = override if override is not None else cfg.get("reviewer_accuracy", 1.0)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"reviewer_accuracy must be in [0, 1], got {a}")
    return float(a)


def policy_outcomes(preds: pd.DataFrame, gts: pd.DataFrame, bands: Dict[int, Tuple[float, float]], cfg: dict,
                    names: List[str], reviewer_accuracy: Optional[float] = None,
                    images: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Expected per-image outcome of a per-class (lower, upper) policy. Only classes in `bands` count.

    conf >= upper            -> alert (no human). An unmatched alert is a false alarm.
    lower <= conf < upper    -> human review, costs cost_review. The reviewer is right with
                                probability `reviewer_accuracy` a, wrong with 1-a, so a reviewed
                                real defect is caught with prob. a and a reviewed false detection is
                                wrongly confirmed (a false alarm) with prob. 1-a.
    A GT never alerted/reviewed is missed for sure; a GT only reached by review is missed w.p. 1-a.
    A miss costs cost_miss * class weight * area factor. lower > upper is clipped to lower = upper
    (zero-width band: no review). a = 1 reproduces the perfect-reviewer model.

    Columns: alerts, reviews (counts), false_alarms (expected), gt, caught (expected), miss_cost, cost.
    """
    a = get_reviewer_accuracy(cfg, reviewer_accuracy)
    if images is None:
        images = sorted(set(preds.image) | set(gts.image))
    out = pd.DataFrame(0.0, index=pd.Index(images, name="image"), columns=OUTCOME_COLS)

    def add(col: str, imgs: pd.Series, vals) -> None:
        s = pd.Series(np.asarray(vals, float), index=imgs.to_numpy()).groupby(level=0).sum()
        out[col] = out[col].add(s.reindex(out.index).fillna(0.0))

    for c, (lo, hi) in bands.items():
        lo = min(lo, hi)
        p = preds[preds.class_id == c]
        alert = (p.conf >= hi).to_numpy()
        rev = ((p.conf >= lo).to_numpy()) & ~alert
        tp = p.tp.to_numpy(bool)
        add("alerts", p.image, alert)
        add("reviews", p.image, rev)
        add("false_alarms", p.image, (alert & ~tp) + (1.0 - a) * (rev & ~tp))
        g = gts[gts.class_id == c]
        if len(g):
            m = np.nan_to_num(g.match_conf.to_numpy(float), nan=-1.0)
            caught = (m >= hi) * 1.0 + ((m >= lo) & (m < hi)) * a
            add("gt", g.image, np.ones(len(g)))
            add("caught", g.image, caught)
            add("miss_cost", g.image, _miss_cost(g, names, cfg).to_numpy() * (1.0 - caught))
    out["cost"] = (cfg["cost_false_alarm"] * out.false_alarms + cfg["cost_review"] * out.reviews
                   + out.miss_cost)
    return out


def total_cost(preds: pd.DataFrame, gts: pd.DataFrame, thresholds: Dict[int, float], cfg: dict,
               names: List[str], review_low: Optional[Dict[int, float]] = None,
               reviewer_accuracy: Optional[float] = None) -> float:
    """
    Total expected cost on matched frames for per-class thresholds (see policy_outcomes).

    thresholds = upper (alert) bound per class; review_low = lower bound (default: no review band).
    reviewer_accuracy defaults to cfg['reviewer_accuracy'] (1.0 if absent).
    """
    review_low = review_low or {}
    bands = {c: (min(review_low.get(c, t), t), t) for c, t in thresholds.items()}
    return float(policy_outcomes(preds, gts, bands, cfg, names, reviewer_accuracy).cost.sum())


def optimize_thresholds(preds: pd.DataFrame, gts: pd.DataFrame, cfg: dict, names: List[str]) -> Dict[int, float]:
    """
    Per-class threshold minimising miss + false-alarm cost (classes are independent).

    Candidates are observed detection confidences plus "alert on nothing" (just above 1.0).
    Ties go to the higher threshold (fewer alerts).
    """
    out = {}
    for c in range(len(names)):
        p = preds[preds.class_id == c]
        g = gts[gts.class_id == c]
        mcost = _miss_cost(g, names, cfg).to_numpy()
        mconf = np.nan_to_num(g.match_conf.to_numpy(float), nan=-1.0)
        fa_conf = p.conf[~p.tp].to_numpy()
        best_t, best_cost = 1.0 + 1e-9, np.inf
        for t in np.unique(np.append(p.conf.to_numpy(), 1.0 + 1e-9)):
            cost = cfg["cost_false_alarm"] * (fa_conf >= t).sum() + mcost[mconf < t].sum()
            if cost <= best_cost:
                best_t, best_cost = float(t), cost
        out[c] = best_t
    return out


def review_bands(thresholds: Dict[int, float], cfg: dict) -> Dict[int, float]:
    """Lower review bound per class: upper threshold minus review_band_width, floored at 0."""
    return {c: max(t - cfg["review_band_width"], 0.0) for c, t in thresholds.items()}


def decide_action(conf: float, sev: float, review_low: float, review_high: float, cfg: dict) -> str:
    """
    Map one detection to an action.

    conf < review_low -> ignore; review_low <= conf < review_high -> human_review;
    conf >= review_high -> severity bands from action_bounds (lower bound inclusive).
    """
    if conf < review_low:
        return "ignore"
    if conf < review_high:
        return HUMAN_REVIEW
    b = cfg["action_bounds"]
    if sev < b["ignore_below"]:
        return "ignore"
    if sev < b["monitor_below"]:
        return "monitor"
    if sev < b["service_below"]:
        return "schedule_service"
    return "stop_now"


def main() -> None:
    """Print optimal per-class thresholds and costs on the cached val detections."""
    df = pd.read_parquet("data/cache/val_detections.parquet")
    cfg = load_cost_config()
    names = list(cfg["class_weights"])
    preds, gts = match_detections(df[df.kind == "pred"], df[df.kind == "gt"])
    th = optimize_thresholds(preds, gts, cfg, names)
    bands = review_bands(th, cfg)
    print("per-class thresholds (val-optimal upper; review-band lower):")
    for c, n in enumerate(names):
        print(f"  {n:16s} upper={th[c]:.4f} lower={bands[c]:.4f}")
    for label, t, b in [("all classes @0.25", {c: 0.25 for c in th}, None),
                        ("optimal", th, None), ("optimal + review band", th, bands),
                        ("no alerts at all", {c: 2.0 for c in th}, None)]:
        print(f"cost {label:24s} {total_cost(preds, gts, t, cfg, names, b):8.2f}")


if __name__ == "__main__":
    main()
