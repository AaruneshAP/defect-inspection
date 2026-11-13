"""Phase 3b: honest evaluation of review-band policies with cross-validation on val images only.

Policies are per-class (lower, upper) bands: conf >= upper alerts, lower <= conf < upper goes to a
human reviewer, below lower is ignored. lower == upper is a zero-width band (no review).
Everything here reads the cached *val* detections; the test split is never opened.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.decision import (_miss_cost, get_reviewer_accuracy, load_cost_config, match_detections,
                          policy_outcomes)

OFF = 1.01  # a threshold above every possible confidence: "never"
GRID = [0.002, 0.005, 0.01, 0.02, 0.05] + [float(round(x, 2)) for x in np.arange(0.10, 0.951, 0.05)]  # coarse threshold grid
BUDGETS = (1, 2, 3, 5)
ACCURACIES = (1.0, 0.9, 0.8)
Bands = Dict[int, Tuple[float, float]]


def band_pairs(grid: List[float] = GRID, allow_band: bool = True) -> List[Tuple[float, float]]:
    """All (lower, upper) with lower <= upper on grid + OFF. allow_band=False keeps lower == upper."""
    vals = list(grid) + [OFF]
    return [(lo, hi) for hi in vals for lo in vals if lo == hi or (allow_band and lo < hi)]


def class_table(preds: pd.DataFrame, gts: pd.DataFrame, c: int, cfg: dict, names: List[str], a: float,
                pairs: List[Tuple[float, float]]) -> pd.DataFrame:
    """Expected cost and load (alerts + reviews) of every (lo, hi) pair for one class. Vectorised
    twin of policy_outcomes; tests assert they agree."""
    p = preds[preds.class_id == c]
    g = gts[gts.class_id == c]
    conf, tp = p.conf.to_numpy(float), p.tp.to_numpy(bool)
    m = np.nan_to_num(g.match_conf.to_numpy(float), nan=-1.0)
    mc = _miss_cost(g, names, cfg).to_numpy(float)
    rows = []
    for lo, hi in pairs:
        alert = conf >= hi
        rev = (conf >= lo) & ~alert
        caught = (m >= hi) * 1.0 + ((m >= lo) & (m < hi)) * a
        cost = (cfg["cost_false_alarm"] * ((alert & ~tp).sum() + (1 - a) * (rev & ~tp).sum())
                + cfg["cost_review"] * rev.sum() + (mc * (1 - caught)).sum())
        rows.append((lo, hi, round(float(cost), 9), int(alert.sum() + rev.sum())))
    return pd.DataFrame(rows, columns=["lo", "hi", "cost", "load"])


def _best_per_class(tab: pd.DataFrame) -> Tuple[float, float]:
    """Minimum-cost option; ties go to the lower load, then the higher upper, then higher lower bound."""
    r = tab.sort_values(["cost", "load", "hi", "lo"], ascending=[True, True, False, False]).iloc[0]
    return float(r.lo), float(r.hi)


def _budget_dp(tables: List[pd.DataFrame], cap: int) -> List[int]:
    """
    Exact multiple-choice knapsack: pick one option row per class minimising total cost with
    total load <= cap. Returns the chosen row position in each table. DP over integer load.
    """
    inf = np.inf
    f = np.zeros(cap + 1)  # f[L]: best cost of the classes processed so far with total load <= L
    choices = []
    for tab in tables:
        # cheapest option for each exact load l <= cap
        best = {}
        for pos, (cost, load) in enumerate(zip(tab.cost.to_numpy(), tab.load.to_numpy())):
            if load <= cap and (load not in best or cost < best[load][0]):
                best[load] = (cost, pos)
        nf = np.full(cap + 1, inf)
        arg_l = np.zeros(cap + 1, int)
        arg_pos = np.zeros(cap + 1, int)
        for load in sorted(best):
            cost, pos = best[load]
            cand = np.full(cap + 1, inf)
            cand[load:] = f[: cap + 1 - load] + cost
            better = cand < nf - 1e-12
            nf[better], arg_l[better], arg_pos[better] = cand[better], load, pos
        choices.append((arg_l, arg_pos))
        f = nf
    picked, L = [], cap
    for arg_l, arg_pos in reversed(choices):
        picked.append(int(arg_pos[L]))
        L -= int(arg_l[L])
    return picked[::-1]


def tune_bands(preds: pd.DataFrame, gts: pd.DataFrame, cfg: dict, names: List[str], a: float,
               n_images: int, allow_band: bool = True, budget: Optional[float] = None) -> Bands:
    """
    Jointly tune (lower, upper) per class on the grid. Without a budget the classes are independent.
    With `budget` B: minimise total cost subject to alerts + reviews <= B per image on these images.
    """
    pairs = band_pairs(allow_band=allow_band)
    tabs = [class_table(preds, gts, c, cfg, names, a, pairs) for c in range(len(names))]
    if budget is None:
        return {c: _best_per_class(t) for c, t in enumerate(tabs)}
    # same tie-break as the unbudgeted search: the DP keeps the first of equal-cost options (higher upper, then lower)
    tabs = [t.sort_values(["hi", "lo"], ascending=False).reset_index(drop=True) for t in tabs]
    picked = _budget_dp(tabs, int(np.floor(budget * n_images + 1e-9)))
    return {c: (float(tabs[c].lo[i]), float(tabs[c].hi[i])) for c, i in enumerate(picked)}


def flat_bands(t: float, n_classes: int) -> Bands:
    return {c: (t, t) for c in range(n_classes)}


def summarise(out: pd.DataFrame) -> dict:
    """Pool per-image outcomes into the reported metrics."""
    n = len(out)
    return {"cost": out.cost.sum(), "cost_per_image": out.cost.sum() / n,
            "recall": out.caught.sum() / out["gt"].sum(), "alerts_per_image": out.alerts.sum() / n,
            "reviews_per_image": out.reviews.sum() / n, "false_alarms_per_image": out.false_alarms.sum() / n}


def make_folds(gts: pd.DataFrame, images: List[str], k: int = 5, seed: int = 0) -> List[Tuple[List[str], List[str]]]:
    """Stratified (by each image's dominant GT class) k folds over images -> [(train, test)]."""
    dom = gts.groupby("image").class_id.agg(lambda s: s.value_counts().idxmax()).reindex(images).fillna(-1)
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    arr = np.array(images)
    return [(list(arr[tr]), list(arr[te])) for tr, te in skf.split(arr, dom.to_numpy())]


def bootstrap_saving(base: np.ndarray, pol: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired image bootstrap of the cost saving (base - policy). Percentile 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(base)
    idx = rng.integers(0, n, size=(n_boot, n))
    sav = (base - pol)[idx].sum(axis=1)
    rel = sav / base[idx].sum(axis=1)
    return {"saving": float((base - pol).sum()), "saving_lo": float(np.percentile(sav, 2.5)),
            "saving_hi": float(np.percentile(sav, 97.5)), "saving_pct": float(100 * (base - pol).sum() / base.sum()),
            "pct_lo": float(100 * np.percentile(rel, 2.5)), "pct_hi": float(100 * np.percentile(rel, 97.5))}


def policy_specs(cfg: dict, names: List[str]) -> Dict[str, Tuple[Callable, float]]:
    """name -> (tuner(train_preds, train_gts, n_train_images) -> bands, reviewer accuracy)."""
    k = len(names)
    specs: Dict[str, Tuple[Callable, float]] = {
        "no alerts": (lambda p, g, n: flat_bands(OFF, k), 1.0),
        "flat 0.25": (lambda p, g, n: flat_bands(0.25, k), 1.0),
        "tuned thresholds, no band": (lambda p, g, n: tune_bands(p, g, cfg, names, 1.0, n, allow_band=False), 1.0),
    }
    for a in ACCURACIES:
        specs[f"tuned band, acc={a:g}"] = (lambda p, g, n, a=a: tune_bands(p, g, cfg, names, a, n), a)
        for B in BUDGETS:
            specs[f"tuned band, acc={a:g}, budget<={B}/img"] = (
                lambda p, g, n, a=a, B=B: tune_bands(p, g, cfg, names, a, n, budget=B), a)
    return specs


def cross_validate(preds: pd.DataFrame, gts: pd.DataFrame, cfg: dict, names: List[str], k: int = 5, seed: int = 0):
    """Tune on k-1 folds, score on the held-out fold. Returns ({policy: out-of-fold per-image outcomes},
    {policy: [bands per fold]})."""
    images = sorted(set(preds.image) | set(gts.image))
    folds = make_folds(gts, images, k, seed)
    oof, fold_bands = {}, {}
    for name, (tuner, a) in policy_specs(cfg, names).items():
        parts, bl = [], []
        for train, test in folds:
            tr, te = set(train), set(test)
            bands = tuner(preds[preds.image.isin(tr)], gts[gts.image.isin(tr)], len(train))
            bl.append(bands)
            parts.append(policy_outcomes(preds[preds.image.isin(te)], gts[gts.image.isin(te)], bands, cfg, names,
                                         a, images=sorted(te)))
        oof[name], fold_bands[name] = pd.concat(parts).loc[images], bl
    return oof, fold_bands


def _fmt(b: Tuple[float, float]) -> str:
    f = lambda x: "off" if x >= OFF else f"{x:.2f}"
    return f"{f(b[0])}-{f(b[1])}" if b[0] != b[1] else f"{f(b[1])}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/cache/val_detections.parquet")
    ap.add_argument("--config", default="configs/cost.yaml")
    ap.add_argument("--out", default="reports/phase3b")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if "test" in Path(a.cache).name.lower():
        ap.error("phase 3b evaluates on val only; the test split is reserved")

    df = pd.read_parquet(a.cache)
    cfg = load_cost_config(a.config)
    names = list(cfg["class_weights"])
    preds, gts = match_detections(df[df.kind == "pred"], df[df.kind == "gt"])
    images = sorted(set(df.image))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    oof, fold_bands = cross_validate(preds, gts, cfg, names, a.folds, a.seed)
    base_none, base_flat = oof["no alerts"].cost.to_numpy(), oof["flat 0.25"].cost.to_numpy()
    rows = []
    for name, o in oof.items():
        r = {"policy": name, **summarise(o)}
        for tag, base in (("none", base_none), ("flat", base_flat)):
            b = bootstrap_saving(base, o.cost.to_numpy(), a.n_boot, a.seed)
            r.update({f"sav_vs_{tag}": b["saving"], f"sav_vs_{tag}_ci": f"[{b['saving_lo']:.1f}, {b['saving_hi']:.1f}]",
                      f"pct_vs_{tag}": b["saving_pct"], f"pct_vs_{tag}_ci": f"[{b['pct_lo']:.1f}, {b['pct_hi']:.1f}]"})
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(out / "policy_table.csv", index=False)

    # value of the review band itself: band policy vs tuned thresholds without a band
    nb = oof["tuned thresholds, no band"].cost.to_numpy()
    band_rows = []
    for acc in ACCURACIES:
        b = bootstrap_saving(nb, oof[f"tuned band, acc={acc:g}"].cost.to_numpy(), a.n_boot, a.seed)
        band_rows.append({"reviewer_accuracy": acc, "saving_vs_no_band": b["saving"],
                          "ci95": f"[{b['saving_lo']:.1f}, {b['saving_hi']:.1f}]", "pct": b["saving_pct"],
                          "pct_ci95": f"[{b['pct_lo']:.1f}, {b['pct_hi']:.1f}]"})
    band_value = pd.DataFrame(band_rows)
    band_value.to_csv(out / "band_vs_no_band.csv", index=False)

    # thresholds tuned on ALL val images (descriptive, in-sample) + stability across CV folds
    th_rows, stab_rows = [], []
    for name, (tuner, acc) in policy_specs(cfg, names).items():
        if name in ("no alerts", "flat 0.25"):
            continue
        bands = tuner(preds, gts, len(images))
        th_rows.append({"policy": name, **{names[c]: _fmt(b) for c, b in bands.items()}})
        if name in ("tuned thresholds, no band", "tuned band, acc=1"):
            for c, n in enumerate(names):
                his = [fb[c][1] for fb in fold_bands[name]]
                los = [fb[c][0] for fb in fold_bands[name]]
                stab_rows.append({"policy": name, "class": n, "upper_across_folds": " ".join(_fmt((h, h)) for h in his),
                                  "lower_across_folds": " ".join(_fmt((l, l)) for l in los)})
    thr = pd.DataFrame(th_rows)
    thr.to_csv(out / "thresholds_full_val.csv", index=False)
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(out / "threshold_stability.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    print(f"{len(images)} val images, {a.folds}-fold CV (stratified), grid={GRID}, bootstrap n={a.n_boot}\n")
    print("Per-class (lower-upper) thresholds, tuned on ALL val images (in-sample, descriptive; single value = no band):")
    print(thr.to_string(index=False))
    print("\nThreshold stability across the CV training folds:")
    print(stab.to_string(index=False))
    cols = ["policy", "cost", "recall", "alerts_per_image", "reviews_per_image", "false_alarms_per_image"]
    print("\nOut-of-fold policy table (cost = total over all val images, each scored by a model that never saw it):")
    print(table[cols].round(3).to_string(index=False))
    print("\nCost saving vs baselines, total over val images, bootstrap 95% CI (resampling images):")
    print(table[["policy", "sav_vs_none", "sav_vs_none_ci", "pct_vs_none", "sav_vs_flat", "sav_vs_flat_ci",
                 "pct_vs_flat", "pct_vs_flat_ci"]].round(1).to_string(index=False))
    print("\nValue of the review band alone (vs tuned thresholds with no band):")
    print(band_value.round(2).to_string(index=False))
    vio = {n: o.loc[:, ["alerts", "reviews"]].sum(axis=1).mean() for n, o in oof.items() if "budget" in n}
    print("\nRealised held-out load (alerts+reviews per image) of budgeted policies:")
    for n, v in vio.items():
        print(f"  {n:40s} {v:.2f}")


if __name__ == "__main__":
    main()
