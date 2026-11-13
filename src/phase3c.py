"""Phase 3c: matched-workload baseline, cost-ratio sensitivity, pre-registration and a single test run.

Sub-commands (all read cached detections; only `test` reads the test cache):
  freeze  tune the primary/secondary policies and the matched flat baseline on ALL val, write
          reports/phase3/preregistration.md + frozen.json. Never reads test.
  val     out-of-fold matched-budget comparison + sensitivity grid + figures (val only).
  test    apply the frozen thresholds to the test detections ONCE (refuses to re-run).
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.decision import _miss_cost, load_cost_config, match_detections, policy_outcomes
from src.evaluate import GRID, OFF, Bands, bootstrap_saving, class_table, flat_bands, make_folds, tune_bands

FLAT_GRID = [float(round(x, 3)) for x in np.arange(0.001, 1.0, 0.001)] + [OFF]
BUDGETS_3C = (2, 3, 5)
ACCURACIES_3C = (0.9, 1.0, 0.8)           # primary first
RATIOS = (2, 3, 5, 10, 20, 50)
REVIEW_COSTS = (0.1, 0.3, 0.6, 1.0)
PRIMARY = {"a": 0.9, "budget": 3}
OUT_DIR = Path("reports/phase3")


# ---- flat (alerts only, one threshold for every class) ---------------------------------------

def flat_curve(preds: pd.DataFrame, gts: pd.DataFrame, cfg: dict, names: List[str], n_images: int,
               grid: Sequence[float] = FLAT_GRID) -> pd.DataFrame:
    """Cost and alerts/image of 'alert on conf >= t' for every t in grid (no review). Vectorised."""
    t = np.asarray(grid, float)
    conf = np.sort(preds.conf.to_numpy(float))
    fa = np.sort(preds.conf[~preds.tp.astype(bool)].to_numpy(float))
    m = np.nan_to_num(gts.match_conf.to_numpy(float), nan=-1.0)
    mc = _miss_cost(gts, names, cfg).to_numpy(float)
    order = np.argsort(m)
    cum = np.concatenate([[0.0], np.cumsum(mc[order])])
    alerts = len(conf) - np.searchsorted(conf, t, side="left")
    n_fa = len(fa) - np.searchsorted(fa, t, side="left")
    miss = cum[np.searchsorted(m[order], t, side="left")]
    return pd.DataFrame({"t": t, "cost": cfg["cost_false_alarm"] * n_fa + miss,
                         "alerts_per_image": alerts / n_images})


def best_flat_threshold(preds: pd.DataFrame, gts: pd.DataFrame, cfg: dict, names: List[str], n_images: int,
                        budget: float) -> float:
    """Cheapest single flat threshold with alerts/image <= budget on these images. Ties -> higher t."""
    cur = flat_curve(preds, gts, cfg, names, n_images)
    ok = cur[cur.alerts_per_image <= budget + 1e-9]          # OFF (0 alerts) is always feasible
    best = ok.sort_values(["cost", "t"], ascending=[True, False]).iloc[0]
    return float(best.t)


# ---- bootstrap -------------------------------------------------------------------------------

def bootstrap_metrics(out: pd.DataFrame, n_boot: int = 10000, seed: int = 0) -> Dict[str, Tuple[float, float, float]]:
    """Image-bootstrap percentile 95% CIs. Returns {metric: (point, lo, hi)}.

    Metrics: cost (total), cost_per_image, recall (pooled caught / gt), alerts_per_image, reviews_per_image.
    """
    rng = np.random.default_rng(seed)
    n = len(out)
    idx = rng.integers(0, n, size=(n_boot, n))
    cost, caught, gt = out.cost.to_numpy(), out.caught.to_numpy(), out["gt"].to_numpy()
    al, rv = out.alerts.to_numpy(), out.reviews.to_numpy()
    draws = {"cost": cost[idx].sum(1), "cost_per_image": cost[idx].mean(1),
             "recall": caught[idx].sum(1) / np.maximum(gt[idx].sum(1), 1e-12),
             "alerts_per_image": al[idx].mean(1), "reviews_per_image": rv[idx].mean(1)}
    point = {"cost": cost.sum(), "cost_per_image": cost.mean(), "recall": caught.sum() / gt.sum(),
             "alerts_per_image": al.mean(), "reviews_per_image": rv.mean()}
    return {k: (float(point[k]), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in draws.items()}


def _ci(t: Tuple[float, float, float], f: str = "{:.2f}") -> str:
    return f"{f.format(t[0])} [{f.format(t[1])}, {f.format(t[2])}]"


# ---- out-of-fold matched-budget comparison ---------------------------------------------------

def cross_validate_matched(preds: pd.DataFrame, gts: pd.DataFrame, cfg: dict, names: List[str],
                           budgets: Sequence[float] = BUDGETS_3C, accs: Sequence[float] = ACCURACIES_3C,
                           k: int = 5, seed: int = 0):
    """Tune on k-1 folds, score on the held-out fold, for the budgeted band policy and the matched flat
    baseline at each budget. Returns ({key: out-of-fold per-image outcomes}, {key: [per-fold tuned thing]}).
    Keys: ("band", a, B), ("flat", B), ("flat0.25",), ("none",)."""
    images = sorted(set(preds.image) | set(gts.image))
    folds = make_folds(gts, images, k, seed)
    nc = len(names)
    parts: Dict[tuple, list] = {}
    tuned: Dict[tuple, list] = {}

    def record(key, te, bands, a):
        o = policy_outcomes(preds[preds.image.isin(te)], gts[gts.image.isin(te)], bands, cfg, names, a,
                            images=sorted(te))
        parts.setdefault(key, []).append(o)

    for train, test in folds:
        tr, te = set(train), set(test)
        ptr, gtr = preds[preds.image.isin(tr)], gts[gts.image.isin(tr)]
        record(("none",), te, flat_bands(OFF, nc), 1.0)
        record(("flat0.25",), te, flat_bands(0.25, nc), 1.0)
        for B in budgets:
            t = best_flat_threshold(ptr, gtr, cfg, names, len(train), B)
            tuned.setdefault(("flat", B), []).append(t)
            record(("flat", B), te, flat_bands(t, nc), 1.0)
            for a in accs:
                bands = tune_bands(ptr, gtr, cfg, names, a, len(train), budget=B)
                tuned.setdefault(("band", a, B), []).append(bands)
                record(("band", a, B), te, bands, a)
    return {k_: pd.concat(v).loc[images] for k_, v in parts.items()}, tuned


def matched_table(oof: dict, budgets: Sequence[float], accs: Sequence[float], n_boot: int, seed: int) -> pd.DataFrame:
    """One row per (accuracy, budget): policy vs matched flat, with CIs."""
    rows = []
    for a in accs:
        for B in budgets:
            pol, flat = oof[("band", a, B)], oof[("flat", B)]
            mp, mf = bootstrap_metrics(pol, n_boot, seed), bootstrap_metrics(flat, n_boot, seed)
            s = bootstrap_saving(flat.cost.to_numpy(), pol.cost.to_numpy(), n_boot, seed)
            rows.append({"reviewer_accuracy": a, "budget": B,
                         "policy_cost": mp["cost"][0], "policy_cost_ci": _ci(mp["cost"], "{:.0f}"),
                         "policy_recall": mp["recall"][0], "policy_recall_ci": _ci(mp["recall"], "{:.3f}"),
                         "policy_alerts_pi": mp["alerts_per_image"][0], "policy_reviews_pi": mp["reviews_per_image"][0],
                         "flat_cost": mf["cost"][0], "flat_cost_ci": _ci(mf["cost"], "{:.0f}"),
                         "flat_recall": mf["recall"][0], "flat_recall_ci": _ci(mf["recall"], "{:.3f}"),
                         "flat_alerts_pi": mf["alerts_per_image"][0],
                         "saving": s["saving"], "saving_lo": s["saving_lo"], "saving_hi": s["saving_hi"],
                         "saving_pct": s["saving_pct"], "pct_lo": s["pct_lo"], "pct_hi": s["pct_hi"]})
    return pd.DataFrame(rows)


# ---- sensitivity ----------------------------------------------------------------------------

def with_costs(cfg: dict, ratio: float, review_cost: float) -> dict:
    """Copy of cfg with cost_miss = ratio * cost_false_alarm and the given review cost."""
    c = copy.deepcopy(cfg)
    c["cost_miss"] = ratio * c["cost_false_alarm"]
    c["cost_review"] = review_cost
    return c


def sensitivity(preds: pd.DataFrame, gts: pd.DataFrame, cfg: dict, names: List[str],
                ratios: Sequence[float] = RATIOS, review_costs: Sequence[float] = REVIEW_COSTS,
                a: float = PRIMARY["a"], budget: float = PRIMARY["budget"], k: int = 5, seed: int = 0,
                n_boot: int = 2000) -> pd.DataFrame:
    """Out-of-fold saving of the primary policy vs the matched flat baseline for each cost setting.
    Both are re-tuned inside every fold under each setting."""
    rows = []
    for r in ratios:
        for rc in review_costs:
            c = with_costs(cfg, r, rc)
            oof, _ = cross_validate_matched(preds, gts, c, names, budgets=[budget], accs=[a], k=k, seed=seed)
            pol, flat = oof[("band", a, budget)], oof[("flat", budget)]
            s = bootstrap_saving(flat.cost.to_numpy(), pol.cost.to_numpy(), n_boot, seed)
            rows.append({"miss_to_fa_ratio": r, "review_cost": rc, "policy_cost": pol.cost.sum(),
                         "flat_cost": flat.cost.sum(), "saving": s["saving"], "saving_lo": s["saving_lo"],
                         "saving_hi": s["saving_hi"], "saving_pct": s["saving_pct"], "pct_lo": s["pct_lo"],
                         "pct_hi": s["pct_hi"], "policy_recall": pol.caught.sum() / pol["gt"].sum(),
                         "flat_recall": flat.caught.sum() / flat["gt"].sum(),
                         "policy_reviews_pi": pol.reviews.mean()})
    return pd.DataFrame(rows)


def saving_status(row) -> str:
    """'significant' if the CI of the saving is entirely > 0, 'worse' if entirely < 0, else 'none'."""
    if row.saving_lo > 0:
        return "significant"
    if row.saving_hi < 0:
        return "worse"
    return "none"


# ---- frozen policy ---------------------------------------------------------------------------

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_policies(preds: pd.DataFrame, gts: pd.DataFrame, cfg: dict, names: List[str]) -> dict:
    """Tune everything that will be applied to test, on ALL val images."""
    n, B = len(set(preds.image) | set(gts.image)), PRIMARY["budget"]
    pol = {}
    for tag, a in (("primary", PRIMARY["a"]), ("secondary_a1.0", 1.0), ("secondary_a0.8", 0.8)):
        bands = tune_bands(preds, gts, cfg, names, a, n, budget=B)
        pol[tag] = {"reviewer_accuracy": a, "budget": B, "bands": {names[c]: list(map(float, b)) for c, b in bands.items()}}
    pol["flat_matched"] = {"budget": B, "threshold": best_flat_threshold(preds, gts, cfg, names, n, B)}
    pol["flat_0.25"] = {"threshold": 0.25}
    pol["n_val_images"] = n
    return pol


def bands_from_frozen(entry: dict, names: List[str]) -> Bands:
    return {i: tuple(entry["bands"][n]) for i, n in enumerate(names)}


def _fmt_band(b) -> str:
    f = lambda x: "off" if x >= OFF else f"{x:.2f}"
    return f(b[1]) if b[0] == b[1] else f"{f(b[0])}-{f(b[1])}"


def preregistration_text(frozen: dict, cfg: dict, names: List[str], hashes: dict, created: str, commit: str) -> str:
    cfg_yaml = json.dumps({k: cfg[k] for k in ("cost_miss", "cost_false_alarm", "cost_review", "class_weights",
                                               "area_severity", "review_band_width")}, indent=2)
    lines = [f"# Phase 3c pre-registration", "",
             f"Written {created} (UTC), before the test split was run through the model. Git HEAD at the time: `{commit}` "
             "(working tree has uncommitted Phase 3 code).", "",
             "## Primary policy", "",
             "- **Per-class review band** (lower, upper): `conf >= upper` alerts, `lower <= conf < upper` goes to a human, "
             "below `lower` is ignored.",
             "- Tuned **once, on all 216 val images**, by `src.evaluate.tune_bands(..., budget=3)` (exact multiple-choice-knapsack DP; "
             f"threshold grid `{[g for g in GRID]}` plus `off`).",
             "- **Workload budget B <= 3**: alerts + reviews per image <= 3 on the val images.",
             "- **reviewer_accuracy = 0.9** (symmetric; a missed-by-reviewer defect is a miss, a falsely confirmed detection is a false alarm).",
             "- Cost model: expected cost as in `src.decision.policy_outcomes` (IoU 0.5 greedy class-aware matching, "
             "miss cost scaled by class weight x area factor).", "",
             "### Frozen per-class bands (lower-upper)", "", "| policy | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]
    for tag in ("primary", "secondary_a1.0", "secondary_a0.8"):
        e = frozen[tag]
        lines.append(f"| {tag} (a={e['reviewer_accuracy']}, B<={e['budget']}) | "
                     + " | ".join(_fmt_band(e["bands"][n]) for n in names) + " |")
    lines += ["", "Exact values (JSON, also in `frozen.json`):", "", "```json", json.dumps(frozen, indent=2), "```", "",
              "## Secondary policies", "", "Same procedure and budget, with reviewer_accuracy = 1.0 and 0.8. Reported alongside the primary; "
              "they do not replace it.", "",
              "## Baselines on test", "",
              "1. `flat 0.25`: one confidence threshold 0.25 for every class, alerts only.",
              f"2. **Matched flat**: the single flat threshold (alerts only) with the lowest val cost among those with alerts/image <= 3 "
              f"on val; threshold = {frozen['flat_matched']['threshold']:.3f}, searched on a 0.001 grid.",
              "3. `no alerts`.", "",
              "## Test protocol", "",
              "- `python -m src.predict --split test --allow-test` is run once; `python -m src.phase3c test` applies the frozen "
              "thresholds above to the cached test detections. Nothing is retuned on test; the test command refuses to run twice.",
              "- Metrics: cost (total and per image), recall (pooled expected caught / GT), alerts and reviews per image, with 95% "
              "percentile CIs from a 10,000-draw image bootstrap (seed 0); savings are paired bootstraps.",
              "- Primary outcome: cost saving of the primary policy versus the matched flat baseline. It counts as supported only if "
              "its 95% CI lies entirely above 0.", "",
              "## Cost configuration (illustrative assumptions)", "", "```json", cfg_yaml, "```", "",
              "## Provenance hashes (SHA-256)", ""]
    lines += [f"- {k}: `{v}`" for k, v in hashes.items()]
    return "\n".join(lines) + "\n"


# ---- figures ---------------------------------------------------------------------------------

BLUE, ORANGE, INK, MUTED = "#2a6fbb", "#d9731a", "#222222", "#888888"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.titlecolor": INK, "text.color": INK})
    return plt


def plot_cost_vs_threshold(preds, gts, cfg, names, frozen_primary: dict, path) -> None:
    """Per class: cost of 'alert at conf >= t' (val, in-sample) with the frozen primary band shaded and its cost dashed."""
    plt = _plt()
    n = len(set(preds.image) | set(gts.image))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for c, (ax, name) in enumerate(zip(axes.ravel(), names)):
        p, g = preds[preds.class_id == c], gts[gts.class_id == c]
        cur = flat_curve(p, g, cfg, names, n, [x for x in FLAT_GRID if x < OFF])
        ax.plot(cur.t, cur.cost, color=BLUE, lw=2, label="alert-only threshold")
        lo, hi = frozen_primary["bands"][name]
        pc = class_table(preds, gts, c, cfg, names, frozen_primary["reviewer_accuracy"], [(lo, hi)]).cost.iloc[0]
        ax.axvspan(lo, hi, color=ORANGE, alpha=0.18, lw=0)
        ax.axhline(pc, color=ORANGE, ls="--", lw=1.5, label="cost of frozen primary band (shaded range)")
        ax.set_title(f"{name}  (band {lo:.2f}-{hi:.2f})")
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("confidence threshold")
    for ax in axes[:, 0]:
        ax.set_ylabel("cost on val images")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Per-class cost vs threshold (all val, in-sample)", y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_savings(val_table: pd.DataFrame, test_table: Optional[pd.DataFrame], path) -> None:
    """Forest plot of % saving vs the matched flat baseline, with 95% CIs."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2 if test_table is not None else 1, figsize=(11, 4.2), squeeze=False)
    ax = axes[0, 0]
    rows = val_table.sort_values(["reviewer_accuracy", "budget"], ascending=[False, False]).reset_index(drop=True)
    y = np.arange(len(rows))
    ax.errorbar(rows.saving_pct, y, xerr=[rows.saving_pct - rows.pct_lo, rows.pct_hi - rows.saving_pct], fmt="o",
                color=BLUE, ecolor=BLUE, capsize=3, ms=6)
    ax.set_yticks(y, [f"a={r.reviewer_accuracy:g}, B<={r.budget:g}" for r in rows.itertuples()])
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("% cost saving vs matched flat baseline")
    ax.set_title("Val, out-of-fold (95% CI)")
    ax.grid(axis="x", color="#e5e5e5", lw=0.6)
    if test_table is not None:
        ax = axes[0, 1]
        t = test_table.reset_index(drop=True)
        y = np.arange(len(t))
        ax.errorbar(t.saving_pct, y, xerr=[t.saving_pct - t.pct_lo, t.pct_hi - t.saving_pct], fmt="o", color=ORANGE,
                    ecolor=ORANGE, capsize=3, ms=6)
        ax.set_yticks(y, list(t.policy))
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlabel("% cost saving vs matched flat baseline")
        ax.set_title("Test, frozen val thresholds (95% CI)")
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(sens: pd.DataFrame, path) -> None:
    """Heatmap of % saving (primary vs matched flat) over miss:FA ratio x review cost. '*' = 95% CI excludes 0."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    ratios, rcs = sorted(sens.miss_to_fa_ratio.unique()), sorted(sens.review_cost.unique())
    pct = sens.pivot(index="miss_to_fa_ratio", columns="review_cost", values="saving_pct").loc[ratios, rcs]
    st = sens.assign(s=sens.apply(saving_status, axis=1)).pivot(index="miss_to_fa_ratio", columns="review_cost", values="s")
    lim = max(abs(pct.to_numpy()).max(), 1.0)
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#eeeeee", BLUE])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(pct.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", origin="lower")
    ax.set_xticks(range(len(rcs)), [f"{x:g}" for x in rcs])
    ax.set_yticks(range(len(ratios)), [f"{x:g}:1" for x in ratios])
    for i, r in enumerate(ratios):
        for j, rc in enumerate(rcs):
            mark = {"significant": "*", "worse": "†", "none": ""}[st.loc[r, rc]]
            ax.text(j, i, f"{pct.loc[r, rc]:.1f}%{mark}", ha="center", va="center", fontsize=9, color=INK)
    ax.set_xlabel("review cost (per reviewed detection)")
    ax.set_ylabel("miss : false-alarm cost ratio")
    ax.set_title("Saving of primary policy vs matched flat (B<=3, a=0.9)\n* CI > 0   † CI < 0   blank = CI spans 0", fontsize=9)
    fig.colorbar(im, ax=ax, label="% saving")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- commands --------------------------------------------------------------------------------

def _load(cache: str, cfg_path: str):
    df = pd.read_parquet(cache)
    cfg = load_cost_config(cfg_path)
    names = list(cfg["class_weights"])
    preds, gts = match_detections(df[df.kind == "pred"].reset_index(drop=True), df[df.kind == "gt"].reset_index(drop=True))
    return preds, gts, cfg, names


def cmd_freeze(a) -> None:
    preds, gts, cfg, names = _load(a.val_cache, a.config)
    frozen = freeze_policies(preds, gts, cfg, names)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frozen.json").exists() and not a.force:
        sys.exit("frozen.json already exists; pre-registration is not overwritten (use --force only before test is run)")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    hashes = {"configs/cost.yaml": sha256(a.config), "models/best.pt": sha256(a.model), "val detections cache": sha256(a.val_cache),
              "src/decision.py": sha256("src/decision.py"), "src/evaluate.py": sha256("src/evaluate.py")}
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    (out / "frozen.json").write_text(json.dumps(frozen, indent=2))
    (out / "preregistration.md").write_text(preregistration_text(frozen, cfg, names, hashes, created, commit), encoding="utf-8")
    print((out / "preregistration.md").read_text(encoding="utf-8"))


def cmd_val(a) -> None:
    preds, gts, cfg, names = _load(a.val_cache, a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    oof, tuned = cross_validate_matched(preds, gts, cfg, names, k=a.folds, seed=a.seed)
    tab = matched_table(oof, BUDGETS_3C, ACCURACIES_3C, a.n_boot, a.seed)
    tab.to_csv(out / "matched_baseline_val.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 40)
    print("Matched flat threshold chosen in each CV training fold:")
    for B in BUDGETS_3C:
        print(f"  B<={B}: {tuned[('flat', B)]}")
    print("\nOut-of-fold matched-workload comparison (val, 216 images, cost = total, 95% image-bootstrap CI):")
    show = tab[["reviewer_accuracy", "budget", "policy_cost", "policy_cost_ci", "flat_cost", "flat_cost_ci", "policy_recall_ci",
                "flat_recall_ci", "policy_alerts_pi", "policy_reviews_pi", "flat_alerts_pi"]]
    print(show.round(3).to_string(index=False))
    sv = tab[["reviewer_accuracy", "budget", "saving", "saving_lo", "saving_hi", "saving_pct", "pct_lo", "pct_hi"]]
    print("\nSaving vs matched flat:")
    print(sv.round(1).to_string(index=False))

    sens = sensitivity(preds, gts, cfg, names, k=a.folds, seed=a.seed, n_boot=a.n_boot_sens)
    sens["status"] = sens.apply(saving_status, axis=1)
    sens.to_csv(out / "sensitivity.csv", index=False)
    print("\nSensitivity: % saving of primary (a=0.9, B<=3) vs matched flat (rows ratio, cols review cost):")
    print(sens.pivot(index="miss_to_fa_ratio", columns="review_cost", values="saving_pct").round(1).to_string())
    print("\nCI status (significant = CI>0, worse = CI<0, none = spans 0):")
    print(sens.pivot(index="miss_to_fa_ratio", columns="review_cost", values="status").to_string())
    print("\nFull sensitivity rows:")
    print(sens[["miss_to_fa_ratio", "review_cost", "policy_cost", "flat_cost", "saving", "saving_lo", "saving_hi",
                "saving_pct", "pct_lo", "pct_hi"]].round(1).to_string(index=False))

    frozen = json.loads((out / "frozen.json").read_text())
    plot_cost_vs_threshold(preds, gts, cfg, names, frozen["primary"], out / "cost_vs_threshold.png")
    plot_heatmap(sens, out / "sensitivity_heatmap.png")
    test_tab = out / "test_savings.csv"
    plot_savings(tab, pd.read_csv(test_tab) if test_tab.exists() else None, out / "savings_ci.png")


def cmd_test(a) -> None:
    out = Path(a.out)
    frozen_path, res_path = out / "frozen.json", out / "test_results.csv"
    if not frozen_path.exists() or not (out / "preregistration.md").exists():
        sys.exit("no pre-registration found; run `freeze` first")
    if res_path.exists():
        sys.exit("test_results.csv exists: the test split is evaluated once, not repeatedly")
    frozen = json.loads(frozen_path.read_text())
    preds, gts, cfg, names = _load(a.test_cache, a.config)
    nc = len(names)
    images = sorted(set(preds.image) | set(gts.image))
    pols = {"primary (a=0.9, B<=3)": (bands_from_frozen(frozen["primary"], names), 0.9),
            "secondary a=1.0": (bands_from_frozen(frozen["secondary_a1.0"], names), 1.0),
            "secondary a=0.8": (bands_from_frozen(frozen["secondary_a0.8"], names), 0.8),
            "flat 0.25": (flat_bands(0.25, nc), 1.0),
            f"matched flat ({frozen['flat_matched']['threshold']:.3f})": (flat_bands(frozen["flat_matched"]["threshold"], nc), 1.0),
            "no alerts": (flat_bands(OFF, nc), 1.0)}
    outs = {k: policy_outcomes(preds, gts, b, cfg, names, acc, images=images) for k, (b, acc) in pols.items()}
    matched_key = [k for k in outs if k.startswith("matched flat")][0]
    rows = []
    for k, o in outs.items():
        m = bootstrap_metrics(o, a.n_boot, a.seed)
        r = {"policy": k, **{f"{n}": m[n][0] for n in m}, **{f"{n}_ci": _ci(m[n], "{:.3f}") for n in m}}
        for tag, base in (("flat025", outs["flat 0.25"]), ("matched", outs[matched_key]), ("none", outs["no alerts"])):
            s = bootstrap_saving(base.cost.to_numpy(), o.cost.to_numpy(), a.n_boot, a.seed)
            r.update({f"saving_vs_{tag}": s["saving"], f"saving_vs_{tag}_lo": s["saving_lo"], f"saving_vs_{tag}_hi": s["saving_hi"],
                      f"pct_vs_{tag}": s["saving_pct"], f"pct_vs_{tag}_lo": s["pct_lo"], f"pct_vs_{tag}_hi": s["pct_hi"]})
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(res_path, index=False)
    sv = res.rename(columns={"pct_vs_matched": "saving_pct", "pct_vs_matched_lo": "pct_lo", "pct_vs_matched_hi": "pct_hi"})
    sv = sv[sv.policy.str.startswith(("primary", "secondary", "flat 0.25"))][["policy", "saving_pct", "pct_lo", "pct_hi"]]
    sv.to_csv(out / "test_savings.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 50)
    print(f"{len(images)} test images, {int(outs['no alerts']['gt'].sum())} GT boxes\n")
    print(res[["policy", "cost_per_image", "cost_per_image_ci", "recall", "recall_ci", "alerts_per_image",
               "alerts_per_image_ci", "reviews_per_image", "reviews_per_image_ci"]].round(3).to_string(index=False))
    print("\nTotal cost on test and saving (95% CI) vs each baseline:")
    for tag in ("flat025", "matched", "none"):
        print(f"\n  vs {tag}:")
        t = res[["policy", "cost", f"saving_vs_{tag}", f"saving_vs_{tag}_lo", f"saving_vs_{tag}_hi", f"pct_vs_{tag}",
                 f"pct_vs_{tag}_lo", f"pct_vs_{tag}_hi"]]
        print(t.round(1).to_string(index=False))
    # refresh savings figure with the test panel
    vt = out / "matched_baseline_val.csv"
    if vt.exists():
        plot_savings(pd.read_csv(vt), sv, out / "savings_ci.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["freeze", "val", "test"])
    ap.add_argument("--val-cache", default="data/cache/val_detections.parquet")
    ap.add_argument("--test-cache", default="data/cache/test_detections.parquet")
    ap.add_argument("--config", default="configs/cost.yaml")
    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-boot-sens", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"freeze": cmd_freeze, "val": cmd_val, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
