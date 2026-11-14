"""Two-tier alerts on the SIMULATED inspection stream (Phase 7b), built on the drift flags of src.drift.

Tier 1 "input drift alert" (informational): at least `min_flagged` of the 6 features flagged in a window.
Tier 2 "retrain recommended": in at least k of the last m windows, an input statistic AND a prediction feature
(max_confidence or n_detections) are flagged together. k and m are chosen on the calibration seed (100), scored on 200.

  python -m src.triggers          # needs data/inspection/drift_pool.parquet (python -m src.drift pool)
"""

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src import drift as D
from src.drift import (CALIB_SEED, EVAL_SEED, HARMFUL_KINDS, INPUT_FEATURES, DriftConfig, EvalSetup,
                       alert_from_flags, censored_mean, detection_delay, episodes, flag_features)


@dataclass(frozen=True)
class TriggerConfig:
    """Base drift config (window, alpha, Holm) + tier-1 rule + tier-2 k-of-m."""

    drift: DriftConfig = DriftConfig(window=30, alpha=0.001, min_flagged=1)
    k: int = 5
    m: int = 30


def window_flags(pvals: pd.DataFrame, cfg: DriftConfig) -> pd.DataFrame:
    """Per-feature drift flags for each window (Holm across all 6 features)."""
    return pd.DataFrame(flag_features(pvals.to_numpy(), cfg.alpha, cfg.correction), index=pvals.index,
                        columns=pvals.columns)


def tier1(flags: pd.DataFrame, min_flagged: int) -> pd.Series:
    """Input drift alert: at least `min_flagged` features flagged."""
    return pd.Series(alert_from_flags(flags.to_numpy(), min_flagged), index=flags.index, name="tier1")


def joint_flags(flags: pd.DataFrame) -> pd.Series:
    """True where some input statistic and some prediction feature are flagged in the same window."""
    return pd.Series(flags[INPUT_FEATURES].any(axis=1) & flags[D.PREDICTION_FEATURES].any(axis=1), index=flags.index)


def tier2(flags: pd.DataFrame, k: int, m: int) -> pd.Series:
    """Retrain recommended: joint input+prediction drift in at least k of the last m windows (full history only)."""
    if not 1 <= k <= m:
        raise ValueError("need 1 <= k <= m")
    count = joint_flags(flags).astype(int).rolling(m).sum()
    return (count >= k).rename("tier2")


def _flag_cache(pcache: dict, cfg: DriftConfig) -> dict:
    return {key: [window_flags(p, cfg) for p in v] for key, v in pcache.items()}


def _rates(series: Sequence[pd.Series]) -> dict:
    n = sum(len(s) for s in series)
    return {"episodes_per_1000": 1000.0 * sum(episodes(s) for s in series) / n,
            "window_rate": float(sum(s.sum() for s in series) / n),
            "streams_with_alert": float(np.mean([s.any() for s in series]))}


def score(fcache: dict, rule, setup: EvalSetup) -> dict:
    """rule(flags) -> bool Series. Clean false-trigger rates plus trigger rate and delay per (kind, severity)."""
    clean = _rates([rule(f) for f in fcache["clean"]])
    per = {}
    for key, fl in fcache.items():
        if key == "clean":
            continue
        ds = [detection_delay(rule(f), setup.drift_start) for f in fl]
        det = [d for d in ds if d is not None]
        per[key] = {"trigger_rate": len(det) / len(ds), "delay_median": float(np.median(det)) if det else None,
                    "delay_p90": float(np.percentile(det, 90)) if det else None,
                    "delay_censored": censored_mean(ds, setup.horizon), "n_streams": len(ds)}
    return {"clean": clean, "per": per}


def select_km(fcache: dict, setup: EvalSetup, ms=(10, 30, 60), ks=(1, 3, 5, 10, 20, 30, 60),
              max_fa_per_1000: float = 0.0, base: DriftConfig = TriggerConfig().drift):
    """Pre-stated rule: among (k, m) with calibration false-trigger episodes <= max_fa_per_1000 (default none),
    take the lowest mean delay over blur/noise/contrast (not triggered = horizon); ties: fewer false triggers, smaller m."""
    rows = []
    for m, k in product(ms, ks):
        if k > m:
            continue
        s = score(fcache, lambda f: tier2(f, k, m), setup)
        harm = [s["per"][key] for key in s["per"] if key[0] in HARMFUL_KINDS]
        rows.append({"m": m, "k": k, "fa_episodes_per_1000": s["clean"]["episodes_per_1000"],
                     "trigger_rate_harmful": float(np.mean([h["trigger_rate"] for h in harm])),
                     "mean_delay_harmful": float(np.mean([h["delay_censored"] for h in harm]))})
    grid = pd.DataFrame(rows)
    grid["feasible"] = grid.fa_episodes_per_1000 <= max_fa_per_1000
    pick = grid[grid.feasible].sort_values(["mean_delay_harmful", "fa_episodes_per_1000", "m"])
    if pick.empty:
        raise RuntimeError("no (k, m) meets the false-trigger budget on the calibration stream")
    best = pick.iloc[0]
    return TriggerConfig(base, int(best.k), int(best.m)), grid


def table(sc: dict) -> pd.DataFrame:
    rows = [{"perturbation": k[0], "severity": k[1], **v} for k, v in sc["per"].items()]
    return pd.DataFrame(rows)


def run(pool: pd.DataFrame, out_dir: Path, max_fa: float = 0.0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = TriggerConfig().drift
    cal_setup, ev_setup = EvalSetup(), EvalSetup(n_clean=20, n_drift_reps=30)
    cal_f = _flag_cache(D._pvals_cache(pool, CALIB_SEED, [base.window], cal_setup)[base.window], base)
    cfg, grid = select_km(cal_f, cal_setup, max_fa_per_1000=max_fa, base=base)
    ev_p = D._pvals_cache(pool, EVAL_SEED, [base.window], ev_setup)[base.window]
    ev_f = _flag_cache(ev_p, base)
    s2 = score(ev_f, lambda f: tier2(f, cfg.k, cfg.m), ev_setup)

    variants = {"k1=1 (alpha .001)": (0.001, 1), "k1=2 (alpha .001)": (0.001, 2), "k1=2 (alpha .01)": (0.01, 2)}
    t1_rows = []
    for name, (alpha, k1) in variants.items():
        vc = DriftConfig(window=base.window, alpha=alpha, min_flagged=k1)
        sc = score(_flag_cache(ev_p, vc), lambda f, k1=k1: tier1(f, k1), ev_setup)
        t = table(sc)
        b = t[t.perturbation == "brightness"]
        h = t[t.perturbation.isin(HARMFUL_KINDS)]
        t1_rows.append({"variant": name, "alpha": alpha, "min_flagged": k1,
                        "fa_episodes_per_1000": sc["clean"]["episodes_per_1000"],
                        "fa_streams_with_alert": sc["clean"]["streams_with_alert"],
                        "brightness_alert_rate": float(b.trigger_rate.mean()),
                        "brightness_delay_median": float(b.delay_median.median()),
                        "harmful_alert_rate": float(h.trigger_rate.mean()),
                        "harmful_delay_median": float(h.delay_median.median()),
                        **{f"delay_{r.perturbation}{r.severity}": r.delay_median for r in t.itertuples()}})
    t1 = pd.DataFrame(t1_rows)

    t2 = table(s2)
    t2["tier1_alert_rate"] = [score(ev_f, lambda f: tier1(f, base.min_flagged), ev_setup)["per"][(r.perturbation, r.severity)]
                              ["trigger_rate"] for r in t2.itertuples()]
    grid.to_csv(out_dir / "triggers_calibration_grid.csv", index=False)
    t2.to_csv(out_dir / "triggers_eval.csv", index=False)
    t1.to_csv(out_dir / "tier1_comparison.csv", index=False)
    summary = {"simulated": True, "calibration_seed": CALIB_SEED, "eval_seed": EVAL_SEED, "chosen": asdict(cfg),
               "max_fa_per_1000_calibration": max_fa, "eval_setup": asdict(ev_setup),
               "tier2_clean": s2["clean"], "tier2_by_condition": t2.to_dict("records"),
               "tier1_comparison": t1.to_dict("records")}
    (out_dir / "triggers_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


def plot_summary(t2: pd.DataFrame, path: Path) -> None:
    """Tier-1 alert rate vs tier-2 retrain-trigger rate per (perturbation, severity); brightness is hatched."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r.perturbation} {r.severity}" for r in t2.itertuples()]
    x = np.arange(len(t2))
    fig, ax = plt.subplots(figsize=(10, 4))
    b1 = ax.bar(x - 0.2, t2.tier1_alert_rate, 0.4, label="tier 1: input drift alert", color="#9ecae1")
    b2 = ax.bar(x + 0.2, t2.trigger_rate, 0.4, label="tier 2: retrain recommended", color="#08519c")
    for i, r in enumerate(t2.itertuples()):
        if r.perturbation == "brightness":
            b1[i].set_hatch("//")
            b2[i].set_hatch("//")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("fraction of 30 streams triggering within 300 events")
    ax.set_title("SIMULATED streams, seed 200. Hatched = brightness (model is robust to it)")
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(D.POOL_PATH))
    ap.add_argument("--out-dir", default=str(D.OUT_DIR))
    ap.add_argument("--max-fa", type=float, default=0.0, help="calibration false-trigger episodes per 1000 events")
    a = ap.parse_args()
    s = run(pd.read_parquet(a.pool), Path(a.out_dir), a.max_fa)
    plot_summary(pd.DataFrame(s["tier2_by_condition"]), Path(a.out_dir) / "triggers_summary.png")
    print("chosen:", s["chosen"]["k"], "of", s["chosen"]["m"], "| clean false triggers:", s["tier2_clean"])
    print(pd.DataFrame(s["tier2_by_condition"]).round(3).to_string())
    print(pd.DataFrame(s["tier1_comparison"]).round(3).T.to_string())


if __name__ == "__main__":
    main()
