"""Drift detection on the SIMULATED inspection log (Phase 7a): reference window vs sliding current window.

Features: 4 input statistics (input drift) + max_confidence, n_detections (prediction drift). Per window: a two-sample
KS test per feature, Holm correction across the 6 features, and an alert when >= `min_flagged` features are flagged.
Sliding windows overlap, so no p-value is valid across time; the false-alarm rate is measured on clean streams and
(window, alpha, min_flagged) is chosen on a calibration seed, then scored on a different seed.
Streams come from a pool of val-image features (same perturbation and statistics code as the simulator).

  python -m src.drift pool                 # build data/inspection/drift_pool.parquet (runs the model; val images only)
  python -m src.drift evaluate             # calibrate on seed 100, evaluate on seed 200 -> reports/phase7/
  python -m src.drift report --log data/inspection/sim.parquet   # Evidently report for a simulator log
"""

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import kolmogorov

from src.inspection_log import STAT_COLUMNS
from src.perturb import LEVELS

INPUT_FEATURES: List[str] = list(STAT_COLUMNS)
PREDICTION_FEATURES: List[str] = ["max_confidence", "n_detections"]
FEATURES: List[str] = INPUT_FEATURES + PREDICTION_FEATURES
KINDS = ("blur", "noise", "contrast", "brightness")
HARMFUL_KINDS = ("blur", "noise", "contrast")  # brightness barely moves mAP (Phase 4), so it is not tuned for
OUT_DIR = Path("reports/phase7")
POOL_PATH = Path("data/inspection/drift_pool.parquet")
CALIB_SEED, EVAL_SEED = 100, 200


@dataclass(frozen=True)
class DriftConfig:
    """Window sizes in events; `alpha` is the per-window family-wise level over the features."""

    window: int = 50
    alpha: float = 0.01
    min_flagged: int = 2
    ref_size: int = 200
    correction: str = "holm"  # 'holm' or 'none'


def reference_window(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """The first n events (assumed clean)."""
    if len(df) < n:
        raise ValueError(f"need at least {n} events for the reference window, got {len(df)}")
    return df.iloc[:n]


def window_ends(n_events: int, cfg: DriftConfig) -> range:
    """Index of the last event of each evaluable window; windows never overlap the reference."""
    return range(cfg.ref_size + cfg.window - 1, n_events)


def _ks_p(ref_sorted: np.ndarray, x: np.ndarray) -> float:
    """Two-sided two-sample KS p-value (asymptotic with Stephens' small-sample correction); ref_sorted is sorted."""
    n, m = len(ref_sorted), len(x)
    xs = np.sort(x)
    both = np.concatenate([ref_sorted, xs])
    d = np.abs(np.searchsorted(ref_sorted, both, "right") / n - np.searchsorted(xs, both, "right") / m).max()
    en = np.sqrt(n * m / (n + m))
    return float(kolmogorov((en + 0.12 + 0.11 / en) * d))


def ks_pvalues(ref: pd.DataFrame, cur: pd.DataFrame, features: Sequence[str] = FEATURES) -> np.ndarray:
    """Two-sample KS p-value per feature (fast asymptotic form; scipy's ks_2samp costs ~2 ms per call)."""
    return np.array([_ks_p(np.sort(ref[f].to_numpy()), cur[f].to_numpy()) for f in features])


def holm_reject(p: np.ndarray, alpha: float) -> np.ndarray:
    """Holm step-down rejections (family-wise error <= alpha). p has shape (..., m); returns a bool array."""
    p = np.atleast_2d(p)
    m = p.shape[-1]
    order = np.argsort(p, axis=-1)
    ok = np.take_along_axis(p, order, -1) <= alpha / (m - np.arange(m))
    ok = np.cumprod(ok, axis=-1).astype(bool)  # stop at the first non-rejection
    out = np.zeros_like(ok)
    np.put_along_axis(out, order, ok, -1)
    return out


def flag_features(p: np.ndarray, alpha: float, correction: str = "holm") -> np.ndarray:
    """Per-feature drift flags from p-values, shape (..., m)."""
    p = np.asarray(p, dtype=float)
    if correction == "holm":
        return holm_reject(p, alpha).reshape(p.shape)
    if correction == "none":
        return p < alpha
    raise ValueError(f"unknown correction {correction!r}")


def alert_from_flags(flags: np.ndarray, min_flagged: int) -> np.ndarray:
    """Alert rule: at least `min_flagged` features flagged."""
    return np.asarray(flags).sum(axis=-1) >= min_flagged


def sliding_pvalues(df: pd.DataFrame, cfg: DriftConfig, features: Sequence[str] = FEATURES) -> pd.DataFrame:
    """p-values per feature for each sliding window; index = last event of the window, reference = first ref_size."""
    ref = reference_window(df, cfg.ref_size)
    ref_sorted = [np.sort(ref[f].to_numpy()) for f in features]
    cols = [df[f].to_numpy() for f in features]
    ends = window_ends(len(df), cfg)
    rows = [[_ks_p(r, c[t - cfg.window + 1:t + 1]) for r, c in zip(ref_sorted, cols)] for t in ends]
    return pd.DataFrame(rows, index=pd.Index(list(ends), name="window_end"), columns=list(features))


def alert_series(pvals: pd.DataFrame, cfg: DriftConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """(per-feature flags, alert) for each window of a sliding_pvalues frame."""
    flags = pd.DataFrame(flag_features(pvals.to_numpy(), cfg.alpha, cfg.correction), index=pvals.index,
                         columns=pvals.columns)
    return flags, pd.Series(alert_from_flags(flags.to_numpy(), cfg.min_flagged), index=pvals.index, name="alert")


def episodes(alert: pd.Series) -> int:
    """Number of alert episodes (contiguous runs of alerting windows)."""
    a = alert.to_numpy().astype(int)
    return int(((np.diff(a, prepend=0)) == 1).sum())


def detection_delay(alert: pd.Series, drift_start: int) -> Optional[int]:
    """Events from drift start to the first alert at or after it; None if never, or if an alert was already on at
    drift_start - 1 (a false alarm would otherwise be credited as an instant detection)."""
    before = alert[alert.index < drift_start]
    if len(before) and bool(before.iloc[-1]):
        return None
    after = alert[alert.index >= drift_start]
    hit = after[after]
    return int(hit.index[0] - drift_start) if len(hit) else None


# ---- streams -------------------------------------------------------------------------------------------------

def build_pool(backend, image_dir: str, kinds: Sequence[str] = KINDS) -> pd.DataFrame:
    """Features of every val image under every (perturbation, severity), through the same code as the simulator.

    `backend` is a src.simulator Backend. The noise realisation is fixed per image, so stream seeds vary which images
    are drawn, not the noise draw.
    """
    from datetime import datetime, timezone
    from PIL import Image

    from src.inspection_log import event_from_result, image_stats
    from src.perturb import apply_perturbation, image_seed
    from src.simulator import list_images

    conds = [("none", 0)] + [(k, s) for k in kinds for s in range(1, len(LEVELS[k]) + 1)]
    rows = []
    for path in list_images(image_dir):
        with Image.open(path) as im:
            base = np.asarray(im.convert("RGB"))
        for kind, sev in conds:
            arr = apply_perturbation(base, kind, sev, seed=image_seed(path.name, 0)) if sev else base
            result, version = backend.infer(Image.fromarray(arr), path.name)
            e = event_from_result(result, event_id="", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                                  model_version=version, latency_ms=0.0, perturbation=kind, severity=sev,
                                  simulated=True, stats=image_stats(arr))
            rows.append({"image_id": path.name, "perturbation": kind, "severity": sev,
                         **{f: getattr(e, f) for f in FEATURES}})
    return pd.DataFrame(rows)


def make_stream(pool: pd.DataFrame, n_events: int, seed: int,
                drift: Optional[Tuple[str, int, int]] = None) -> pd.DataFrame:
    """A SIMULATED stream: images drawn with replacement; from event `start` every image gets (kind, severity).

    `drift` = (kind, severity, start). Abrupt (not ramped), so delay is attributable to one severity.
    """
    rng = np.random.default_rng(seed)
    by_cond = {k: g.sort_values("image_id").reset_index(drop=True)
               for k, g in pool.groupby(["perturbation", "severity"])}
    n_img = len(by_cond[("none", 0)])
    idx = rng.integers(n_img, size=n_events)
    parts = []
    start = n_events if drift is None else drift[2]
    parts.append(by_cond[("none", 0)].iloc[idx[:start]])
    if drift is not None:
        parts.append(by_cond[(drift[0], drift[1])].iloc[idx[start:]])
    out = pd.concat(parts, ignore_index=True)
    out["event_index"] = np.arange(len(out))
    return out


# ---- evaluation ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalSetup:
    """Stream shapes shared by calibration and evaluation (only the seed and replicate counts differ)."""

    drift_start: int = 400  # reference = events 0..199, then 200 more clean events before the drift
    horizon: int = 300  # events observed after the drift starts
    clean_events: int = 1000
    n_clean: int = 6
    n_drift_reps: int = 10


def _clean_stream_stats(pool, seed, cfg: DriftConfig, setup: EvalSetup) -> Tuple[List[pd.DataFrame], List[pd.Series]]:
    frames, alerts = [], []
    for r in range(setup.n_clean):
        df = make_stream(pool, setup.clean_events, seed=[seed, 0, r])
        p = sliding_pvalues(df, cfg)
        frames.append(p)
        alerts.append(alert_series(p, cfg)[1])
    return frames, alerts


def conditions(kinds: Sequence[str] = KINDS) -> List[Tuple[str, int]]:
    return [(k, s) for k in kinds for s in range(1, len(LEVELS[k]) + 1)]


def _pvals_cache(pool, seed, windows: Sequence[int], setup: EvalSetup, ref_size: int = 200, kinds=KINDS) -> dict:
    """p-values for every stream, per window size: cache[w] = {'clean': [...], (kind, sev): [(pvals), ...]}."""
    cache = {}
    for w in windows:
        cfg = DriftConfig(window=w, ref_size=ref_size)
        d = {"clean": [sliding_pvalues(make_stream(pool, setup.clean_events, seed=[seed, 0, r]), cfg)
                       for r in range(setup.n_clean)]}
        n = setup.drift_start + setup.horizon
        for kind, sev in conditions(kinds):
            d[(kind, sev)] = [sliding_pvalues(make_stream(pool, n, seed=[seed, 1, r], drift=(kind, sev, setup.drift_start)),
                                              cfg) for r in range(setup.n_drift_reps)]
        cache[w] = d
    return cache


def score_config(cache_w: dict, cfg: DriftConfig, setup: EvalSetup) -> dict:
    """False-alarm and delay numbers for one config from cached p-values (cache_w = cache[cfg.window])."""
    n_eval = sum(len(p) for p in cache_w["clean"])
    alerts = [alert_series(p, cfg)[1] for p in cache_w["clean"]]
    fa = {"fa_episodes_per_1000": 1000.0 * sum(episodes(a) for a in alerts) / n_eval,
          "fa_window_rate": float(sum(a.sum() for a in alerts) / n_eval),
          "fa_streams_with_alert": float(np.mean([a.any() for a in alerts]))}
    delays = {}
    for key, plist in cache_w.items():
        if key == "clean":
            continue
        ds = [detection_delay(alert_series(p, cfg)[1], setup.drift_start) for p in plist]
        delays[key] = ds
    return {**fa, "delays": delays}


def censored_mean(delays: Sequence[Optional[int]], horizon: int) -> float:
    """Mean delay with undetected / contaminated streams counted as the full horizon."""
    return float(np.mean([horizon if d is None else d for d in delays]))


def select_config(calib_cache: dict, setup: EvalSetup, windows, alphas, ks, max_fa_per_1000: float,
                  correction: str = "holm") -> Tuple[DriftConfig, pd.DataFrame]:
    """Rule fixed before looking at the evaluation seed: among configs with calibration false alarms
    <= max_fa_per_1000 episodes per 1000 events, pick the lowest mean delay over the harmful perturbations
    (blur, noise, contrast; undetected = horizon). Ties go to fewer false alarms, then the smaller window."""
    rows = []
    for w, a, k in product(windows, alphas, ks):
        cfg = DriftConfig(window=w, alpha=a, min_flagged=k, correction=correction)
        s = score_config(calib_cache[w], cfg, setup)
        harm = [d for key, ds in s["delays"].items() if key[0] in HARMFUL_KINDS for d in ds]
        rows.append({"window": w, "alpha": a, "min_flagged": k, "fa_episodes_per_1000": s["fa_episodes_per_1000"],
                     "fa_window_rate": s["fa_window_rate"], "mean_delay_harmful": censored_mean(harm, setup.horizon),
                     "detected_harmful": float(np.mean([d is not None for d in harm]))})
    grid = pd.DataFrame(rows)
    grid["feasible"] = grid.fa_episodes_per_1000 <= max_fa_per_1000
    pick = grid[grid.feasible].sort_values(["mean_delay_harmful", "fa_episodes_per_1000", "window"])
    if pick.empty:
        raise RuntimeError("no config meets the false-alarm budget on the calibration stream")
    best = pick.iloc[0]
    return DriftConfig(window=int(best.window), alpha=float(best.alpha), min_flagged=int(best.min_flagged),
                       correction=correction), grid


def clean_multiple_testing(pvals_clean: Sequence[pd.DataFrame], cfg: DriftConfig) -> dict:
    """On clean streams: per-feature flag rate at raw alpha, P(any feature flagged) raw vs Holm, and the alert rate."""
    p = np.concatenate([x.to_numpy() for x in pvals_clean])
    raw, holm = p < cfg.alpha, holm_reject(p, cfg.alpha)
    return {"n_windows": int(len(p)), "alpha": cfg.alpha, "n_features": p.shape[1],
            "per_feature_raw_flag_rate": dict(zip(FEATURES, map(float, raw.mean(0)))),
            "any_feature_flagged_raw": float(raw.any(1).mean()),
            "any_feature_flagged_holm": float(holm.any(1).mean()),
            "alert_rate_raw_rule": float(alert_from_flags(raw, cfg.min_flagged).mean()),
            "alert_rate_holm_rule": float(alert_from_flags(holm, cfg.min_flagged).mean())}


def evaluate_fresh(pool: pd.DataFrame, cfg: DriftConfig, setup: EvalSetup, seed: int) -> dict:
    """Score a fixed config on a fresh seed: false alarms, delay and firing features per (kind, severity)."""
    cache = _pvals_cache(pool, seed, [cfg.window], setup, cfg.ref_size)[cfg.window]
    s = score_config(cache, cfg, setup)
    full = setup.drift_start + cfg.window - 1  # first window made only of drifted events
    clean_ref = pool[(pool.perturbation == "none")]
    rows = []
    for (kind, sev), ds in s["delays"].items():
        det = [d for d in ds if d is not None]
        flags = np.array([flag_features(p.loc[full].to_numpy(), cfg.alpha, cfg.correction) for p in cache[(kind, sev)]])
        cond = pool[(pool.perturbation == kind) & (pool.severity == sev)]
        rows.append({"perturbation": kind, "severity": sev, "level": LEVELS[kind][sev - 1], "n_streams": len(ds),
                     "detected": len(det) / len(ds), "delay_median": float(np.median(det)) if det else None,
                     "delay_p90": float(np.percentile(det, 90)) if det else None,
                     "delay_mean_censored": censored_mean(ds, setup.horizon),
                     **{f"flag_{f}": float(v) for f, v in zip(FEATURES, flags.mean(0))},
                     "mean_conf_shift": float(cond.max_confidence.mean() - clean_ref.max_confidence.mean()),
                     "mean_ndet_shift": float(cond.n_detections.mean() - clean_ref.n_detections.mean())})
    return {"config": asdict(cfg), "fa_episodes_per_1000": s["fa_episodes_per_1000"],
            "fa_window_rate": s["fa_window_rate"], "fa_streams_with_alert": s["fa_streams_with_alert"],
            "multiple_testing": clean_multiple_testing(cache["clean"], cfg), "table": rows}


def add_model_impact(table: pd.DataFrame, robustness_csv: str = "reports/phase4/robustness_deltas.csv") -> pd.DataFrame:
    """Attach the Phase 4 mAP50 change (with its CI) per (perturbation, severity), when that file exists."""
    p = Path(robustness_csv)
    if not p.exists():
        return table
    r = pd.read_csv(p)
    r = r[r.metric == "mAP50"][["perturbation", "severity", "delta", "delta_lo", "delta_hi"]]
    return table.merge(r.rename(columns={"delta": "map50_delta", "delta_lo": "map50_delta_lo",
                                         "delta_hi": "map50_delta_hi"}), on=["perturbation", "severity"], how="left")


# ---- Evidently -----------------------------------------------------------------------------------------------

def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, html_path: str, json_path: str, cfg: DriftConfig) -> dict:
    """Evidently data-drift report (KS per column at cfg.alpha) saved as HTML and JSON; returns per-column results."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=FEATURES)
    ref_ds, cur_ds = Dataset.from_pandas(ref[FEATURES], data_definition=definition), \
        Dataset.from_pandas(cur[FEATURES], data_definition=definition)
    snap = Report([DataDriftPreset(columns=FEATURES, method="ks", threshold=cfg.alpha)]).run(cur_ds, ref_ds)
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    snap.save_html(html_path)
    snap.save_json(json_path)
    return snap.dict()


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def cmd_pool(a) -> None:
    from src.simulator import ServiceBackend
    from app.service import InspectionService, Settings

    service = InspectionService(Settings.from_env())
    if not service.loaded:
        raise SystemExit(f"no weights at {service.settings.model_path}")
    pool = build_pool(ServiceBackend(service), a.image_dir)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(a.out)
    print(f"wrote {len(pool)} rows ({pool.image_id.nunique()} val images) to {a.out}")


def cmd_evaluate(a) -> None:
    pool = pd.read_parquet(a.pool)
    setup = EvalSetup()
    windows, alphas, ks = (30, 50, 100), (0.05, 0.01, 0.001), (1, 2, 3)
    calib = _pvals_cache(pool, CALIB_SEED, windows, setup)
    cfg, grid = select_config(calib, setup, windows, alphas, ks, max_fa_per_1000=a.max_fa)
    print(f"calibration seed {CALIB_SEED}: chose {cfg}")
    eval_setup = EvalSetup(n_clean=20, n_drift_reps=30)  # more replicates for the single scored run
    ev = evaluate_fresh(pool, cfg, eval_setup, EVAL_SEED)
    table = add_model_impact(pd.DataFrame(ev.pop("table")))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "calibration_grid.csv", index=False)
    table.to_csv(out / "eval_delay_by_condition.csv", index=False)
    summary = {"simulated": True, "calibration_seed": CALIB_SEED, "eval_seed": EVAL_SEED, "calibration_setup": asdict(setup), "eval_setup": asdict(eval_setup),
               "max_fa_per_1000": a.max_fa, "chosen": asdict(cfg), "evaluation": ev,
               "delay_by_condition": table.to_dict("records")}
    _write_json(summary, out / "drift_summary.json")
    print(table.round(3).to_string())
    print({k: ev[k] for k in ("fa_episodes_per_1000", "fa_window_rate", "fa_streams_with_alert")})
    demo_reports(pool, cfg, eval_setup, out)


def demo_reports(pool, cfg: DriftConfig, setup: EvalSetup, out: Path) -> None:
    """Evidently HTML+JSON for one fresh-seed stream per demo drift (blur hurts the model, brightness does not)."""
    checks = {}
    for kind, sev in (("blur", 2), ("brightness", 3)):
        df = make_stream(pool, setup.drift_start + cfg.window, seed=[EVAL_SEED, 2, 0], drift=(kind, sev, setup.drift_start))
        ref, cur = reference_window(df, cfg.ref_size), df.iloc[-cfg.window:]
        ev = evidently_report(ref, cur, str(out / f"evidently_{kind}.html"), str(out / f"evidently_{kind}.json"), cfg)
        theirs = evidently_pvalues(ev)
        mine = dict(zip(FEATURES, ks_pvalues(ref, cur)))
        checks[kind] = {"own_p": mine, "evidently_p": theirs,
                        "flags_agree_at_raw_alpha": all((mine[f] < cfg.alpha) == (theirs[f] < cfg.alpha) for f in FEATURES)}
    _write_json(checks, out / "evidently_crosscheck.json")


def evidently_pvalues(report: dict) -> Dict[str, float]:
    """{column: KS p-value} from an Evidently 0.7 report dict (ValueDrift metrics; drift means p < threshold)."""
    return {m["config"]["column"]: float(m["value"]) for m in report["metrics"]
            if m["config"]["type"].endswith("ValueDrift")}


def cmd_report(a) -> None:
    from src.inspection_log import normalize

    df = normalize(pd.read_parquet(a.log) if not Path(a.log).is_dir() else pd.concat(
        [pd.read_parquet(p) for p in sorted(Path(a.log).glob("*.parquet"))])).sort_values("timestamp").reset_index(drop=True)
    cfg = DriftConfig(window=a.window, alpha=a.alpha, min_flagged=a.min_flagged)
    ref, cur = reference_window(df, cfg.ref_size), df.iloc[-cfg.window:]
    out = Path(a.out_dir)
    evidently_report(ref, cur, str(out / "evidently_log.html"), str(out / "evidently_log.json"), cfg)
    flags = flag_features(ks_pvalues(ref, cur), cfg.alpha, cfg.correction)
    _write_json({"flags": dict(zip(FEATURES, map(bool, flags))), "alert": bool(alert_from_flags(flags, cfg.min_flagged)),
                 "config": asdict(cfg)}, out / "drift_log_summary.json")
    print(f"alert={bool(alert_from_flags(flags, cfg.min_flagged))} flagged={[f for f, v in zip(FEATURES, flags) if v]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--image-dir", default="data/yolo/images/val")
    p.add_argument("--out", default=str(POOL_PATH))
    p.set_defaults(fn=cmd_pool)
    e = sub.add_parser("evaluate")
    e.add_argument("--pool", default=str(POOL_PATH))
    e.add_argument("--out-dir", default=str(OUT_DIR))
    e.add_argument("--max-fa", type=float, default=1.0, help="calibration false-alarm episodes per 1000 events")
    e.set_defaults(fn=cmd_evaluate)
    r = sub.add_parser("report")
    r.add_argument("--log", required=True)
    r.add_argument("--out-dir", default=str(OUT_DIR))
    r.add_argument("--window", type=int, default=50)
    r.add_argument("--alpha", type=float, default=0.01)
    r.add_argument("--min-flagged", type=int, default=2)
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
