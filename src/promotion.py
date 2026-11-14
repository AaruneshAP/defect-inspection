"""Model-promotion gate (Phase 7b): compare a candidate with the current model on VAL metrics; promote or reject.

The rules live in configs/promotion.yaml. `evaluate_gate` is a pure function. Dry-run is the default; --apply moves the
MLflow alias only when the gate says promote.

  python -m src.promotion                                    # E1, E2, E3 vs E0 from reports/phase2b/comparison.csv
  python -m src.promotion --candidate e1_imgsz320 --apply --version 3
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

CONFIG = Path("configs/promotion.yaml")
COMPARISON = Path("reports/phase2b/comparison.csv")
CLASS_SUFFIX = "_AP50"


@dataclass(frozen=True)
class Rules:
    min_map50_gain: float
    max_class_ap50_drop: float
    ci_required: bool = True
    n_boot: int = 2000
    level: float = 0.95
    seed: int = 0
    ci_when_missing: str = "skip"


@dataclass
class GateResult:
    decision: str  # 'promote' or 'reject'
    reasons: List[str] = field(default_factory=list)
    checks: Dict[str, dict] = field(default_factory=dict)


def load_rules(path: Path = CONFIG) -> Rules:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["rules"]
    ci = cfg["ci"]
    if ci["when_missing"] not in ("skip", "reject"):
        raise ValueError("ci.when_missing must be 'skip' or 'reject'")
    return Rules(cfg["min_map50_gain"], cfg["max_class_ap50_drop"], ci["required_when_available"], ci["n_boot"],
                 ci["level"], ci["seed"], ci["when_missing"])


def paired_bootstrap_ci(cur: np.ndarray, cand: np.ndarray, n_boot: int = 2000, level: float = 0.95,
                        seed: int = 0) -> tuple:
    """CI of mean(cand - cur) over images, resampling images with replacement (same indices for both models)."""
    d = np.asarray(cand, float) - np.asarray(cur, float)
    if d.ndim != 1 or len(d) == 0 or len(d) != len(cur):
        raise ValueError("per-image scores must be equal-length 1-D arrays")
    idx = np.random.default_rng(seed).integers(len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def evaluate_gate(current: dict, candidate: dict, rules: Rules, per_image: Optional[tuple] = None) -> GateResult:
    """current/candidate: {'mAP50': float, 'per_class': {name: AP50}}. per_image: optional (cur_scores, cand_scores)."""
    res = GateResult("promote")

    def fail(name, msg, **info):
        res.checks[name] = {"passed": False, **info}
        res.reasons.append(msg)

    gain = candidate["mAP50"] - current["mAP50"]
    if gain >= rules.min_map50_gain - 1e-9:
        res.checks["map50_gain"] = {"passed": True, "value": gain, "min": rules.min_map50_gain}
    else:
        fail("map50_gain", f"mAP50 gain {gain:+.4f} is below the required {rules.min_map50_gain:+.4f}",
             value=gain, min=rules.min_map50_gain)

    drops = {c: current["per_class"][c] - candidate["per_class"][c] for c in current["per_class"]}
    bad = {c: d for c, d in drops.items() if d > rules.max_class_ap50_drop + 1e-9}
    if bad:
        fail("class_drop", "AP50 drop above %.3f in: %s" % (
            rules.max_class_ap50_drop, ", ".join(f"{c} ({d:.4f})" for c, d in sorted(bad.items()))),
            worst=max(drops.values()), classes=bad, max=rules.max_class_ap50_drop)
    else:
        res.checks["class_drop"] = {"passed": True, "worst": max(drops.values()), "max": rules.max_class_ap50_drop}

    if per_image is not None and rules.ci_required:
        lo, hi = paired_bootstrap_ci(*per_image, n_boot=rules.n_boot, level=rules.level, seed=rules.seed)
        if lo > 0:
            res.checks["paired_ci"] = {"passed": True, "ci": [lo, hi]}
        else:
            fail("paired_ci", f"paired bootstrap CI [{lo:+.4f}, {hi:+.4f}] does not exclude zero", ci=[lo, hi])
    elif per_image is None and rules.ci_required:
        if rules.ci_when_missing == "reject":
            fail("paired_ci", "no per-image predictions, so the paired CI cannot be computed (rule: reject)")
        else:
            res.checks["paired_ci"] = {"passed": None, "applied": False,
                                       "note": "not applied: no per-image predictions available"}

    if res.reasons:
        res.decision = "reject"
    else:
        res.reasons.append("all applied rules passed")
    return res


def metrics_from_comparison(df: pd.DataFrame, experiment: str) -> dict:
    """{'mAP50', 'per_class'} for one experiment row of reports/phase2b/comparison.csv."""
    row = df[df.experiment == experiment]
    if row.empty:
        raise KeyError(f"{experiment!r} not in {sorted(df.experiment)}")
    row = row.iloc[0]
    return {"mAP50": float(row["mAP50"]),
            "per_class": {c[:-len(CLASS_SUFFIX)]: float(row[c]) for c in df.columns if c.endswith(CLASS_SUFFIX)}}


def apply_promotion(result: GateResult, model_name: str, alias: str, version: str, apply: bool = False,
                    client=None) -> str:
    """Move the MLflow alias to `version` only if apply=True and the gate said promote; otherwise just describe it."""
    if result.decision != "promote":
        return "no change: gate rejected"
    if not apply:
        return f"DRY RUN: would set alias {alias!r} of {model_name!r} to version {version}"
    if client is None:
        from src.tracking import setup

        client = setup()
    client.set_registered_model_alias(model_name, alias, str(version))
    return f"alias {alias!r} of {model_name!r} now points to version {version}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--metrics", default=str(COMPARISON))
    ap.add_argument("--current", default="e0_baseline")
    ap.add_argument("--candidate", action="append", help="repeatable; default: every other experiment")
    ap.add_argument("--version", help="registered-model version of the candidate (needed for --apply)")
    ap.add_argument("--apply", action="store_true", help="really move the MLflow alias (default: dry run)")
    ap.add_argument("--out", default="reports/phase7/promotion_demo.json")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    rules = load_rules(Path(a.config))
    df = pd.read_csv(a.metrics)
    cur = metrics_from_comparison(df, a.current)
    cands = a.candidate or [e for e in df.experiment if e != a.current]
    if a.apply and (len(cands) != 1 or not a.version):
        ap.error("--apply needs exactly one --candidate and a --version")
    out = {"rules": asdict(rules), "current": a.current, "data": "val metrics only; no per-image predictions", "results": {}}
    for c in cands:
        r = evaluate_gate(cur, metrics_from_comparison(df, c), rules)
        action = apply_promotion(r, cfg["model_name"], cfg["alias"], a.version or "?", apply=a.apply)
        print(f"{c} vs {a.current}: {r.decision.upper()}")
        for reason in r.reasons:
            print(f"  - {reason}")
        if "paired_ci" in r.checks and r.checks["paired_ci"].get("applied") is False:
            print("  - paired CI: not applied (no per-image predictions)")
        print(f"  -> {action}")
        out["results"][c] = {**asdict(r), "action": action}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
