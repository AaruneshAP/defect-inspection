"""Retroactively log stored Phase 2/3 results to the local MLflow store (idempotent)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from src import tracking as T

E_DIR = Path("reports/phase2b/raw/phase2b_outputs")
EXPERIMENTS = {
    "e0_baseline": "configs/baseline.yaml",
    "e1_imgsz320": "configs/e1_imgsz320.yaml",
    "e2_yolov8m_640": "configs/e2_yolov8m_640.yaml",
    "e3_augmented": "configs/e3_augmented.yaml",
}


def _results_commit() -> str | None:
    """Commit that recorded the Phase 2b comparison (raw outputs are git-ignored)."""
    return T.git_commit("reports/phase2b/comparison.csv")


def log_phase2(uri: str) -> dict[str, str]:
    """Log E0-E3 from their stored results.json plus the YAML config; returns {name: run_id}."""
    ids = {}
    for name, cfg_path in EXPERIMENTS.items():
        res = json.loads((E_DIR / name / "results.json").read_text())
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        per_class = res.pop("per_class_ap")
        for k in ("best_checkpoint", "last_checkpoint", "smoke_test"):
            res.pop(k, None)
        metrics = {k: v for k, v in res.items() if k in ("mAP50", "mAP50-95", "precision", "recall")}
        metrics.update({f"{c}.{k}": v for c, d in per_class.items() for k, v in d.items()})
        arts = [E_DIR / name / f for f in ("results.json", "results.csv", "BoxPR_curve.png")]
        ids[name] = T.log_run(
            name,
            params={**cfg, "seed": res["seed"]},
            metrics=metrics,
            artifacts=[a for a in arts if a.exists()] + [Path(cfg_path)],
            tags={"phase": "2b", "split": "val", "git_commit": _results_commit() or "unknown",
                  "source": "retroactive", "config_file": cfg_path},
            tracking_uri=uri,
        )
    return ids


def log_phase3(uri: str) -> str:
    """Log the Phase 3 frozen policy and test-split numbers as one run."""
    frozen = json.loads(Path("reports/phase3/frozen.json").read_text())
    cost = yaml.safe_load(Path("configs/cost.yaml").read_text())
    test = pd.read_csv("reports/phase3/test_results.csv")
    metrics = {}
    for _, r in test.iterrows():
        key = re.sub(r"[^a-z0-9.]+", "_", r["policy"].split(" (a=")[0].lower()).strip("_")
        for col in ("cost_per_image", "recall", "alerts_per_image", "reviews_per_image",
                    "pct_vs_flat025", "pct_vs_matched", "pct_vs_none"):
            metrics[f"{key}.{col}"] = r[col]
    sav = pd.read_csv("reports/phase3/test_savings.csv").iloc[0]
    metrics.update({"primary.saving_pct_vs_matched": sav["saving_pct"],
                    "primary.saving_pct_vs_matched_lo": sav["pct_lo"],
                    "primary.saving_pct_vs_matched_hi": sav["pct_hi"]})
    return T.log_run(
        "phase3_test",
        params={"frozen": frozen, "cost": cost, "model": "e0_baseline", "split": "test", "model_seed": 42,
                "bootstrap_seed": "0 (CLI default; invocation not recorded)"},
        metrics=metrics,
        artifacts=[Path(p) for p in ("reports/phase3/frozen.json", "reports/phase3/test_results.csv",
                                     "reports/phase3/test_savings.csv", "reports/phase3/preregistration.md",
                                     "configs/cost.yaml")],
        tags={"phase": "3", "split": "test", "source": "retroactive",
              "git_commit": T.git_commit("reports/phase3/test_results.csv") or "unknown"},
        tracking_uri=uri,
    )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uri", default=T.DEFAULT_TRACKING_URI)
    ap.add_argument("--weights", default="models/best.pt")
    a = ap.parse_args()
    ids = log_phase2(a.uri)
    p3 = log_phase3(a.uri)
    client = T.setup(a.uri)
    try:
        mv = client.get_model_version_by_alias("defect-detector", "production")
        version = mv.version if mv.run_id == ids["e0_baseline"] else None
    except Exception:
        version = None
    if version is None:
        version = T.register_model(ids["e0_baseline"], a.weights, "defect-detector", "production", a.uri)
    print("runs:", {**ids, "phase3_test": p3})
    print(f"registered defect-detector v{version} (alias: production)")


if __name__ == "__main__":
    main()
