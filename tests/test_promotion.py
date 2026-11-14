"""Promotion-gate tests: pure rules, the missing-CI case, and the dry-run default."""

import sys

import numpy as np
import pandas as pd
import pytest

from src import promotion
from src.promotion import Rules, apply_promotion, evaluate_gate, load_rules, metrics_from_comparison, paired_bootstrap_ci

RULES = Rules(min_map50_gain=0.01, max_class_ap50_drop=0.05)
CUR = {"mAP50": 0.75, "per_class": {"a": 0.8, "b": 0.7}}


def cand(m, a=0.8, b=0.7):
    return {"mAP50": m, "per_class": {"a": a, "b": b}}


def test_promote_when_gain_and_no_class_drop():
    r = evaluate_gate(CUR, cand(0.77, 0.83, 0.71), RULES, per_image=None)
    assert r.decision == "promote" and r.checks["map50_gain"]["passed"] and r.checks["class_drop"]["passed"]


def test_reject_when_gain_too_small():
    r = evaluate_gate(CUR, cand(0.755), RULES)
    assert r.decision == "reject" and "below the required" in r.reasons[0]


def test_reject_on_a_single_class_drop_even_with_a_big_gain():
    r = evaluate_gate(CUR, cand(0.80, a=0.70, b=0.95), RULES)
    assert r.decision == "reject" and r.checks["map50_gain"]["passed"]
    assert list(r.checks["class_drop"]["classes"]) == ["a"] and "a (" in r.reasons[0]


def test_drop_equal_to_the_limit_is_allowed():
    assert evaluate_gate(CUR, cand(0.77, a=0.75), RULES).decision == "promote"


def test_missing_ci_is_recorded_as_not_applied_by_default():
    r = evaluate_gate(CUR, cand(0.77), RULES, per_image=None)
    assert r.checks["paired_ci"] == {"passed": None, "applied": False,
                                     "note": "not applied: no per-image predictions available"}
    assert r.decision == "promote"


def test_missing_ci_can_be_made_a_rejection():
    r = evaluate_gate(CUR, cand(0.77), Rules(0.01, 0.05, ci_when_missing="reject"))
    assert r.decision == "reject" and r.checks["paired_ci"]["passed"] is False


def test_paired_ci_must_exclude_zero():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.3, 0.9, 200)
    better = evaluate_gate(CUR, cand(0.77), RULES, per_image=(base, base + 0.05 + rng.normal(0, 0.02, 200)))
    assert better.decision == "promote" and better.checks["paired_ci"]["ci"][0] > 0
    noisy = evaluate_gate(CUR, cand(0.77), RULES, per_image=(base, base + rng.normal(0, 0.2, 200)))
    assert noisy.decision == "reject" and "does not exclude zero" in noisy.reasons[0]


def test_paired_bootstrap_is_deterministic_and_validates():
    a, b = np.arange(10.0), np.arange(10.0) + 1
    assert paired_bootstrap_ci(a, b, 200, seed=1) == paired_bootstrap_ci(a, b, 200, seed=1) == (1.0, 1.0)
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, b[:5])


def test_config_values_load():
    r = load_rules()
    assert (r.min_map50_gain, r.max_class_ap50_drop, r.ci_when_missing) == (0.01, 0.05, "skip")


def test_metrics_from_comparison_and_stored_demo():
    df = pd.read_csv(promotion.COMPARISON)
    m = metrics_from_comparison(df, "e0_baseline")
    assert set(m["per_class"]) >= {"crazing", "scratches"} and 0 < m["mAP50"] < 1
    for e in ("e1_imgsz320", "e2_yolov8m_640", "e3_augmented"):
        assert evaluate_gate(m, metrics_from_comparison(df, e), load_rules()).decision == "reject"
    with pytest.raises(KeyError):
        metrics_from_comparison(df, "nope")


class FakeClient:
    def __init__(self):
        self.calls = []

    def set_registered_model_alias(self, *a):
        self.calls.append(a)


def test_dry_run_is_the_default_and_apply_only_moves_on_promote():
    ok = evaluate_gate(CUR, cand(0.8), RULES)
    c = FakeClient()
    assert apply_promotion(ok, "m", "production", "3", client=c).startswith("DRY RUN") and c.calls == []
    assert "now points" in apply_promotion(ok, "m", "production", "3", apply=True, client=c)
    assert c.calls == [("m", "production", "3")]
    bad = evaluate_gate(CUR, cand(0.7), RULES)
    c2 = FakeClient()
    assert apply_promotion(bad, "m", "production", "3", apply=True, client=c2) == "no change: gate rejected"
    assert c2.calls == []


def test_cli_has_no_apply_by_default(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["promotion", "--out", str(tmp_path / "o.json")])
    called = []
    monkeypatch.setattr(promotion, "apply_promotion", lambda *a, **k: called.append(k.get("apply")) or "x")
    promotion.main()
    assert called == [False, False, False] and "REJECT" in capsys.readouterr().out
