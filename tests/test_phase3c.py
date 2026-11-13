"""Tests for src/phase3c.py (matched-workload baseline, sensitivity, freeze, test guard)."""

import json

import numpy as np
import pandas as pd
import pytest

from src import decision as d
from src import evaluate as E
from src import phase3c as P

NAMES = ["a", "b"]
CFG = {
    "cost_miss": 10, "cost_false_alarm": 1, "cost_review": 0.3,
    "class_weights": {"a": 1.0, "b": 1.0},
    "area_severity": {"area_frac_small": 0.02, "area_frac_large": 0.20, "factor_small": 1.0, "factor_large": 1.0},
    "action_bounds": {"ignore_below": 0.5, "monitor_below": 0.8, "service_below": 1.3},
    "review_band_width": 0.15,
}
BOX, FAR = (0, 0, 10, 10), (50, 50, 60, 60)


def row(kind, image, cls, conf, box):
    return dict(kind=kind, image=image, class_id=cls, conf=conf, x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                img_w=100, img_h=100)


def synthetic(n_img=40, seed=0):
    """Random detections: every image has a GT of class (i%2) found at a random confidence, plus false detections."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_img):
        im, c = f"im{i}.jpg", i % 2
        rows.append(row("gt", im, c, np.nan, BOX))
        if rng.random() < 0.85:
            rows.append(row("pred", im, c, float(rng.uniform(0.05, 0.95)), BOX))
        for _ in range(rng.integers(0, 4)):
            rows.append(row("pred", im, int(rng.integers(0, 2)), float(rng.uniform(0.01, 0.6)), FAR))
    df = pd.DataFrame(rows)
    return d.match_detections(df[df.kind == "pred"].reset_index(drop=True), df[df.kind == "gt"].reset_index(drop=True))


# ---- flat baseline ---------------------------------------------------------------------------

def test_flat_curve_hand_computed():
    # GT found at 0.9; false detections at 0.4 and 0.2. cost(t) = 1*#FA>=t + 10*[0.9<t]
    rows = [row("gt", "i", 0, np.nan, BOX), row("pred", "i", 0, 0.9, BOX), row("pred", "i", 0, 0.4, FAR),
            row("pred", "i", 0, 0.2, FAR)]
    df = pd.DataFrame(rows)
    p, g = d.match_detections(df[df.kind == "pred"].reset_index(drop=True), df[df.kind == "gt"].reset_index(drop=True))
    cur = P.flat_curve(p, g, CFG, NAMES, 1, [0.1, 0.3, 0.5, 0.95, P.OFF]).set_index("t")
    assert cur.cost.tolist() == pytest.approx([2, 1, 0, 10, 10])
    assert cur.alerts_per_image.tolist() == pytest.approx([3, 2, 1, 0, 0])


def test_flat_curve_agrees_with_policy_outcomes():
    p, g = synthetic()
    n = 40
    ts = [0.002, 0.1, 0.25, 0.37, 0.6, 0.9, P.OFF]
    cur = P.flat_curve(p, g, CFG, NAMES, n, ts).set_index("t")
    for t in ts:
        o = d.policy_outcomes(p, g, E.flat_bands(t, 2), CFG, NAMES)
        assert cur.cost[t] == pytest.approx(o.cost.sum())
        assert cur.alerts_per_image[t] == pytest.approx(o.alerts.sum() / n)


def test_best_flat_respects_budget_and_is_optimal_among_feasible():
    p, g = synthetic()
    n = 40
    for B in (0.5, 1, 2, 3):
        t = P.best_flat_threshold(p, g, CFG, NAMES, n, B)
        cur = P.flat_curve(p, g, CFG, NAMES, n)
        sel = cur[cur.t == t].iloc[0]
        assert sel.alerts_per_image <= B + 1e-9
        feasible = cur[cur.alerts_per_image <= B + 1e-9]
        assert sel.cost == pytest.approx(feasible.cost.min())


def test_best_flat_budget_zero_means_no_alerts():
    p, g = synthetic()
    assert P.best_flat_threshold(p, g, CFG, NAMES, 40, 0) == P.OFF


def test_loose_budget_gives_unconstrained_optimum():
    p, g = synthetic()
    t = P.best_flat_threshold(p, g, CFG, NAMES, 40, 1e6)
    cur = P.flat_curve(p, g, CFG, NAMES, 40)
    assert cur[cur.t == t].cost.iloc[0] == pytest.approx(cur.cost.min())


# ---- bootstrap -------------------------------------------------------------------------------

def _out(n=30, seed=1):
    rng = np.random.default_rng(seed)
    gt = rng.integers(0, 3, n).astype(float)
    return pd.DataFrame({"alerts": rng.integers(0, 4, n).astype(float), "reviews": rng.integers(0, 3, n).astype(float),
                         "gt": gt, "caught": gt * rng.uniform(0, 1, n), "cost": rng.uniform(0, 10, n)})


def test_bootstrap_metrics_point_estimates_and_ordering():
    o = _out()
    m = P.bootstrap_metrics(o, n_boot=500)
    assert m["cost"][0] == pytest.approx(o.cost.sum())
    assert m["cost_per_image"][0] == pytest.approx(o.cost.mean())
    assert m["recall"][0] == pytest.approx(o.caught.sum() / o["gt"].sum())
    assert m["alerts_per_image"][0] == pytest.approx(o.alerts.mean())
    assert m["reviews_per_image"][0] == pytest.approx(o.reviews.mean())
    for pt, lo, hi in m.values():
        assert lo <= pt <= hi


def test_bootstrap_metrics_constant_has_degenerate_ci():
    o = _out()
    o["cost"] = 2.0
    m = P.bootstrap_metrics(o, n_boot=200)
    assert m["cost_per_image"] == pytest.approx((2.0, 2.0, 2.0))


def test_bootstrap_metrics_deterministic():
    o = _out()
    assert P.bootstrap_metrics(o, 300, 3) == P.bootstrap_metrics(o, 300, 3)


# ---- cross-validation / sensitivity ----------------------------------------------------------

def test_cross_validate_matched_covers_every_image_once_and_respects_keys():
    p, g = synthetic()
    oof, tuned = P.cross_validate_matched(p, g, CFG, NAMES, budgets=[2], accs=[0.9], k=4)
    assert set(oof) == {("none",), ("flat0.25",), ("flat", 2), ("band", 0.9, 2)}
    images = sorted(set(p.image) | set(g.image))
    for o in oof.values():
        assert sorted(o.index) == images
    assert oof[("none",)]["gt"].sum() == 40 and oof[("none",)].cost.sum() > 0
    assert oof[("none",)].alerts.sum() == 0
    assert len(tuned[("flat", 2)]) == 4


def test_band_with_budget_never_beats_unconstrained_in_sample_and_load_is_bounded():
    p, g = synthetic()
    b = E.tune_bands(p, g, CFG, NAMES, 0.9, 40, budget=2)
    o = d.policy_outcomes(p, g, b, CFG, NAMES, 0.9)
    assert (o.alerts + o.reviews).sum() <= 2 * 40


def test_with_costs_does_not_mutate_and_sets_ratio():
    c = P.with_costs(CFG, 20, 0.6)
    assert c["cost_miss"] == 20 and c["cost_review"] == 0.6 and CFG["cost_miss"] == 10 and CFG["cost_review"] == 0.3


def test_sensitivity_shape_and_columns_and_extreme_ratio():
    p, g = synthetic()
    s = P.sensitivity(p, g, CFG, NAMES, ratios=[2, 50], review_costs=[0.1, 1.0], budget=2, k=4, n_boot=200)
    assert len(s) == 4 and set(zip(s.miss_to_fa_ratio, s.review_cost)) == {(2, 0.1), (2, 1.0), (50, 0.1), (50, 1.0)}
    assert (s.saving_lo <= s.saving).all() and (s.saving <= s.saving_hi).all()
    # saving = flat_cost - policy_cost
    assert (s.flat_cost - s.policy_cost).to_numpy() == pytest.approx(s.saving.to_numpy())


def test_sensitivity_review_cost_prohibitive_removes_reviews():
    p, g = synthetic()
    s = P.sensitivity(p, g, CFG, NAMES, ratios=[10], review_costs=[1e6], budget=3, k=4, n_boot=100)
    assert s.policy_reviews_pi.iloc[0] == 0.0


def test_saving_status():
    mk = lambda lo, hi: pd.Series({"saving_lo": lo, "saving_hi": hi})
    assert P.saving_status(mk(1, 5)) == "significant"
    assert P.saving_status(mk(-5, -1)) == "worse"
    assert P.saving_status(mk(-1, 5)) == "none"
    assert P.saving_status(mk(0, 5)) == "none"


def test_matched_table_saving_matches_costs():
    p, g = synthetic()
    oof, _ = P.cross_validate_matched(p, g, CFG, NAMES, budgets=[2], accs=[0.9], k=4)
    t = P.matched_table(oof, [2], [0.9], n_boot=200, seed=0)
    r = t.iloc[0]
    assert r.saving == pytest.approx(r.flat_cost - r.policy_cost)
    assert r.saving_lo <= r.saving <= r.saving_hi


# ---- freeze / test guard ---------------------------------------------------------------------

def test_freeze_policies_structure_and_budget():
    p, g = synthetic()
    f = P.freeze_policies(p, g, CFG, NAMES)
    assert f["primary"]["reviewer_accuracy"] == 0.9 and f["primary"]["budget"] == 3
    assert f["secondary_a1.0"]["reviewer_accuracy"] == 1.0 and f["secondary_a0.8"]["reviewer_accuracy"] == 0.8
    assert f["n_val_images"] == 40 and f["flat_0.25"]["threshold"] == 0.25
    b = P.bands_from_frozen(f["primary"], NAMES)
    o = d.policy_outcomes(p, g, b, CFG, NAMES, 0.9)
    assert (o.alerts + o.reviews).sum() <= 3 * 40
    json.dumps(f)  # serialisable


def test_preregistration_text_contains_exact_config():
    p, g = synthetic()
    f = P.freeze_policies(p, g, CFG, NAMES)
    txt = P.preregistration_text(f, CFG, NAMES, {"x": "abc"}, "2026-01-01 00:00:00", "deadbeef")
    assert "reviewer_accuracy = 0.9" in txt and "B <= 3" in txt and "abc" in txt
    assert json.dumps(f, indent=2) in txt


def test_test_command_refuses_without_preregistration(tmp_path):
    import argparse
    a = argparse.Namespace(out=str(tmp_path), test_cache="nope.parquet", config="configs/cost.yaml", n_boot=10, seed=0)
    with pytest.raises(SystemExit) as e:
        P.cmd_test(a)
    assert "pre-registration" in str(e.value)


def test_test_command_refuses_to_rerun(tmp_path):
    import argparse
    (tmp_path / "frozen.json").write_text("{}")
    (tmp_path / "preregistration.md").write_text("x")
    (tmp_path / "test_results.csv").write_text("x")
    a = argparse.Namespace(out=str(tmp_path), test_cache="nope.parquet", config="configs/cost.yaml", n_boot=10, seed=0)
    with pytest.raises(SystemExit) as e:
        P.cmd_test(a)
    assert "once" in str(e.value)


# ---- figures (smoke) -------------------------------------------------------------------------

def test_figures_are_written(tmp_path):
    p, g = synthetic()
    f = P.freeze_policies(p, g, CFG, NAMES)
    P.plot_cost_vs_threshold(p, g, CFG, NAMES, f["primary"], tmp_path / "a.png")
    s = P.sensitivity(p, g, CFG, NAMES, ratios=[2, 10], review_costs=[0.1, 0.3], budget=2, k=4, n_boot=100)
    P.plot_heatmap(s, tmp_path / "b.png")
    oof, _ = P.cross_validate_matched(p, g, CFG, NAMES, budgets=[2], accs=[0.9], k=4)
    t = P.matched_table(oof, [2], [0.9], 100, 0)
    P.plot_savings(t, pd.DataFrame({"policy": ["x"], "saving_pct": [5.0], "pct_lo": [1.0], "pct_hi": [9.0]}),
                   tmp_path / "c.png")
    for n in "abc":
        assert (tmp_path / f"{n}.png").stat().st_size > 1000
