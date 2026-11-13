"""Tests for src/evaluate.py (Phase 3b) using small hand-computed examples."""

import itertools
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src import decision as d
from src import evaluate as E

NAMES = ["a"]
CFG = {
    "cost_miss": 10, "cost_false_alarm": 1, "cost_review": 0.3,
    "class_weights": {"a": 1.0, "b": 1.0},
    "area_severity": {"area_frac_small": 0.02, "area_frac_large": 0.20, "factor_small": 1.0, "factor_large": 1.0},
    "action_bounds": {"ignore_below": 0.5, "monitor_below": 0.8, "service_below": 1.3},
    "review_band_width": 0.15,
}
BOX, FAR = (0, 0, 10, 10), (50, 50, 60, 60)


def P(image, cls, conf, box):
    return dict(kind="pred", image=image, class_id=cls, conf=conf, x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                img_w=100, img_h=100)


def G(image, cls, box=BOX):
    return dict(kind="gt", image=image, class_id=cls, conf=float("nan"), x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                img_w=100, img_h=100)


def matched(rows):
    df = pd.DataFrame(rows)
    return d.match_detections(df[df.kind == "pred"].reset_index(drop=True), df[df.kind == "gt"].reset_index(drop=True))


# Image i: GT, TP at 0.9, one false detection at 0.4.  Image j: GT whose only detection is a TP at 0.4.
TWO_IMAGES = [G("i", 0), P("i", 0, 0.9, BOX), P("i", 0, 0.4, FAR), G("j", 0), P("j", 0, 0.4, BOX)]

# One class, two images. GT1 (image i) found at 0.9; GT2 (image j) found only at 0.2;
# four false detections at 0.5, 0.45, 0.4, 0.35.  Used to show when a band helps and when it does not.
BAND_HELPS = [G("i", 0), G("j", 0), P("i", 0, 0.9, BOX), P("j", 0, 0.2, BOX),
              P("i", 0, 0.5, FAR), P("i", 0, 0.45, FAR), P("i", 0, 0.4, FAR), P("i", 0, 0.35, FAR)]


# ---- per-image outcomes and summary metrics ---------------------------------------------------

def test_policy_outcomes_per_image_hand_computed():
    p, g = matched(TWO_IMAGES)
    o = d.policy_outcomes(p, g, {0: (0.3, 0.5)}, CFG, NAMES, reviewer_accuracy=0.9)
    # image i: alert 0.9 (TP); 0.4 is a false detection in the band -> reviewed, falsely confirmed w.p. 0.1.
    #   cost = 0.3 review + 0.1 * 1 false alarm = 0.4
    assert o.loc["i", ["alerts", "reviews", "false_alarms", "gt", "caught", "miss_cost"]].tolist() == \
        pytest.approx([1, 1, 0.1, 1, 1.0, 0.0])
    assert o.loc["i", "cost"] == pytest.approx(0.4)
    # image j: the real defect is only in the band -> reviewed (0.3), caught w.p. 0.9, missed w.p. 0.1 (10*0.1=1)
    assert o.loc["j", ["alerts", "reviews", "false_alarms", "gt", "caught", "miss_cost"]].tolist() == \
        pytest.approx([0, 1, 0.0, 1, 0.9, 1.0])
    assert o.loc["j", "cost"] == pytest.approx(1.3)


def test_summarise_hand_computed():
    p, g = matched(TWO_IMAGES)
    o = d.policy_outcomes(p, g, {0: (0.3, 0.5)}, CFG, NAMES, reviewer_accuracy=0.9)
    s = E.summarise(o)
    assert s["cost"] == pytest.approx(1.7)             # 0.4 + 1.3
    assert s["cost_per_image"] == pytest.approx(0.85)
    assert s["recall"] == pytest.approx(0.95)          # (1 + 0.9) / 2 GT
    assert s["alerts_per_image"] == pytest.approx(0.5)  # 1 alert / 2 images
    assert s["reviews_per_image"] == pytest.approx(1.0)  # 2 reviews / 2 images
    assert s["false_alarms_per_image"] == pytest.approx(0.05)  # 0.1 / 2


def test_policy_outcomes_total_matches_total_cost_and_zero_width_band_never_reviews():
    p, g = matched(TWO_IMAGES)
    o = d.policy_outcomes(p, g, {0: (0.5, 0.5)}, CFG, NAMES)
    assert o.reviews.sum() == 0
    # alert only on 0.9: image j's defect (0.4) missed = 10; nothing else costs -> 10
    assert o.cost.sum() == pytest.approx(10.0)
    assert o.cost.sum() == pytest.approx(d.total_cost(p, g, {0: 0.5}, CFG, NAMES))


# ---- vectorised table agrees with the reference evaluator -------------------------------------

@pytest.mark.parametrize("a", [1.0, 0.9, 0.8])
def test_class_table_matches_policy_outcomes_for_every_pair(a):
    p, g = matched(BAND_HELPS + TWO_IMAGES[:0])
    pairs = E.band_pairs()
    tab = E.class_table(p, g, 0, CFG, NAMES, a, pairs)
    for (lo, hi), cost, load in zip(pairs, tab.cost, tab.load):
        o = d.policy_outcomes(p, g, {0: (lo, hi)}, CFG, NAMES, a)
        assert cost == pytest.approx(o.cost.sum()), (lo, hi)
        assert load == o.alerts.sum() + o.reviews.sum(), (lo, hi)


def test_band_pairs_include_zero_width_and_off():
    pairs = E.band_pairs()
    assert (0.25, 0.25) in pairs and (E.OFF, E.OFF) in pairs and (0.1, 0.5) in pairs
    assert all(lo <= hi for lo, hi in pairs)
    assert all(lo == hi for lo, hi in E.band_pairs(allow_band=False))


# ---- joint band tuning ------------------------------------------------------------------------

def test_tune_bands_picks_a_band_when_review_is_reliable():
    p, g = matched(BAND_HELPS)
    # Hand check, a=1: alert-only best is hi=0.2 (catch both, 4 FA) = 4. Band [0.2, 0.9): alert 0.9, review the
    # five items 0.5, 0.45, 0.4, 0.35, 0.2 = 5*0.3 = 1.5. Cheaper -> band; ties on hi resolve to the highest (0.9).
    assert E.tune_bands(p, g, CFG, NAMES, 1.0, 2) == {0: (0.2, 0.9)}
    assert E.tune_bands(p, g, CFG, NAMES, 1.0, 2, allow_band=False) == {0: (0.2, 0.2)}


def test_tune_bands_allows_zero_width_band_when_review_is_unreliable():
    p, g = matched(BAND_HELPS)
    # a=0.5: the same band costs 5*0.3 + 4*0.5 (falsely confirmed) + 10*0.5 (dismissed GT2) = 8.5 > 4 for alert-only
    assert E.tune_bands(p, g, CFG, NAMES, 0.5, 2) == {0: (0.2, 0.2)}


def test_tune_bands_all_false_detections_means_off():
    p, g = matched([P("i", 0, 0.9, FAR), P("i", 0, 0.4, FAR), G("j", 1)])
    assert E.tune_bands(p, g, CFG, ["a", "b"], 1.0, 2)[0] == (E.OFF, E.OFF)


def test_tune_bands_exact_tie_breaking_prefers_fewer_reviews():
    # A single TP at 0.9. (0.9, 0.9) costs 0 with load 1; nothing cheaper exists, and a wider band costs more.
    p, g = matched([G("i", 0), P("i", 0, 0.9, BOX), P("i", 0, 0.6, FAR)])
    assert E.tune_bands(p, g, CFG, NAMES, 1.0, 1) == {0: (0.9, 0.9)}


# ---- alert budget -----------------------------------------------------------------------------

def _tab(rows):
    return pd.DataFrame(rows, columns=["cost", "load"])


def test_budget_dp_hand_computed_multiple_choice_knapsack():
    A = _tab([(10, 0), (4, 1), (1, 3)])   # class A options: (cost, load)
    B = _tab([(8, 0), (3, 2)])
    # cap 0: A0+B0 = 18 ; cap 1: A1+B0 = 12 ; cap 2: A1+B0 = 12 (A0+B2 = 13) ;
    # cap 3: A1+B2 = 7 (A3+B0 = 9) ; cap 5: A3+B2 = 4
    for cap, picks in [(0, [0, 0]), (1, [1, 0]), (2, [1, 0]), (3, [1, 1]), (5, [2, 1])]:
        assert E._budget_dp([A, B], cap) == picks, cap


def test_budget_dp_matches_brute_force_on_random_tables():
    rng = np.random.default_rng(0)
    for _ in range(20):
        # like real tables, every class has a zero-load "do nothing" option, so a solution always exists
        tabs = [_tab([(float(rng.integers(0, 30)), 0)]
                     + [(float(rng.integers(0, 30)), int(rng.integers(0, 6))) for _ in range(5)]) for _ in range(3)]
        cap = int(rng.integers(0, 12))
        best = min((sum(t.cost[i] for t, i in zip(tabs, combo)), combo)
                   for combo in itertools.product(range(6), repeat=3)
                   if sum(t.load[i] for t, i in zip(tabs, combo)) <= cap)
        picked = E._budget_dp(tabs, cap)
        assert sum(t.cost[i] for t, i in zip(tabs, picked)) == pytest.approx(best[0])
        assert sum(t.load[i] for t, i in zip(tabs, picked)) <= cap


def test_tune_bands_budget_is_respected_and_hand_computed():
    p, g = matched(BAND_HELPS)
    # 2 images, budget 2/img -> at most 4 alerts+reviews. Every option that catches GT2 (conf 0.2) needs load 6,
    # so GT2 stays missed (10). Best: alert only on 0.9 -> cost 10 at load 1 (highest tied upper = 0.9).
    bands = E.tune_bands(p, g, CFG, NAMES, 1.0, 2, budget=2)
    assert bands == {0: (0.9, 0.9)}
    o = d.policy_outcomes(p, g, bands, CFG, NAMES)
    assert o.cost.sum() == pytest.approx(10.0)
    assert (o.alerts + o.reviews).sum() <= 2 * 2
    # budget 3/img -> 6 allowed: the unconstrained optimum (cost 1.5, load 6) becomes feasible
    assert E.tune_bands(p, g, CFG, NAMES, 1.0, 2, budget=3) == {0: (0.2, 0.9)}


def test_budget_never_beats_unconstrained_and_is_monotone():
    p, g = matched(BAND_HELPS)
    costs = [d.policy_outcomes(p, g, E.tune_bands(p, g, CFG, NAMES, 1.0, 2, budget=B), CFG, NAMES).cost.sum()
             for B in (0.5, 1, 2, 3, 5)]
    free = d.policy_outcomes(p, g, E.tune_bands(p, g, CFG, NAMES, 1.0, 2), CFG, NAMES).cost.sum()
    assert costs == sorted(costs, reverse=True)
    assert min(costs) >= free - 1e-9


# ---- cross-validation and bootstrap -----------------------------------------------------------

def test_make_folds_partition_images_and_stratify():
    rows = [G(f"{c}{i}", c) for c in range(2) for i in range(10)]
    gts = pd.DataFrame(rows)
    images = sorted(gts.image)
    folds = E.make_folds(gts, images, k=5, seed=0)
    assert len(folds) == 5
    tests = [im for _, te in folds for im in te]
    assert sorted(tests) == images                       # every image held out exactly once
    for tr, te in folds:
        assert not set(tr) & set(te) and len(tr) + len(te) == 20
        assert sorted(int(im[0]) for im in te) == [0, 0, 1, 1]  # 2 images of each class per fold


def test_cross_validate_scores_held_out_images_only_and_is_reproducible():
    rows = []
    for i in range(10):
        rows += [G(f"i{i}", 0), P(f"i{i}", 0, 0.9, BOX), P(f"i{i}", 0, 0.1, FAR)]
    p, g = matched(rows)
    oof1, _ = E.cross_validate(p, g, CFG, NAMES, k=5)
    oof2, _ = E.cross_validate(p, g, CFG, NAMES, k=5)
    o = oof1["no alerts"]
    assert len(o) == 10 and o.cost.sum() == pytest.approx(100.0)   # 10 GT all missed, no alerts at all
    # flat 0.25 alerts only on the 0.9 TPs: nothing else costs anything
    assert oof1["flat 0.25"].cost.sum() == pytest.approx(0.0)
    assert oof1["tuned band, acc=1"].cost.sum() == pytest.approx(0.0)
    pd.testing.assert_frame_equal(oof1["tuned band, acc=0.9"], oof2["tuned band, acc=0.9"])


def test_bootstrap_saving_constant_saving_has_degenerate_ci():
    base, pol = np.full(50, 5.0), np.full(50, 3.0)
    r = E.bootstrap_saving(base, pol, n_boot=200, seed=1)
    assert r["saving"] == pytest.approx(100.0)       # 50 * 2
    assert r["saving_lo"] == pytest.approx(100.0) and r["saving_hi"] == pytest.approx(100.0)
    assert r["saving_pct"] == pytest.approx(40.0) and r["pct_lo"] == pytest.approx(40.0)


def test_bootstrap_saving_ci_contains_estimate_and_widens_with_noise():
    rng = np.random.default_rng(3)
    base = rng.uniform(0, 10, 200)
    pol = base - rng.normal(0.5, 2.0, 200)
    r = E.bootstrap_saving(base, pol, n_boot=2000, seed=0)
    assert r["saving_lo"] < r["saving"] < r["saving_hi"]
    assert r["saving_hi"] - r["saving_lo"] > 0


# ---- guards -----------------------------------------------------------------------------------

def test_evaluate_cli_refuses_test_split_cache():
    r = subprocess.run([sys.executable, "-m", "src.evaluate", "--cache", "data/cache/test_detections.parquet"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "reserved" in r.stderr
