"""Tests for src/robustness.py: weighted AP, per-image metrics, paired bootstrap and the test-split guard."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import average_precision
from src.robustness import METRICS, Prepared, collect_perturbed, paired_bootstrap, tag, weighted_ap

BOX = (0, 0, 100, 100)


def row(kind, image, cls, conf, box=BOX):
    return dict(kind=kind, image=image, class_id=cls, conf=conf, x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                img_w=200, img_h=200)


def two_image_run(conf_a=0.9, conf_b=0.05):
    """GT of class 0 in images a and b. a is found at conf_a; b has an exact detection at conf_b."""
    return pd.DataFrame([row("gt", "a", 0, np.nan), row("pred", "a", 0, conf_a),
                         row("gt", "b", 0, np.nan), row("pred", "b", 0, conf_b)])


def test_tag():
    assert tag("blur", 0) == "clean" and tag("noise", 2) == "noise_2"


def test_weighted_ap_matches_unweighted_ap():
    rng = np.random.default_rng(0)
    n = 60
    img = rng.integers(0, 10, n)
    conf = rng.uniform(0, 1, n)
    tp = (rng.uniform(size=n) < conf).astype(float)
    o = np.argsort(-conf, kind="stable")
    ref = average_precision(conf, tp.astype(bool), 30)
    assert weighted_ap(img[o], tp[o], np.ones(10), 30) == pytest.approx(ref)
    # every image counted twice: same precision-recall curve
    assert weighted_ap(img[o], tp[o], 2 * np.ones(10), 60) == pytest.approx(ref)


def test_weighted_ap_zero_weight_equals_dropping_the_image():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 5, 40)
    conf = rng.uniform(0, 1, 40)
    tp = (rng.uniform(size=40) < 0.5).astype(float)
    o = np.argsort(-conf, kind="stable")
    img, tp, conf = img[o], tp[o], conf[o]
    keep = img != 2
    ref = average_precision(conf[keep], tp[keep].astype(bool), 20)
    assert weighted_ap(img, tp, np.array([1, 1, 0, 1, 1.0]), 20) == pytest.approx(ref)


def test_weighted_ap_edge_cases():
    assert np.isnan(weighted_ap(np.array([0]), np.array([1.0]), np.ones(1), 0.0))
    assert weighted_ap(np.array([], int), np.array([]), np.ones(1), 3.0) == 0.0
    assert weighted_ap(np.array([0]), np.array([1.0]), np.zeros(1), 1.0) == 0.0


def test_prepared_metrics_hand_computed():
    p = Prepared(two_image_run(), ["a", "b"], threshold=0.127)
    m = p.metrics()
    assert m["recall"] == pytest.approx(0.5)             # b's detection (0.05) is below the policy threshold
    assert m["alerts_per_image"] == pytest.approx(0.5)   # only a's detection is kept
    assert m["AP50_crazing"] == pytest.approx(1.0, abs=0.01)  # AP uses every detection, both are TPs
    assert np.isnan(m["AP50_inclusion"])                  # no GT of that class
    assert m["mAP50"] == pytest.approx(m["AP50_crazing"])    # nanmean over classes that have GT
    assert p.metrics(np.array([2.0, 0.0]))["recall"] == pytest.approx(1.0)   # a bootstrap draw with only a
    assert p.metrics(np.array([0.0, 3.0]))["recall"] == pytest.approx(0.0)


def test_prepared_wrong_localisation_is_not_a_hit():
    df = pd.DataFrame([row("gt", "a", 0, np.nan), row("pred", "a", 0, 0.9, (0, 0, 100, 30))])  # IoU 0.3
    assert Prepared(df, ["a"]).metrics()["recall"] == 0.0
    assert Prepared(df, ["a"], iou=0.3).metrics()["recall"] == 1.0


def test_paired_bootstrap_identical_runs_have_zero_delta():
    p = Prepared(two_image_run(), ["a", "b"])
    bt = paired_bootstrap(p, p, n_boot=50, seed=0).set_index("metric")
    assert set(bt.index) == set(METRICS)
    for col in ("delta", "delta_lo", "delta_hi"):
        assert (bt[col].fillna(0) == 0).all()


def test_paired_bootstrap_detects_a_real_drop_and_is_reproducible():
    images = [f"i{k}" for k in range(30)]
    clean = pd.DataFrame([r for im in images for r in (row("gt", im, 0, np.nan), row("pred", im, 0, 0.9))])
    broken = pd.DataFrame([r for im in images for r in (row("gt", im, 0, np.nan), row("pred", im, 0, 0.05))])
    pc, pb = Prepared(clean, images), Prepared(broken, images)
    bt = paired_bootstrap(pc, pb, n_boot=200, seed=0)
    t = bt.set_index("metric")
    assert t.loc["recall", "delta"] == pytest.approx(-1.0)
    assert t.loc["recall", "delta_hi"] < 0                     # CI entirely below zero
    assert t.loc["alerts_per_image", "delta"] == pytest.approx(-1.0)
    assert t.loc["mAP50", "delta"] == pytest.approx(0.0, abs=1e-6)   # AP uses every detection: ranking unchanged
    pd.testing.assert_frame_equal(bt, paired_bootstrap(pc, pb, n_boot=200, seed=0))


def test_test_split_is_refused():
    with pytest.raises(ValueError, match="test split"):
        collect_perturbed("models/best.pt", "data/yolo", "test", "blur", 1)
