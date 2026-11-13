"""Tests for the match_iou plumbing in src/phase3c.py (exploratory matching-IoU sensitivity)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import phase3c as P
from src.decision import match_detections

CFG = {"cost_miss": 10, "cost_false_alarm": 1, "cost_review": 0.3, "class_weights": {"a": 1.0, "b": 1.0},
       "area_severity": {"area_frac_small": 0.02, "area_frac_large": 0.20, "factor_small": 1.0, "factor_large": 1.0},
       "action_bounds": {"ignore_below": 0.5, "monitor_below": 0.8, "service_below": 1.3}, "review_band_width": 0.15}
GT = (0, 0, 100, 100)
LOOSE = (0, 0, 100, 40)          # IoU 0.4 with GT: a hit at match_iou 0.3, not at 0.5


def row(kind, image, cls, conf, box):
    return dict(kind=kind, image=image, class_id=cls, conf=conf, x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                img_w=100, img_h=100)


@pytest.fixture
def files(tmp_path):
    rows = []
    for i in range(10):          # ten images, one GT each, one detection with IoU 0.4 at conf 0.9
        rows += [row("gt", f"i{i}", 0, np.nan, GT), row("pred", f"i{i}", 0, 0.9, LOOSE)]
    cache, cfg = tmp_path / "val.parquet", tmp_path / "cost.yaml"
    pd.DataFrame(rows).to_parquet(cache)
    cfg.write_text(yaml.safe_dump(CFG))
    return str(cache), str(cfg)


def test_default_is_the_frozen_setting(files):
    assert P.MATCH_IOU == 0.5
    cache, cfg = files
    p_default, g_default, *_ = P._load(cache, cfg)
    p_half, g_half, *_ = P._load(cache, cfg, 0.5)
    pd.testing.assert_frame_equal(p_default, p_half)
    pd.testing.assert_frame_equal(g_default, g_half)
    assert not p_default.tp.any() and g_default.match_conf.isna().all()     # IoU 0.4 is not a hit at 0.5


def test_looser_match_iou_turns_the_detection_into_a_hit(files):
    cache, cfg = files
    p, g, *_ = P._load(cache, cfg, 0.3)
    assert p.tp.all() and (g.match_conf == 0.9).all()


def test_match_iou_is_monotone_on_random_boxes():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        x, y = rng.uniform(0, 40, 2)
        rows.append(row("gt", f"i{i}", 0, np.nan, (x, y, x + 50, y + 50)))
        dx, dy = rng.uniform(-30, 30, 2)
        rows.append(row("pred", f"i{i}", 0, float(rng.uniform()), (x + dx, y + dy, x + dx + 50, y + dy + 50)))
    df = pd.DataFrame(rows)
    n_tp = [match_detections(df[df.kind == "pred"], df[df.kind == "gt"], t)[0].tp.sum() for t in (0.7, 0.5, 0.3, 0.1)]
    assert n_tp == sorted(n_tp) and n_tp[0] < n_tp[-1]


def test_sensitivity_hand_computed(files):
    cache, cfg = files
    tab, thr = P.match_iou_sensitivity(cache, cfg, [0.5, 0.3], budget=3, k=5, seed=0, n_boot=200)
    t = tab.set_index(["match_iou", "policy"])
    # IoU 0.5: alerting costs 1 false alarm + a 10 miss per image, so matched flat never alerts: cost 100, recall 0
    assert t.loc[(0.5, "matched flat"), "cost"] == pytest.approx(100.0)
    assert t.loc[(0.5, "matched flat"), "recall"] == 0.0 and t.loc[(0.5, "matched flat"), "alerts_per_image"] == 0.0
    # flat 0.25 alerts on every image: 10 false alarms + 10 misses = 110
    assert t.loc[(0.5, "flat 0.25"), "cost"] == pytest.approx(110.0)
    # IoU 0.3: every detection is a hit, so alerting is free: cost 0, recall 1, one alert per image
    assert t.loc[(0.3, "matched flat"), "cost"] == pytest.approx(0.0)
    assert t.loc[(0.3, "matched flat"), "recall"] == pytest.approx(1.0)
    assert t.loc[(0.3, "matched flat"), "alerts_per_image"] == pytest.approx(1.0)
    assert t.loc[(0.3, "flat 0.25"), "cost"] == pytest.approx(0.0)
    assert t.loc[(0.5, "matched flat"), "saving_vs_flat025"] == pytest.approx(10.0)
    assert tab.exploratory.all() and thr.exploratory.all()
    th = thr.set_index("match_iou")
    assert th.loc[0.5, "matched_flat_threshold_full_val"] == P.OFF
    assert 0.3 < th.loc[0.3, "matched_flat_threshold_full_val"] <= 0.9
    assert th.loc[0.5, "n_gt_matched"] == 0 and th.loc[0.3, "n_gt_matched"] == 10


def test_frozen_matched_threshold_is_unchanged_at_default():
    """Regression: on the real val cache the default match IoU still gives the frozen matched flat threshold."""
    cache = Path("data/cache/val_detections.parquet")
    if not cache.exists():
        pytest.skip("val cache not available")
    preds, gts, cfg, names = P._load(str(cache), "configs/cost.yaml")
    t = P.best_flat_threshold(preds, gts, cfg, names, len(set(preds.image) | set(gts.image)), 3)
    assert t == pytest.approx(0.127)


def test_cli_refuses_frozen_dir_and_test_cache(files):
    cache, cfg = files

    class A:
        out, val_cache, config = "reports/phase3", cache, cfg
        match_ious, folds, seed, n_boot = [0.3], 5, 0, 10

    with pytest.raises(SystemExit):
        P.cmd_match_iou(A)
    A.out, A.val_cache = "somewhere", "data/cache/test_detections.parquet"
    with pytest.raises(SystemExit):
        P.cmd_match_iou(A)
