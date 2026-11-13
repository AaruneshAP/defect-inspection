"""Tests for src/decision.py and src/predict.py using small hand-computed examples."""

import math
import subprocess
import sys

import pandas as pd
import pytest

from src import decision as d
from src import predict

NAMES = ["a", "b"]
CFG = {
    "cost_miss": 10, "cost_false_alarm": 1, "cost_review": 0.3,
    "class_weights": {"a": 1.0, "b": 1.0},
    # flat factor 1.0 so hand arithmetic is trivial
    "area_severity": {"area_frac_small": 0.02, "area_frac_large": 0.20, "factor_small": 1.0, "factor_large": 1.0},
    "action_bounds": {"ignore_below": 0.5, "monitor_below": 0.8, "service_below": 1.3},
    "review_band_width": 0.15,
}
SEV_CFG = {**CFG, "area_severity": {"area_frac_small": 0.02, "area_frac_large": 0.20,
                                    "factor_small": 0.4, "factor_large": 2.0}}


def P(image, cls, conf, box):
    return dict(kind="pred", image=image, class_id=cls, conf=conf, x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                img_w=100, img_h=100)


def G(image, cls, box):
    return dict(kind="gt", image=image, class_id=cls, conf=float("nan"), x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                img_w=100, img_h=100)


def frames(rows):
    df = pd.DataFrame(rows)
    return df[df.kind == "pred"].reset_index(drop=True), df[df.kind == "gt"].reset_index(drop=True)


# One image, class 0: GT_A at (0,0,10,10), GT_B at (50,50,60,60).
# Detections: 0.9 hits A; 0.8 and 0.7 are in empty space. B is never detected.
ONE_MISS_TWO_FA = [
    G("i", 0, (0, 0, 10, 10)), G("i", 0, (50, 50, 60, 60)),
    P("i", 0, 0.9, (0, 0, 10, 10)), P("i", 0, 0.8, (80, 80, 90, 90)), P("i", 0, 0.7, (20, 20, 30, 30)),
]


def test_iou_known_values():
    assert d.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert d.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # overlap 5x10=50, union 100+100-50=150
    assert d.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_match_iou_boundary_is_inclusive():
    # IoU exactly 0.5: pred (0,0,10,10) vs gt (0,0,10,5) -> inter 50, union 100
    p, g = frames([G("i", 0, (0, 0, 10, 5)), P("i", 0, 0.9, (0, 0, 10, 10))])
    mp, mg = d.match_detections(p, g)
    assert mp.tp.tolist() == [True]
    # just under 0.5 -> no match
    p, g = frames([G("i", 0, (0, 0, 10, 4.9)), P("i", 0, 0.9, (0, 0, 10, 10))])
    assert d.match_detections(p, g)[0].tp.tolist() == [False]


def test_match_is_class_aware_and_one_to_one():
    rows = [
        G("i", 0, (0, 0, 10, 10)),
        P("i", 0, 0.9, (0, 0, 10, 10)),   # takes the GT
        P("i", 0, 0.6, (0, 0, 10, 10)),   # duplicate -> false alarm
        P("i", 1, 0.95, (0, 0, 10, 10)),  # wrong class -> false alarm
    ]
    mp, mg = d.match_detections(*frames(rows))
    assert mp.sort_values("conf", ascending=False).tp.tolist() == [False, True, False]
    assert mg.match_conf.tolist() == [0.9]


def test_match_prefers_higher_confidence_detection():
    # lower-confidence pred listed first must not steal the GT
    rows = [G("i", 0, (0, 0, 10, 10)), P("i", 0, 0.3, (0, 0, 10, 10)), P("i", 0, 0.8, (0, 0, 10, 10))]
    mp, mg = d.match_detections(*frames(rows))
    assert mg.match_conf.tolist() == [0.8]
    assert mp.set_index("conf").tp.to_dict() == {0.3: False, 0.8: True}


def test_total_cost_one_miss_two_false_alarms():
    p, g = d.match_detections(*frames(ONE_MISS_TWO_FA))
    # threshold 0.5: alerts 0.9(TP) 0.8(FA) 0.7(FA); GT_B missed -> 1*10 + 2*1 = 12
    assert d.total_cost(p, g, {0: 0.5}, CFG, NAMES) == pytest.approx(12.0)
    # threshold 0.95: nothing alerts -> both GTs missed -> 20
    assert d.total_cost(p, g, {0: 0.95}, CFG, NAMES) == pytest.approx(20.0)
    # threshold 0.85: only 0.9 alerts -> 1 miss (B) = 10
    assert d.total_cost(p, g, {0: 0.85}, CFG, NAMES) == pytest.approx(10.0)


def test_total_cost_with_review_band():
    p, g = d.match_detections(*frames(ONE_MISS_TWO_FA))
    # upper 0.85, lower 0.65: 0.9 alerts (TP, free); 0.8 and 0.7 reviewed: 2*0.3; B missed: 10
    assert d.total_cost(p, g, {0: 0.85}, CFG, NAMES, {0: 0.65}) == pytest.approx(10.6)


def test_review_catches_real_defect_below_alert_threshold():
    rows = [G("i", 0, (0, 0, 10, 10)), P("i", 0, 0.4, (0, 0, 10, 10))]
    p, g = d.match_detections(*frames(rows))
    # no review: miss 10. With band [0.3, 0.5): reviewed (0.3) and caught -> 0.3
    assert d.total_cost(p, g, {0: 0.5}, CFG, NAMES) == pytest.approx(10.0)
    assert d.total_cost(p, g, {0: 0.5}, CFG, NAMES, {0: 0.3}) == pytest.approx(0.3)


def test_total_cost_sums_classes_and_severity_weights_misses():
    cfg = {**SEV_CFG, "class_weights": {"a": 1.0, "b": 2.0}}
    # class b GT covers 50% of image (area factor clamps to 2.0) -> miss = 10 * 2.0 * 2.0 = 40
    rows = [G("i", 1, (0, 0, 100, 50))]
    p, g = d.match_detections(pd.DataFrame(columns=["kind", "image", "class_id", "conf", "x1", "y1", "x2", "y2",
                                                    "img_w", "img_h"]).astype({"conf": float, "class_id": int,
                                                                               "image": str}),
                              pd.DataFrame(rows))
    assert d.total_cost(p, g, {0: 0.5, 1: 0.5}, cfg, NAMES) == pytest.approx(40.0)


def test_optimize_thresholds_synthetic_optimum():
    # class 0: TP at 0.9 and 0.6; FAs at 0.5 and 0.2. Miss 10, FA 1.
    rows = [
        G("i", 0, (0, 0, 10, 10)), G("j", 0, (0, 0, 10, 10)),
        P("i", 0, 0.9, (0, 0, 10, 10)), P("j", 0, 0.6, (0, 0, 10, 10)),
        P("i", 0, 0.5, (50, 50, 60, 60)), P("j", 0, 0.2, (50, 50, 60, 60)),
    ]
    p, g = d.match_detections(*frames(rows))
    # t=0.2: 2 FA = 2; t=0.5: 1 FA = 1; t=0.6: 0 FA, 0 miss = 0; t=0.9: 1 miss = 10 -> optimum 0.6
    assert d.optimize_thresholds(p, g, CFG, NAMES)[0] == pytest.approx(0.6)
    t = d.optimize_thresholds(p, g, CFG, NAMES)
    assert d.total_cost(p, g, {0: t[0]}, CFG, NAMES) == pytest.approx(0.0)


def test_optimize_prefers_alerting_when_misses_are_expensive():
    # one TP at 0.1 and one FA at 0.8: alerting on the TP only costs 0, but threshold must go
    # low enough (0.1) to catch it, accepting the FA at 0.8 (1) instead of missing (10).
    rows = [G("i", 0, (0, 0, 10, 10)), P("i", 0, 0.1, (0, 0, 10, 10)), P("i", 0, 0.8, (50, 50, 60, 60))]
    p, g = d.match_detections(*frames(rows))
    assert d.optimize_thresholds(p, g, CFG, NAMES)[0] == pytest.approx(0.1)


def test_optimize_alerts_on_nothing_when_all_detections_are_false():
    rows = [P("i", 0, 0.9, (50, 50, 60, 60)), P("i", 0, 0.4, (70, 70, 80, 80))]
    p, g = d.match_detections(*frames(rows))
    # no GT in class 0 -> best is no alerts at all (threshold above 1)
    assert d.optimize_thresholds(p, g, CFG, NAMES)[0] > 1.0


def test_area_factor_and_severity():
    assert d.area_factor(0.0, SEV_CFG) == pytest.approx(0.4)
    assert d.area_factor(0.02, SEV_CFG) == pytest.approx(0.4)
    assert d.area_factor(0.11, SEV_CFG) == pytest.approx(1.2)  # midpoint of 0.4 and 2.0
    assert d.area_factor(0.9, SEV_CFG) == pytest.approx(2.0)
    cfg = {**SEV_CFG, "class_weights": {"a": 1.5, "b": 1.0}}
    # 100x100 image, 20x20 box = 0.04 of area -> factor 0.4 + 1.6*(0.02/0.18)
    expected = 1.5 * (0.4 + 1.6 * (0.02 / 0.18))
    assert d.severity("a", (0, 0, 20, 20), 100, 100, cfg) == pytest.approx(expected)


@pytest.mark.parametrize("conf,sev,expected", [
    (0.299, 9.0, "ignore"),            # below review band, whatever the severity
    (0.30, 9.0, "human_review"),       # lower bound inclusive
    (0.599, 9.0, "human_review"),
    (0.60, 0.49, "ignore"),            # upper bound inclusive -> severity bands; below ignore_below
    (0.60, 0.50, "monitor"),           # ignore_below inclusive upward
    (0.60, 0.79, "monitor"),
    (0.60, 0.80, "schedule_service"),
    (0.60, 1.29, "schedule_service"),
    (0.60, 1.30, "stop_now"),
    (1.00, 2.00, "stop_now"),
])
def test_decide_action_boundaries(conf, sev, expected):
    assert d.decide_action(conf, sev, 0.30, 0.60, CFG) == expected


def test_review_bands():
    b = d.review_bands({0: 0.5, 1: 0.1}, CFG)
    assert b[0] == pytest.approx(0.35)
    assert b[1] == 0.0


def test_default_cost_config_loads():
    cfg = d.load_cost_config("configs/cost.yaml")
    assert (cfg["cost_miss"], cfg["cost_false_alarm"], cfg["cost_review"]) == (10, 1, 0.3)
    assert len(set(cfg["class_weights"].values())) == 1  # equal by default
    assert len(cfg["class_weights"]) == 6


def test_read_gt_converts_yolo_to_pixels(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("2 0.5 0.5 0.5 0.25\n")
    (row,) = predict.read_gt(f, 200, 100, "x.jpg")
    assert row[0] == "gt" and row[2] == 2 and math.isnan(row[3])
    assert row[4:8] == (50.0, 37.5, 150.0, 62.5)


def test_predict_cli_requires_flag_for_test_split():
    r = subprocess.run([sys.executable, "-m", "src.predict", "--split", "test"], capture_output=True, text=True)
    assert r.returncode != 0
    assert "--allow-test" in r.stderr


# ---- Phase 3b: reviewer_accuracy -------------------------------------------------------------

def test_reviewer_accuracy_false_detection_in_band():
    p, g = d.match_detections(*frames(ONE_MISS_TWO_FA))
    # upper 0.85, lower 0.65: 0.9 alerts (TP); 0.8 and 0.7 are FALSE detections that get reviewed.
    # a=1.0: review 2*0.3 + miss B 10                      = 10.6
    # a=0.9: + 2 * (1-0.9) * 1 wrongly confirmed alarms    = 10.8
    # a=0.8: + 2 * 0.2 * 1                                  = 11.0
    for a, expected in [(1.0, 10.6), (0.9, 10.8), (0.8, 11.0)]:
        assert d.total_cost(p, g, {0: 0.85}, CFG, NAMES, {0: 0.65}, reviewer_accuracy=a) == pytest.approx(expected)


def test_reviewer_accuracy_real_defect_in_band():
    rows = [G("i", 0, (0, 0, 10, 10)), P("i", 0, 0.4, (0, 0, 10, 10))]
    p, g = d.match_detections(*frames(rows))
    # band [0.3, 0.5): review 0.3, caught w.p. a, else a miss of 10 -> 0.3 + 10*(1-a)
    for a, expected in [(1.0, 0.3), (0.9, 1.3), (0.8, 2.3), (0.0, 10.3)]:
        assert d.total_cost(p, g, {0: 0.5}, CFG, NAMES, {0: 0.3}, reviewer_accuracy=a) == pytest.approx(expected)


def test_reviewer_accuracy_read_from_config_and_override():
    rows = [G("i", 0, (0, 0, 10, 10)), P("i", 0, 0.4, (0, 0, 10, 10))]
    p, g = d.match_detections(*frames(rows))
    cfg = {**CFG, "reviewer_accuracy": 0.9}
    assert d.total_cost(p, g, {0: 0.5}, cfg, NAMES, {0: 0.3}) == pytest.approx(1.3)           # from config
    assert d.total_cost(p, g, {0: 0.5}, cfg, NAMES, {0: 0.3}, reviewer_accuracy=1.0) == pytest.approx(0.3)  # override
    assert d.total_cost(p, g, {0: 0.5}, CFG, NAMES, {0: 0.3}) == pytest.approx(0.3)           # absent -> 1.0


def test_reviewer_accuracy_does_not_affect_alert_only_policies():
    p, g = d.match_detections(*frames(ONE_MISS_TWO_FA))
    for a in (1.0, 0.5, 0.0):
        assert d.total_cost(p, g, {0: 0.5}, CFG, NAMES, reviewer_accuracy=a) == pytest.approx(12.0)


@pytest.mark.parametrize("bad", [-0.1, 1.01])
def test_reviewer_accuracy_must_be_a_probability(bad):
    with pytest.raises(ValueError):
        d.get_reviewer_accuracy(CFG, bad)


def test_default_config_has_perfect_reviewer():
    assert d.load_cost_config("configs/cost.yaml")["reviewer_accuracy"] == 1.0
