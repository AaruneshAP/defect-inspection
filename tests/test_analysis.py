"""Tests for src/analysis.py using small hand-built boxes (IoUs below are computed by hand)."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import (average_precision, breakdown, categorise_dets, categorise_gts, DET_CATS, GT_CATS,
                          gt_stats, iou_matrix, match, per_class_ap, select_false_alarms, select_misses)

GT_BOX = (0, 0, 100, 100)  # area 10000 in a 200x200 image


def row(image, cls, box, conf=None, kind="pred"):
    return dict(kind=kind, image=image, class_id=cls, conf=conf, x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                img_w=200, img_h=200)


def gt(image, cls=0, box=GT_BOX):
    return row(image, cls, box, None, "gt")


def frames(rows):
    df = pd.DataFrame(rows)
    return df[df.kind == "pred"].reset_index(drop=True), df[df.kind == "gt"].reset_index(drop=True)


# ---------------------------------------------------------------- iou / matching
def test_iou_matrix_hand_values():
    m = iou_matrix(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]))
    assert m[0].tolist() == pytest.approx([1.0, 1 / 3, 0.0])


def test_match_is_one_to_one_highest_confidence_wins():
    p, g = frames([gt("a"), row("a", 0, GT_BOX, 0.6), row("a", 0, GT_BOX, 0.9)])
    assert match(p, g, 0.5).tolist() == [-1, 0]  # the 0.9 detection takes the only GT


def test_match_is_class_aware():
    p, g = frames([gt("a", 0), row("a", 1, GT_BOX, 0.9)])
    assert match(p, g, 0.5).tolist() == [-1]


def test_match_iou_threshold_is_inclusive():
    # (0,0,100,50) has IoU exactly 0.5 with GT_BOX; (0,0,100,49) has 0.49
    p, g = frames([gt("a"), row("a", 0, (0, 0, 100, 50), 0.9), gt("b"), row("b", 0, (0, 0, 100, 49), 0.9)])
    m = match(p, g, 0.5)
    assert m[0] >= 0 and m[1] == -1


# ---------------------------------------------------------------- GT categories
def gt_categories(rows, thr=0.127):
    p, g = frames(rows)
    return categorise_gts(p[p.conf >= thr], g).set_index("image").category.to_dict()


def test_gt_categories_cover_every_case():
    rows = [
        gt("tp"), row("tp", 0, GT_BOX, 0.9),
        gt("none"),
        gt("loc"), row("loc", 0, (0, 0, 100, 30), 0.9),            # IoU 0.30
        gt("far"), row("far", 0, (0, 0, 100, 5), 0.9),             # IoU 0.05
        gt("wrongcls"), row("wrongcls", 1, GT_BOX, 0.9),           # other class, IoU 1
        gt("lowconf"), row("lowconf", 0, GT_BOX, 0.05),            # below threshold -> not kept
        gt("shared1", box=(0, 0, 100, 100)), gt("shared2", box=(0, 0, 100, 100)),
    ]
    cats = gt_categories(rows)
    assert cats["tp"] == "tp"
    assert cats["none"] == "missed"
    assert cats["loc"] == "poorly_localized"
    assert cats["far"] == "missed"
    assert cats["wrongcls"] == "wrong_class"
    assert cats["lowconf"] == "missed"


def test_gt_localization_boundaries():
    # IoU exactly 0.1 -> missed (needs > 0.1); just above -> poorly localized
    # IoU exactly 0.5 -> tp; just below -> poorly localized
    rows = [gt("i10"), row("i10", 0, (0, 0, 100, 10), 0.9),        # 0.10
            gt("i11"), row("i11", 0, (0, 0, 100, 11), 0.9),        # 0.11
            gt("i50"), row("i50", 0, (0, 0, 100, 50), 0.9),        # 0.50
            gt("i49"), row("i49", 0, (0, 0, 100, 49), 0.9)]        # 0.49
    cats = gt_categories(rows)
    assert cats == {"i10": "missed", "i11": "poorly_localized", "i50": "tp", "i49": "poorly_localized"}


def test_gt_shared_detection_when_two_gts_have_one_detection():
    rows = [gt("a"), gt("a"), row("a", 0, GT_BOX, 0.9)]
    p, g = frames(rows)
    c = categorise_gts(p, g).category.tolist()
    assert sorted(c) == ["shared", "tp"]


def test_confidence_threshold_changes_category():
    rows = [gt("a"), row("a", 0, GT_BOX, 0.2)]
    assert gt_categories(rows, thr=0.127)["a"] == "tp"
    assert gt_categories(rows, thr=0.3)["a"] == "missed"


# ---------------------------------------------------------------- detection categories
def test_detection_categories():
    rows = [
        gt("tp"), row("tp", 0, GT_BOX, 0.9),
        gt("dup"), row("dup", 0, GT_BOX, 0.9), row("dup", 0, GT_BOX, 0.8),
        gt("wrong"), row("wrong", 1, GT_BOX, 0.9),
        gt("loc"), row("loc", 0, (0, 0, 100, 30), 0.9),
        row("bg", 0, (150, 150, 190, 190), 0.9),                    # image without GT
        gt("bg2"), row("bg2", 0, (150, 150, 190, 190), 0.9),        # GT elsewhere
        gt("tenth"), row("tenth", 0, (0, 0, 100, 10), 0.9),         # IoU exactly 0.1 -> background
        gt("wrongloc", cls=1), row("wrongloc", 0, (0, 0, 100, 30), 0.9),  # other-class IoU 0.3 only -> background
    ]
    p, g = frames(rows)
    c = categorise_dets(p, g)
    got = {(r.image, r.conf): r.category for r in c.itertuples()}
    assert got[("tp", 0.9)] == "tp"
    assert got[("dup", 0.9)] == "tp" and got[("dup", 0.8)] == "duplicate"
    assert got[("wrong", 0.9)] == "wrong_class"
    assert got[("loc", 0.9)] == "poorly_localized"
    assert got[("bg", 0.9)] == "background"
    assert got[("bg2", 0.9)] == "background"
    assert got[("tenth", 0.9)] == "background"
    assert got[("wrongloc", 0.9)] == "background"


def test_breakdown_counts_and_shares():
    rows = [gt("a"), row("a", 0, GT_BOX, 0.9), gt("b"), gt("c"), row("c", 0, (0, 0, 100, 30), 0.9),
            gt("d", cls=1)]
    p, g = frames(rows)
    bd = breakdown(categorise_gts(p, g), GT_CATS, ["c0", "c1"])
    assert bd.loc["c0", ["n", "tp", "missed", "poorly_localized"]].tolist() == [3, 1, 1, 1]
    assert bd.loc["c0", "tp_share"] == pytest.approx(1 / 3, abs=1e-3)
    assert bd.loc["c1", "missed"] == 1 and bd.loc["all", "n"] == 4
    assert bd.loc["all", GT_CATS].sum() == bd.loc["all", "n"]
    dbd = breakdown(categorise_dets(p, g), DET_CATS, ["c0", "c1"])
    assert dbd.loc["all", "n"] == 2


# ---------------------------------------------------------------- AP / IoU threshold logic
def test_ap_perfect_empty_and_false_positive_first():
    assert average_precision(np.array([0.9]), np.array([True]), 1) == pytest.approx(1.0, abs=0.01)
    assert average_precision(np.array([]), np.array([], bool), 1) == 0.0
    assert np.isnan(average_precision(np.array([0.9]), np.array([False]), 0))
    # FP ranked above the only TP: precision 0.5 at recall 1 -> AP ~ 0.5
    assert average_precision(np.array([0.95, 0.9]), np.array([False, True]), 1) == pytest.approx(0.5, abs=0.02)


def test_ap_depends_on_iou_threshold():
    # one GT, one detection with IoU 0.3: a hit at IoU 0.1 and 0.3, a miss at 0.5 and 0.75
    p, g = frames([gt("a"), row("a", 0, (0, 0, 100, 30), 0.9)])
    ap = {t: per_class_ap(p, g, t, 1)[0] for t in (0.1, 0.3, 0.5, 0.75)}
    assert ap[0.1] == pytest.approx(1.0, abs=0.01) and ap[0.3] == pytest.approx(1.0, abs=0.01)
    assert ap[0.5] == 0.0 and ap[0.75] == 0.0


def test_ap_is_non_increasing_in_iou():
    rows = [gt("a"), row("a", 0, (0, 0, 100, 80), 0.9), gt("b"), row("b", 0, (0, 0, 100, 40), 0.8),
            gt("c"), row("c", 0, (0, 0, 100, 20), 0.7), gt("d"), row("d", 0, GT_BOX, 0.6)]
    p, g = frames(rows)
    aps = [per_class_ap(p, g, t, 1)[0] for t in (0.1, 0.3, 0.5, 0.75)]
    assert all(a >= b - 1e-9 for a, b in zip(aps, aps[1:]))
    assert aps[0] > aps[-1]


def test_ap_classes_are_independent():
    p, g = frames([gt("a", 0), row("a", 0, GT_BOX, 0.9), gt("b", 1)])
    assert per_class_ap(p, g, 0.5, 2) == pytest.approx([1.0, 0.0], abs=0.01)


# ---------------------------------------------------------------- GT stats
def test_gt_stats_overlap_area_and_aspect():
    rows = [gt("a", 0, (0, 0, 100, 100)),             # overlaps the next (IoU 1/3), area 0.25
            gt("a", 0, (50, 0, 150, 100)),
            gt("a", 0, (0, 150, 40, 190)),            # isolated, area 0.04
            gt("b", 0, (0, 0, 100, 50)),              # only box of its class in the image, aspect 2
            gt("a", 1, (0, 0, 100, 100)), gt("a", 1, (100, 100, 200, 200))]  # touching corners: no overlap
    _, g = frames(rows)
    s = gt_stats(g, ["c0", "c1"])
    assert s.loc["c0", "n_gt"] == 4 and s.loc["c0", "n_images"] == 2
    assert s.loc["c0", "frac_overlap_sibling"] == pytest.approx(0.5)
    assert s.loc["c0", "frac_overlap_sibling_iou30"] == pytest.approx(0.5)
    assert s.loc["c0", "max_sibling_iou_median"] == pytest.approx(round(1 / 3, 4) / 2, abs=1e-3)  # [1/3,1/3,0,0]
    assert s.loc["c0", "aspect_max_median"] == pytest.approx(1.0)  # {1,1,1,2}
    assert s.loc["c1", "frac_overlap_sibling"] == 0.0


# ---------------------------------------------------------------- selection
def test_select_misses_orders_missed_by_area_then_localization():
    rows = [gt("small", box=(0, 0, 20, 20)), gt("big", box=(0, 0, 150, 150)),
            gt("loc"), row("loc", 0, (0, 0, 100, 30), 0.9),
            gt("tp"), row("tp", 0, GT_BOX, 0.9)]
    p, g = frames(rows)
    sel = select_misses(categorise_gts(p, g), 0)
    assert sel.image.tolist() == ["big", "small", "loc"]


def test_select_false_alarms_sorted_by_confidence_and_class():
    rows = [gt("a"), row("a", 0, GT_BOX, 0.9),
            row("b", 0, (0, 0, 10, 10), 0.4), row("c", 0, (0, 0, 10, 10), 0.7), row("d", 1, (0, 0, 10, 10), 0.99)]
    p, g = frames(rows)
    sel = select_false_alarms(categorise_dets(p, g), 0)
    assert sel.image.tolist() == ["c", "b"]
