"""Trigger-logic tests on synthetic flag frames."""

import pandas as pd
import pytest

from src.drift import FEATURES, INPUT_FEATURES, PREDICTION_FEATURES
from src.triggers import joint_flags, select_km, tier1, tier2


def flags(rows):
    """rows: list of sets of flagged feature names, one per window."""
    return pd.DataFrame([{f: f in r for f in FEATURES} for r in rows], index=pd.RangeIndex(100, 100 + len(rows)))


def test_tier1_counts_flagged_features():
    f = flags([set(), {"sharpness"}, {"sharpness", "noise_est"}])
    assert tier1(f, 1).tolist() == [False, True, True]
    assert tier1(f, 2).tolist() == [False, False, True]


def test_joint_needs_an_input_and_a_prediction_feature():
    f = flags([{"sharpness"}, {"max_confidence"}, {"sharpness", "n_detections"}, {"brightness_mean", "max_confidence"}])
    assert joint_flags(f).tolist() == [False, False, True, True]


def test_input_drift_alone_never_recommends_retraining():
    f = flags([set(INPUT_FEATURES)] * 50)
    assert tier1(f, 2).all() and not tier2(f, 1, 5).any()
    g = flags([set(PREDICTION_FEATURES)] * 50)
    assert not tier2(g, 1, 5).any()


def test_tier2_is_k_of_last_m_windows():
    j = {"sharpness", "max_confidence"}
    f = flags([j, set(), set(), j, set(), set(), set(), set()])  # joint at windows 0 and 3
    out = tier2(f, 2, 4)
    assert out.tolist() == [False, False, False, True, False, False, False, False]  # only window 3 sees both
    assert tier2(f, 1, 3).tolist() == [False, False, True, True, True, True, False, False]


def test_tier2_needs_a_full_history_and_valid_args():
    f = flags([{"sharpness", "max_confidence"}] * 3)
    assert not tier2(f, 1, 5).any()
    with pytest.raises(ValueError):
        tier2(f, 6, 5)
    with pytest.raises(ValueError):
        tier2(f, 0, 5)


def test_select_km_obeys_the_false_trigger_budget_and_prefers_fast():
    from src.drift import EvalSetup

    setup = EvalSetup(drift_start=3, horizon=40)
    j = {"sharpness", "max_confidence"}
    n = 60
    clean = [flags([j if i in (10, 11) else set() for i in range(n)])]  # a 2-window blip in a clean stream
    drifted = [flags([set() if i < 3 else j for i in range(n)])]
    fc = {"clean": clean, ("blur", 1): drifted, ("noise", 1): drifted, ("contrast", 1): drifted}
    cfg, grid = select_km(fc, setup, ms=(3, 5), ks=(1, 2, 3, 5), max_fa_per_1000=0.0)
    # budget 0: needs k > 2 (a 2-window blip must not fire), k=3 is the fastest that does that
    assert cfg.k == 3 and cfg.m == 3
    assert not grid[(grid.k <= 2)].feasible.any()
    with pytest.raises(RuntimeError):
        select_km({"clean": [flags([j] * n)], ("blur", 1): drifted}, setup, ms=(3,), ks=(1, 2), max_fa_per_1000=0.0)
