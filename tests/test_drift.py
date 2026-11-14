"""Drift-detection tests on synthetic data. Evidently-dependent tests are skipped when it is not installed."""

import numpy as np
import pandas as pd
import pytest

from src.drift import (FEATURES, DriftConfig, alert_from_flags, alert_series, detection_delay, episodes,
                       flag_features, holm_reject, make_stream, reference_window, select_config, sliding_pvalues,
                       window_ends)
from src.perturb import LEVELS
from src.simulator import DriftSchedule


def frame(n, shift=None, seed=0, start=0):
    """n synthetic events; features in `shift` get +shift sigma from event `start`."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    for f, d in (shift or {}).items():
        df.loc[start:, f] += d
    return df


def test_reference_window_is_the_first_events():
    df = frame(30)
    assert reference_window(df, 10).equals(df.iloc[:10])
    with pytest.raises(ValueError):
        reference_window(df, 31)


def test_windows_never_overlap_the_reference():
    cfg = DriftConfig(window=20, ref_size=50)
    ends = window_ends(100, cfg)
    assert ends[0] == 69 and ends[-1] == 99 and len(ends) == 31
    p = sliding_pvalues(frame(100), cfg)
    assert list(p.index) == list(ends) and list(p.columns) == FEATURES


def test_sliding_window_contains_exactly_the_last_events():
    cfg = DriftConfig(window=10, ref_size=20)
    df = frame(60)
    df.loc[40:, "sharpness"] += 100  # unmistakable: only windows ending at >= 40 see it
    p = sliding_pvalues(df, cfg)["sharpness"]
    assert p.loc[39] > 1e-3 and p.loc[49] < 1e-3  # window 40..49 is fully shifted
    assert p.loc[41] > p.loc[49]  # one shifted event in the window is much weaker evidence than ten


def test_fast_ks_matches_scipy():
    from scipy.stats import ks_2samp

    from src.drift import ks_pvalues

    rng = np.random.default_rng(0)
    ref = pd.DataFrame({f: rng.normal(size=200) for f in FEATURES})
    for shift in (0.0, 0.3, 1.0):
        cur = pd.DataFrame({f: rng.normal(shift, 1, size=50) for f in FEATURES})
        exact = np.array([ks_2samp(ref[f], cur[f]).pvalue for f in FEATURES])
        got = ks_pvalues(ref, cur)
        assert np.allclose(np.log10(got), np.log10(exact), atol=0.35)  # same order of magnitude


def test_known_shift_is_flagged_and_clean_is_not():
    cfg = DriftConfig(window=50, ref_size=100, alpha=0.01, min_flagged=2)
    shifted = frame(300, shift={"brightness_mean": 2.0, "sharpness": 2.0}, start=200, seed=1)
    _, alert = alert_series(sliding_pvalues(shifted, cfg), cfg)
    assert not alert.loc[:199].any() and alert.loc[260:].all()
    delay = detection_delay(alert, 200)
    assert delay is not None and 0 <= delay < 50
    clean_alerts = [alert_series(sliding_pvalues(frame(400, seed=s), cfg), cfg)[1] for s in range(5)]
    assert sum(a.mean() for a in clean_alerts) / 5 < 0.02


def test_one_feature_does_not_trigger_a_two_feature_rule():
    cfg = DriftConfig(window=50, ref_size=100, alpha=0.01, min_flagged=2)
    flags, alert = alert_series(sliding_pvalues(frame(300, shift={"noise_est": 3.0}, start=200, seed=2), cfg), cfg)
    assert flags.loc[260:, "noise_est"].all() and flags.loc[260:].drop(columns="noise_est").sum().sum() == 0
    assert not alert.any()
    assert alert_series(sliding_pvalues(frame(300, shift={"noise_est": 3.0}, start=200, seed=2), cfg),
                        DriftConfig(window=50, ref_size=100, min_flagged=1))[1].loc[260:].all()


def test_holm_controls_the_number_of_false_flags():
    p = np.array([[0.004, 0.02, 0.5, 0.9, 0.8, 0.7],   # 0.004 < 0.05/6 -> rejected; 0.02 > 0.05/5 -> stop
                  [0.04, 0.5, 0.5, 0.5, 0.5, 0.5],     # raw would flag it; Holm does not
                  [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]])
    assert holm_reject(p, 0.05).tolist() == [[True, False, False, False, False, False],
                                             [False] * 6, [True] * 6]
    assert flag_features(p, 0.05, "none").sum(axis=1).tolist() == [2, 1, 6]
    with pytest.raises(ValueError):
        flag_features(p, 0.05, "bogus")


def test_uncorrected_testing_inflates_false_flags_holm_does_not():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=(20000, 6))  # all nulls, independent
    raw = (p < 0.05).any(axis=1).mean()
    holm = holm_reject(p, 0.05).any(axis=1).mean()
    assert raw > 0.2 and holm < 0.06


def test_alert_rule_and_episodes():
    flags = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=bool)
    assert alert_from_flags(flags, 2).tolist() == [False, True, True, False]
    assert alert_from_flags(flags, 3).tolist() == [False, False, True, False]
    a = pd.Series([False, True, True, False, True, False])
    assert episodes(a) == 2


def test_detection_delay_cases():
    idx = pd.RangeIndex(10, 30)
    never = pd.Series(False, index=idx)
    assert detection_delay(never, 20) is None
    a = never.copy()
    a.loc[25:] = True
    assert detection_delay(a, 20) == 5
    a.loc[:20] = True  # alarm already on before the drift: not credited as an instant detection
    assert detection_delay(a, 20) is None
    a2 = never.copy()
    a2.loc[20] = True
    assert detection_delay(a2, 20) == 0


def pool():
    rows = []
    for kind, sev in [("none", 0)] + [(k, s) for k in ("blur", "brightness") for s in range(1, len(LEVELS[k]) + 1)]:
        for i in range(30):
            rows.append({"image_id": f"i{i:02d}.png", "perturbation": kind, "severity": sev,
                         **{f: float(i + 100 * sev) for f in FEATURES}})
    return pd.DataFrame(rows)


def test_make_stream_is_deterministic_and_switches_at_drift_start():
    p = pool()
    a = make_stream(p, 40, seed=3, drift=("blur", 2, 25))
    b = make_stream(p, 40, seed=3, drift=("blur", 2, 25))
    assert a.equals(b) and not a.equals(make_stream(p, 40, seed=4, drift=("blur", 2, 25)))
    assert (a.perturbation[:25] == "none").all() and (a.perturbation[25:] == "blur").all()
    assert (a.severity[25:] == 2).all() and a.event_index.tolist() == list(range(40))
    assert (make_stream(p, 10, seed=1).perturbation == "none").all()
    # same images are drawn with and without drift, so only the perturbation differs
    clean = make_stream(p, 40, seed=3)
    assert (a.image_id == clean.image_id).all()


def test_select_config_respects_the_false_alarm_budget():
    from src.drift import EvalSetup, _pvals_cache

    setup = EvalSetup(drift_start=150, horizon=60, clean_events=250, n_clean=2, n_drift_reps=2)
    p = pool()
    # a synthetic pool where blur shifts every feature strongly
    p.loc[p.perturbation == "blur", FEATURES] += 50
    cache = _pvals_cache(p, 5, [20], setup, ref_size=100, kinds=("blur", "brightness"))
    cfg, grid = select_config(cache, setup, [20], [0.01], [1, 2], max_fa_per_1000=1000.0)
    assert grid.feasible.all() and cfg.window == 20
    with pytest.raises(RuntimeError):
        # an impossible budget (negative) leaves nothing feasible
        select_config(cache, setup, [20], [0.01], [1], max_fa_per_1000=-1.0)


def test_brightness_is_opt_in_for_the_simulator_schedule():
    default = DriftSchedule(start=0)
    assert "brightness" not in default.kinds and "brightness" in LEVELS
    rng_a, rng_b = np.random.default_rng([0, 1]), np.random.default_rng([0, 1])
    assert [default.at(i, rng_a) for i in range(30)] == [DriftSchedule(start=0).at(i, rng_b) for i in range(30)]
    with_b = DriftSchedule(start=0, kinds=default.kinds + ("brightness",))
    seen = {with_b.at(i, np.random.default_rng([0, i]))[0] for i in range(60)}
    assert "brightness" in seen


def _has_evidently():
    try:
        from evidently import Report  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_evidently(), reason="requirements-monitoring.txt not installed")
def test_evidently_report_writes_files_and_agrees_with_ks(tmp_path):
    from src.drift import evidently_pvalues, evidently_report, ks_pvalues

    cfg = DriftConfig(window=60, ref_size=100, alpha=0.01)
    df = frame(160, shift={"brightness_mean": 3.0, "sharpness": 3.0}, start=100, seed=3)
    ref, cur = df.iloc[:100], df.iloc[-60:]
    rep = evidently_report(ref, cur, str(tmp_path / "r.html"), str(tmp_path / "r.json"), cfg)
    assert (tmp_path / "r.html").stat().st_size > 1000 and (tmp_path / "r.json").exists()
    theirs = evidently_pvalues(rep)
    mine = dict(zip(FEATURES, ks_pvalues(ref, cur)))
    assert set(theirs) == set(FEATURES)
    assert {f for f in FEATURES if theirs[f] < cfg.alpha} == {f for f in FEATURES if mine[f] < cfg.alpha}         == {"brightness_mean", "sharpness"}
