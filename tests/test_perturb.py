"""Tests for src/perturb.py: determinism, shape/dtype, identity at severity 0, and that each perturbation does
what its parameter says (hand-checkable properties on synthetic images)."""

import numpy as np
import pytest

from src.perturb import LEVELS, all_conditions, apply_perturbation, image_seed, level_value

RNG = np.random.default_rng(123)
TEXTURE = RNG.integers(0, 256, (64, 64, 3), dtype=np.uint8)
GRAY = RNG.integers(0, 256, (64, 64), dtype=np.uint8)
FLAT = np.full((64, 64, 3), 128, np.uint8)
CONDS = all_conditions()
IMAGES = [TEXTURE, GRAY, TEXTURE[:, :, :1]]
IDS = ["bgr", "gray2d", "one_channel"]


def test_levels_match_the_spec():
    assert LEVELS == {"blur": (1.0, 2.0, 3.0), "brightness": (0.6, 0.8, 1.2, 1.4), "contrast": (0.6, 1.4),
                      "noise": (0.02, 0.05, 0.1), "jpeg": (50, 20)}
    assert len(CONDS) == 14


@pytest.mark.parametrize("name", list(LEVELS))
@pytest.mark.parametrize("img", IMAGES, ids=IDS)
def test_identity_at_severity_zero(name, img):
    out = apply_perturbation(img, name, 0, seed=7)
    assert np.array_equal(out, img) and out.dtype == np.uint8 and out is not img


@pytest.mark.parametrize("name,sev", CONDS)
@pytest.mark.parametrize("img", IMAGES, ids=IDS)
def test_shape_and_uint8_dtype_preserved(name, sev, img):
    out = apply_perturbation(img, name, sev, seed=3)
    assert out.shape == img.shape and out.dtype == np.uint8


@pytest.mark.parametrize("name,sev", CONDS)
def test_deterministic_for_a_seed(name, sev):
    assert np.array_equal(apply_perturbation(TEXTURE, name, sev, seed=11), apply_perturbation(TEXTURE, name, sev, seed=11))


@pytest.mark.parametrize("name,sev", CONDS)
def test_does_not_modify_input(name, sev):
    before = TEXTURE.copy()
    apply_perturbation(TEXTURE, name, sev, seed=1)
    assert np.array_equal(TEXTURE, before)


def test_only_noise_depends_on_seed():
    assert not np.array_equal(apply_perturbation(TEXTURE, "noise", 2, 1), apply_perturbation(TEXTURE, "noise", 2, 2))
    for name, sev in CONDS:
        if name != "noise":
            assert np.array_equal(apply_perturbation(TEXTURE, name, sev, 1), apply_perturbation(TEXTURE, name, sev, 2))


def test_every_condition_changes_a_textured_image():
    for name, sev in CONDS:
        assert not np.array_equal(apply_perturbation(TEXTURE, name, sev, 0), TEXTURE), (name, sev)


def test_image_seed_is_stable_and_differs_between_images():
    assert image_seed("crazing_1.jpg") == image_seed("crazing_1.jpg")
    assert image_seed("crazing_1.jpg") != image_seed("crazing_2.jpg")
    assert image_seed("crazing_1.jpg", 0) != image_seed("crazing_1.jpg", 1)


def test_blur_reduces_variance_monotonically_and_keeps_mean():
    stds = [TEXTURE.std()] + [apply_perturbation(TEXTURE, "blur", s).std() for s in (1, 2, 3)]
    assert all(a > b for a, b in zip(stds, stds[1:]))
    assert abs(apply_perturbation(TEXTURE, "blur", 3).mean() - TEXTURE.mean()) < 2


def test_blur_leaves_constant_image_alone_and_spreads_an_impulse():
    assert np.array_equal(apply_perturbation(FLAT, "blur", 3), FLAT)
    imp = np.zeros((41, 41), np.uint8)
    imp[20, 20] = 255
    b = apply_perturbation(imp, "blur", 2)  # sigma 2: peak of a 2-D Gaussian = 255 / (2 pi sigma^2)
    assert b[20, 20] == pytest.approx(255 / (2 * np.pi * 4), abs=1.5)
    assert b[20, 20] > b[20, 23] > b[20, 27] >= 0


def test_brightness_scales_values_and_clips():
    mid = np.full((8, 8), 100, np.uint8)
    assert (apply_perturbation(mid, "brightness", 1) == 60).all()      # factor 0.6
    assert (apply_perturbation(mid, "brightness", 2) == 80).all()      # 0.8
    assert (apply_perturbation(mid, "brightness", 3) == 120).all()     # 1.2
    assert (apply_perturbation(mid, "brightness", 4) == 140).all()     # 1.4
    assert (apply_perturbation(np.full((8, 8), 200, np.uint8), "brightness", 4) == 255).all()  # clipped, no wrap
    means = [apply_perturbation(TEXTURE, "brightness", s).mean() for s in (1, 2, 3, 4)]
    assert means == sorted(means)


def test_contrast_scales_deviation_from_mean():
    img = np.array([[100, 140]], np.uint8)  # mean 120
    assert apply_perturbation(img, "contrast", 1).tolist() == [[108, 132]]   # 0.6: 120 -/+ 12
    assert apply_perturbation(img, "contrast", 2).tolist() == [[92, 148]]    # 1.4: 120 -/+ 28
    assert apply_perturbation(TEXTURE, "contrast", 1).std() < TEXTURE.std()
    assert abs(apply_perturbation(FLAT, "contrast", 2).mean() - 128) < 0.5


@pytest.mark.parametrize("sev,sigma", [(1, 0.02), (2, 0.05), (3, 0.1)])
def test_noise_has_the_requested_sigma(sev, sigma):
    out = apply_perturbation(np.full((200, 200), 128, np.uint8), "noise", sev, seed=0).astype(float)
    assert out.std() == pytest.approx(sigma * 255, rel=0.06)       # mid-grey: clipping is negligible
    assert out.mean() == pytest.approx(128, abs=1.0)


def test_noise_is_clipped_not_wrapped():
    out = apply_perturbation(np.full((100, 100), 255, np.uint8), "noise", 3, seed=0)
    assert out.max() == 255 and 200 < out.mean() < 255


def test_jpeg_error_grows_as_quality_drops():
    err = [np.abs(apply_perturbation(TEXTURE, "jpeg", s).astype(float) - TEXTURE).mean() for s in (1, 2)]
    assert 0 < err[0] < err[1]  # quality 50, then 20


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        apply_perturbation(TEXTURE, "fog", 1)
    for bad in (-1, 4, 1.5):
        with pytest.raises(ValueError):
            apply_perturbation(TEXTURE, "blur", bad)
    with pytest.raises(ValueError):
        apply_perturbation(TEXTURE.astype(np.float32), "blur", 1)
    with pytest.raises(ValueError):
        apply_perturbation(np.zeros((4, 4, 2), np.uint8), "blur", 1)


def test_level_value():
    assert level_value("blur", 2) == 2.0 and level_value("jpeg", 2) == 20 and np.isnan(level_value("noise", 0))
