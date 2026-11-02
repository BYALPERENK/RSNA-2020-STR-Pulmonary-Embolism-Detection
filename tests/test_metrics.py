"""Tests for the competition metric and the series bootstrap in src/pe/metrics.py."""
import numpy as np
import pytest
from sklearn.metrics import log_loss

from pe.data.prepare import IMAGE_LABEL, SERIES_LABELS
from pe.metrics import IMAGE_WEIGHT, SERIES_WEIGHTS, bootstrap, image_weights, paired_bootstrap, score, score_by_label, series_losses


def random_data(n_series=40, seed=0):
    # y_series, p_series, image_series, y_image, p_image; half of the series have positive slices
    rng = np.random.default_rng(seed)
    y_series = (rng.random((n_series, 9)) < 0.3).astype(int)
    image_series = np.repeat(np.arange(n_series), rng.integers(20, 60, n_series))
    y_image = ((rng.random(len(image_series)) < 0.3) & (image_series % 2 == 0)).astype(int)
    return y_series, rng.random((n_series, 9)), image_series, y_image, rng.random(len(image_series))


def test_a_perfect_prediction_scores_zero():
    y_series, _, image_series, y_image, _ = random_data()
    assert score(y_series, y_series, image_series, y_image, y_image) == pytest.approx(0, abs=1e-12)


def test_a_constant_half_scores_log_2():
    y_series, _, image_series, y_image, _ = random_data()
    half = score(y_series, np.full(y_series.shape, 0.5), image_series, y_image, np.full(y_image.shape, 0.5))
    assert half == pytest.approx(np.log(2))


def test_the_series_weights_add_up_to_one_and_the_largest_is_rv_lv():
    assert SERIES_WEIGHTS.sum() == pytest.approx(1)
    assert SERIES_LABELS[SERIES_WEIGHTS.argmax()] == "rv_lv_ratio_gte_1"
    assert SERIES_WEIGHTS[SERIES_LABELS.index("rv_lv_ratio_gte_1")] == pytest.approx(0.2346625767)
    assert IMAGE_WEIGHT == pytest.approx(0.07361963)


def test_the_score_matches_sklearn_with_the_weights_of_the_formula():
    y_series, p_series, image_series, y_image, p_image = random_data()
    series_part = sum(w * len(y_series) * log_loss(y_series[:, j], p_series[:, j], labels=[0, 1]) for j, w in enumerate(SERIES_WEIGHTS))
    q = np.bincount(image_series, weights=y_image) / np.bincount(image_series)
    weights = IMAGE_WEIGHT * q[image_series]
    image_part = weights.sum() * log_loss(y_image, p_image, sample_weight=weights, labels=[0, 1])
    denominator = len(y_series) + IMAGE_WEIGHT * y_image.sum()
    assert score(y_series, p_series, image_series, y_image, p_image) == pytest.approx((series_part + image_part) / denominator)


def test_a_small_case_by_hand():
    # one series with 1 positive slice of 2 (q = 0.5), every prediction 0.5 except the positive slice at 0.25
    y_series, p_series = np.zeros((1, 9)), np.full((1, 9), 0.5)
    expected = (np.log(2) + IMAGE_WEIGHT * 0.5 * (np.log(4) + np.log(2))) / (1 + IMAGE_WEIGHT)
    assert score(y_series, p_series, np.array([0, 0]), np.array([1, 0]), np.array([0.25, 0.5])) == pytest.approx(expected)


def test_slices_of_a_series_without_positive_slices_do_not_count():
    y_series, p_series, image_series, y_image, p_image = random_data()
    changed = p_image.copy()
    changed[image_series % 2 == 1] = 0.999
    assert score(y_series, p_series, image_series, y_image, changed) == pytest.approx(score(y_series, p_series, image_series, y_image, p_image))


def test_q_divides_by_all_slices_of_the_series():
    # a stack left out of the table would raise q from 1/4 to 1/2
    assert image_weights(np.zeros(4, int), [1, 0, 0, 0]) == pytest.approx(IMAGE_WEIGHT / 4)
    y_series, p_series = np.zeros((1, 9)), np.full((1, 9), 0.5)
    all_slices = score(y_series, p_series, np.zeros(4, int), np.array([1, 0, 0, 0]), np.array([0.1, 0.5, 1e-9, 1e-9]))
    main_only = score(y_series, p_series, np.zeros(2, int), np.array([1, 0]), np.array([0.1, 0.5]))
    assert all_slices != pytest.approx(main_only)


def test_the_denominator_counts_positive_slices_not_rows():
    # averaging over all rows would divide by 9 N + slices; the weights give N + 0.0736 x positive slices
    y_series, p_series, image_series, y_image, p_image = random_data()
    _, weights = series_losses(y_series, p_series, image_series, y_image, p_image)
    assert weights.sum() == pytest.approx(len(y_series) + IMAGE_WEIGHT * y_image.sum())


def test_predictions_are_clipped_to_1e_15():
    y_series, p_series = np.zeros((1, 9)), np.zeros((1, 9))
    result = score(y_series, p_series, np.zeros(1, int), np.ones(1), np.zeros(1))
    assert result == pytest.approx(IMAGE_WEIGHT * -np.log(1e-15) / (1 + IMAGE_WEIGHT))


def test_the_shares_of_the_labels_add_up_to_the_score():
    data = random_data()
    parts = score_by_label(*data)
    assert list(parts) == SERIES_LABELS + [IMAGE_LABEL]
    assert sum(part["share of the score"] for part in parts.values()) == pytest.approx(score(*data))


def test_series_without_positive_slices_have_no_slice_loss():
    y_series, p_series, image_series, y_image, p_image = random_data()
    part = score_by_label(y_series, p_series, image_series, np.zeros_like(y_image), p_image)[IMAGE_LABEL]
    assert np.isnan(part["log loss"])
    assert part["share of the score"] == 0


def test_the_score_is_the_ratio_of_the_series_losses():
    data = random_data()
    losses, weights = series_losses(*data)
    assert losses.shape == weights.shape == (40,)
    assert losses.sum() / weights.sum() == pytest.approx(score(*data))


def test_bootstrap_resamples_whole_groups():
    # two groups with constant losses 0 and 1: every resample is a mix of whole groups
    losses = np.array([0.0] * 10 + [1.0] * 10)
    groups = np.array(["a"] * 10 + ["b"] * 10)
    result = bootstrap(losses, np.ones(20), groups, n_boot=500, seed=1)
    assert result["mean"] == pytest.approx(0.5)
    assert {result["low"], result["high"]} <= {0.0, 0.5, 1.0}


def test_bootstrap_divides_by_the_weights():
    assert bootstrap(np.array([2.0, 4.0]), np.array([1.0, 3.0]), n_boot=10)["mean"] == pytest.approx(1.5)


def test_paired_bootstrap_of_a_constant_difference_has_no_width():
    rng = np.random.default_rng(0)
    a = rng.random(100)
    result = paired_bootstrap(a, a + 0.01, np.ones(100), rng.integers(0, 20, 100), n_boot=200)
    assert result["mean"] == pytest.approx(0.01)
    assert result["high"] - result["low"] == pytest.approx(0, abs=1e-12)
