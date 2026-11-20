"""Tests for the label consistency rules and their fix (05_series_models)."""
import numpy as np

from pe.consistency import HIGH, LOW, make_consistent, violations
from pe.data.prepare import SERIES_LABELS

COLUMN = {label: i for i, label in enumerate(SERIES_LABELS)}


def series_row(**on):
    """Nine predictions of 0.1, with the named labels at the given values."""
    row = np.full(len(SERIES_LABELS), 0.1)
    for label, value in on.items():
        row[COLUMN[label]] = value
    return row


def test_the_rules_follow_the_host():
    p_series = np.stack([
        series_row(central_pe=0.9, rv_lv_ratio_lt_1=0.9),                       # positive, fine
        series_row(central_pe=0.9),                                             # positive without an RV/LV label
        series_row(central_pe=0.9, rv_lv_ratio_lt_1=0.9, rv_lv_ratio_gte_1=0.9),  # positive with both RV/LV labels
        series_row(rv_lv_ratio_lt_1=0.9),                                       # positive without a location
        series_row(central_pe=0.9, rv_lv_ratio_lt_1=0.9, chronic_pe=0.9, acute_and_chronic_pe=0.9),  # both chronic labels
        series_row(central_pe=0.9, rv_lv_ratio_lt_1=0.9, negative_exam_for_pe=0.9),  # positive and negative
        series_row(negative_exam_for_pe=0.9),                                   # negative, fine
        series_row(),                                                           # negative without negative or indeterminate
        series_row(negative_exam_for_pe=0.9, indeterminate=0.9),                # negative with both
        series_row(negative_exam_for_pe=0.9, leftsided_pe=0.9),                 # negative with a location
    ])
    positive = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    image_series = np.repeat(np.arange(len(p_series)), 2)
    p_image = np.where(positive[image_series] == 1, [0.9, 0.2] * len(p_series), 0.2)
    assert violations(p_series, image_series, p_image).tolist() == [False, True, True, True, True, True, False, True, True, True]


def test_the_fix_leaves_no_violation_and_keeps_consistent_series():
    rng = np.random.default_rng(0)
    consistent = np.stack([series_row(negative_exam_for_pe=0.7), series_row(central_pe=0.8, rv_lv_ratio_lt_1=0.6)])
    p_series = np.vstack([rng.uniform(size=(500, 9)), consistent])
    image_series = np.repeat(np.arange(502), 20)
    p_image = rng.uniform(size=len(image_series)) ** 3
    p_image[-40:-20] = np.minimum(p_image[-40:-20], 0.4)  # the negative one
    p_image[-1] = 0.8  # the positive one
    fixed_series, fixed_image = make_consistent(p_series, image_series, p_image)
    assert not violations(fixed_series, image_series, fixed_image).any()
    good = ~violations(p_series, image_series, p_image)
    assert good.any()
    assert np.array_equal(fixed_series[good], p_series[good])
    assert np.array_equal(fixed_image[good[image_series]], p_image[good[image_series]])


def test_inside_a_group_the_cheaper_label_by_weight_stays():
    # rv_lv_ratio_gte_1 weighs three times rv_lv_ratio_lt_1: lowering lt_1 from 0.9 costs less than gte_1 from 0.8
    p_series = np.stack([series_row(central_pe=0.9, rv_lv_ratio_lt_1=0.9, rv_lv_ratio_gte_1=0.8)])
    fixed_series, fixed_image = make_consistent(p_series, np.zeros(2, int), np.array([0.9, 0.1]))
    assert fixed_series[0, COLUMN["rv_lv_ratio_lt_1"]] == LOW
    assert fixed_series[0, COLUMN["rv_lv_ratio_gte_1"]] == 0.8
    assert fixed_image.tolist() == [0.9, 0.1]
    # No location above 0.5: leftsided_pe (weight 0.063) is raised from 0.43 rather than central_pe (0.188) from 0.45
    p_series = np.stack([series_row(central_pe=0.45, leftsided_pe=0.43, rv_lv_ratio_lt_1=0.9)])
    fixed_series, _ = make_consistent(p_series, np.zeros(2, int), np.array([0.9, 0.1]))
    assert fixed_series[0, COLUMN["leftsided_pe"]] == HIGH
    assert fixed_series[0, COLUMN["central_pe"]] == 0.45


def test_only_slices_that_may_be_raised_are_raised():
    # A clear positive whose predicted slices are all below the fixed prediction of its slice outside the main stack
    p_series = np.stack([series_row(central_pe=0.95, rv_lv_ratio_gte_1=0.9, negative_exam_for_pe=0.01, indeterminate=0.01)])
    p_image = np.array([0.0005, 0.0008, 0.001])
    _, fixed_image = make_consistent(p_series, np.zeros(3, int), p_image, can_raise=np.array([True, True, False]))
    assert fixed_image.tolist() == [0.0005, HIGH, 0.001]


def test_the_fix_picks_the_cheaper_side():
    # One slice just above 0.5 but a confident negative label: lowering the slice costs less than lowering the label
    p_series = np.stack([series_row(negative_exam_for_pe=0.95)])
    fixed_series, fixed_image = make_consistent(p_series, np.zeros(3, int), np.array([0.51, 0.1, 0.1]))
    assert fixed_series[0, COLUMN["negative_exam_for_pe"]] == 0.95
    assert fixed_image.tolist() == [LOW, 0.1, 0.1]
    # A clear positive whose slices all stay below 0.5: the highest slice is raised
    p_series = np.stack([series_row(central_pe=0.95, rightsided_pe=0.9, rv_lv_ratio_gte_1=0.9, negative_exam_for_pe=0.05, indeterminate=0.01)])
    fixed_series, fixed_image = make_consistent(p_series, np.zeros(3, int), np.array([0.45, 0.3, 0.1]))
    assert fixed_image.tolist() == [HIGH, 0.3, 0.1]
    assert np.array_equal(fixed_series, p_series)
