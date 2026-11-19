"""Tests for the slice dataset, the slice model, the GPU steps of the training and the validation scores (04_slice_models)."""
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit

from pe.data.dataset import SliceDataset, neighbour_images, positive_slice_targets, q_weights, screening_series, series_boxes
from pe.data.prepare import IMAGE_LABEL, SERIES_LABELS, STORE, store_path
from pe.data.windows import WINDOWS
from pe.metrics import score
from pe.models.slice_model import SliceModel
from pe.training import augment, learning_rate, slice_loss, to_window, to_windows
from pe.validation import OTHER_STACK, correct_level, fit_level, slice_scores, validation_fold


def test_neighbours_stay_inside_the_stack():
    table = pd.DataFrame({"series": ["s", "s", "s", "s", "t"], "stack": [0, 0, 0, 1, 0], "image": list("abcde")})
    assert neighbour_images(table).tolist() == [["a", "a", "b"], ["a", "b", "c"], ["b", "c", "c"], ["d", "d", "d"], ["e", "e", "e"]]


def test_far_neighbours_stop_at_the_end_of_the_stack():
    table = pd.DataFrame({"series": ["s", "s", "s", "s", "t"], "stack": [0, 0, 0, 1, 0], "image": list("abcde")})
    assert neighbour_images(table, [-2, 0, 2]).tolist() == [["a", "a", "c"], ["a", "b", "c"], ["a", "c", "c"], ["d", "d", "d"], ["e", "e", "e"]]
    assert neighbour_images(table, [0]).tolist() == [["a"], ["b"], ["c"], ["d"], ["e"]]


def test_a_series_box_is_the_union_of_its_slices_with_a_margin_in_store_pixels():
    table = pd.DataFrame({"series": ["s", "s", "s", "t"], "lung_top": [100, 120, -1, 0], "lung_bottom": [300, 400, -1, 512],
                          "lung_left": [50, 40, -1, 0], "lung_right": [240, 200, -1, 512]})
    boxes = series_boxes(table, "lung", 256, margin=0.1)
    assert boxes[:3].tolist() == [[35, 215, 10, 130]] * 3  # 100-400 and 40-240 with 30 and 20 px around them at 512 px, then at half size
    assert boxes[3].tolist() == [0, 256, 0, 256]


def test_the_dataset_cuts_to_the_box_and_keeps_the_size(tmp_path):
    (tmp_path / "s").mkdir()
    image = np.zeros((8, 8), dtype=np.uint16)
    image[2:6, 4:] = 100
    store_path(tmp_path, "s", "a").write_bytes(STORE.encode(image))
    stored, _ = SliceDataset(tmp_path, np.array(["s"]), np.array([["a"]]), np.array([0.0], dtype=np.float32), np.array([[2, 6, 4, 8]]))[0]
    assert stored.shape == (1, 8, 8) and (stored == 100).all()


def screening_table(n=400):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"series": [f"s{i}" for i in range(n)], "group": rng.choice(["a", "b"], n, p=[0.8, 0.2]),
                         "negative_exam_for_pe": rng.integers(0, 2, n), "indeterminate": 0})


def test_a_smaller_share_lies_inside_a_larger_one():
    table = screening_table()
    half, quarter = screening_series(table, 0.5), screening_series(table, 0.25)
    assert len(half) == pytest.approx(200, abs=2) and len(quarter) == pytest.approx(100, abs=2)
    assert set(quarter) <= set(half)


def test_a_share_keeps_the_mix_of_groups_and_positives():
    table = screening_table()
    kept = table[table["series"].isin(screening_series(table, 0.25))]
    assert (kept["group"] == "b").mean() == pytest.approx((table["group"] == "b").mean(), abs=0.02)
    assert kept["negative_exam_for_pe"].mean() == pytest.approx(table["negative_exam_for_pe"].mean(), abs=0.02)


def test_the_dataset_returns_the_slices_in_the_order_asked(tmp_path):
    (tmp_path / "s").mkdir()
    for value, image in enumerate("abc"):
        store_path(tmp_path, "s", image).write_bytes(STORE.encode(np.full((4, 4), value, dtype=np.uint16)))
    stored, label = SliceDataset(tmp_path, np.array(["s"]), np.array([["c", "a", "b"]]), np.array([1.0], dtype=np.float32))[0]
    assert stored.dtype == np.int16 and stored[:, 0, 0].tolist() == [2, 0, 1] and label == 1


def test_the_window_maps_its_two_ends_to_0_and_1():
    level, width = 100, 700
    stored = torch.tensor([level - width / 2 + 1024, level + 1024, level + width / 2 + 1024, 0, 4095])
    assert to_window(stored, level, width).tolist() == pytest.approx([0, 0.5, 1, 0, 1])


def test_every_channel_goes_through_every_window_and_the_corners_get_air():
    windows = [WINDOWS["pe"], WINDOWS["lung"]]
    x = to_windows(torch.zeros(4, 3, 16, 16), windows)
    assert x.shape == (4, 6, 16, 16)
    air = to_windows(torch.zeros(1, 3, 1, 1), windows)
    assert air[0, :3].flatten().tolist() == pytest.approx([0] * 3) and air[0, 3:].flatten().tolist() == pytest.approx([326 / 1500] * 3)
    assert augment(x, air).sub(air).abs().max() < 1e-5  # an image of air stays air, corners included


def test_augment_does_not_flip_and_fills_with_air():
    x = torch.zeros(64, 1, 64, 64)
    x[:, :, 28:36, :16] = 1  # a bar on the left
    out = augment(x)
    assert out.shape == x.shape and out.min() >= 0
    assert out[:, 0, :, :32].sum() > 10 * out[:, 0, :, 32:].sum()
    assert out[:, 0, 0, 0].max() == 0


def test_the_learning_rate_rises_then_falls_to_0():
    rates = [learning_rate(step, 100) for step in range(100)]
    assert rates[0] == pytest.approx(0.2) and rates[4] == 1 and rates[5] == 1
    assert rates[50] < 1 and rates[99] < 0.01
    assert all(a >= b for a, b in zip(rates[5:], rates[6:]))


def test_the_slice_model_gives_one_logit_per_slice_and_its_features():
    model = SliceModel("efficientnet_b0", pretrained=False).eval()
    logits, features = model(torch.rand(2, 3, 64, 64))
    assert logits.shape == (2,) and features.shape == (2, 1280)
    assert model.mean.item() == pytest.approx(0.449, abs=0.001)


def test_the_q_weights_follow_the_share_of_positive_slices_and_average_1():
    table = pd.DataFrame({"series": list("aaaabb"), IMAGE_LABEL: [1, 1, 1, 0, 0, 0]})
    assert q_weights(table).tolist() == pytest.approx([1.5, 1.5, 1.5, 1.5, 0, 0])


def test_the_series_targets_stay_only_on_positive_slices():
    table = pd.DataFrame({"series": list("aab"), IMAGE_LABEL: [1, 0, 1]})
    series = pd.DataFrame({"series": ["b", "a"], "central_pe": [0, 1], "chronic_pe": [1, 1]})
    assert positive_slice_targets(table, series, ["central_pe", "chronic_pe"]).tolist() == [[1, 1], [0, 0], [0, 1]]


def test_the_loss_is_plain_cross_entropy_by_default_and_focal_lowers_easy_items_most():
    logits, y = torch.tensor([3.0, -0.5]), torch.tensor([1.0, 1.0])
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    assert slice_loss(logits, y, torch.ones(2)).item() == pytest.approx(plain.item())
    assert slice_loss(logits, y, torch.tensor([0.0, 2.0])).item() == pytest.approx(torch.nn.functional.softplus(torch.tensor(0.5)).item())
    easy, hard = (slice_loss(logits[[i]], y[[i]], torch.ones(1), gamma=2.0) / slice_loss(logits[[i]], y[[i]], torch.ones(1)) for i in (0, 1))
    assert easy < 0.01 and hard > 0.1


def test_the_series_head_exists_only_when_asked():
    assert not hasattr(SliceModel("efficientnet_b0", pretrained=False), "series_head")
    assert SliceModel("efficientnet_b0", pretrained=False, series_targets=7).series_head.out_features == 7


def test_the_fitted_level_undoes_a_shift_of_the_logits():
    rng = np.random.default_rng(0)
    true = rng.normal(-1, 2, 50_000)
    y = (rng.random(len(true)) < expit(true)).astype(float)
    assert fit_level(true - 1.5, y, np.ones(len(y))) == pytest.approx(1.5, abs=0.05)


def test_the_correction_of_a_group_does_not_use_that_group():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    groups = np.array([0, 0, 1, 1])
    logits = np.array([0.0, 0.0, 0.0, 0.0])
    # all logits are equal, so the fit in each group alone gives its own rate; changing one group's labels moves the other's correction only
    base, _ = correct_level(logits, y, np.ones(4), groups)
    other = y.copy()
    other[3] = 1.0
    changed, _ = correct_level(logits, other, np.ones(4), groups)
    moved = base != changed
    assert moved.sum() == 2 and (groups[moved] == groups[moved][0]).all()
    assert (y[groups == groups[moved][0]] == other[groups == groups[moved][0]]).all()


def small_fold():
    labels = {label: [0, 0, 0] for label in SERIES_LABELS}
    labels["negative_exam_for_pe"] = [0, 1, 0]
    series = pd.DataFrame({"series": ["p", "n", "w", "o"], "split": "train", "fold": [0, 0, 0, 1], "writer": ["x", "x", "dcm4che-1.4.35", "x"],
                           **{label: values + [1 if label == "central_pe" else 0] for label, values in labels.items()}})
    slices = pd.DataFrame({"series": list("pppp") + list("nn") + list("www") + ["o"], "image": list("abcdefghij"), "main": [True] * 3 + [False] + [True] * 6,
                           IMAGE_LABEL: [1, 1, 0, 0, 0, 0, 1, 0, 0, 0]})
    return slices, series


def test_slices_outside_the_main_stack_get_the_fixed_prediction_and_the_rates_come_from_the_other_folds():
    slices, series = small_fold()
    fold = validation_fold(slices, series, 0)
    logits = np.array([2.0, 1.0, -1.0, -3.0, -3.0, 1.5, -1.0, -2.0])
    assert fold.p_series[0].tolist() == series.loc[3, SERIES_LABELS].astype(float).tolist()
    scores = slice_scores(fold, logits)
    p_image = np.full(9, OTHER_STACK)
    p_image[fold.main] = expit(logits)
    assert scores["full"] == pytest.approx(score(fold.y_series, fold.p_series, fold.image_series, fold.y_image, p_image))


def test_the_test_like_score_leaves_out_the_rare_writers():
    slices, series = small_fold()
    fold = validation_fold(slices, series, 0)
    assert fold.test_like.tolist() == [True, True, False]
    logits = np.array([2.0, 1.0, -1.0, -3.0, -3.0, 1.5, -1.0, -2.0])
    scores = slice_scores(fold, logits)
    keep = np.isin(fold.image_series, [0, 1])
    p_image = np.full(9, OTHER_STACK)
    p_image[fold.main] = expit(logits)
    assert scores["test-like"] == pytest.approx(score(fold.y_series[:2], fold.p_series[:2], fold.image_series[keep], fold.y_image[keep], p_image[keep]))
    assert 0 <= scores["AUROC"] <= 1
