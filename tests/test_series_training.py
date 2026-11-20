"""Tests for the series model, its batches and its loss (05_series_models)."""
import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from pe.data.prepare import IMAGE_LABEL, SERIES_LABELS
from pe.metrics import IMAGE_WEIGHT, SERIES_WEIGHTS, score
from pe.models.series_model import SeriesModel
from pe.series_training import SliceFeatures, batch, on_slices, series_data, series_loss, with_neighbours


def small_data(lengths=(5, 3, 7), n_in=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s, n in enumerate(lengths):
        for i in range(n + 1):  # one slice of each series is outside the main stack
            rows.append({"series": f"s{s}", "image": f"s{s}_{i}", "main": i < n, "slice": n - i, IMAGE_LABEL: int(rng.random() < 0.3)})
    table = pd.DataFrame(rows)
    series = pd.DataFrame({"series": [f"s{s}" for s in range(len(lengths))], **{label: rng.integers(0, 2, len(lengths)) for label in SERIES_LABELS}})
    features = rng.normal(size=(int(table["main"].sum()), n_in)).astype(np.float32)
    return table, series, features


def test_series_data_puts_the_main_slices_in_body_order():
    table, series, features = small_data()
    data = series_data(table, series, features)
    first = table.index[(table["series"] == "s0") & table["main"]]
    assert data.rows[0].tolist() == first[::-1].tolist()  # the slice numbers count down in the table
    assert np.array_equal(data.features[0], features[:5][::-1])
    assert data.q[0] == table.loc[table["series"] == "s0", IMAGE_LABEL].mean()


def test_neighbour_differences_keep_the_ends_at_zero():
    x = torch.tensor([[[1.0], [3.0], [6.0]]])
    assert with_neighbours(x)[0].tolist() == [[1, 0, -2], [3, 2, -3], [6, 3, 0]]


def test_a_series_gives_the_same_outputs_alone_and_in_a_padded_batch():
    torch.manual_seed(0)
    table, series, features = small_data()
    data = series_data(table, series, features)
    model = SeriesModel(6, hidden=8).eval()
    together = batch(data, np.arange(3), device="cpu")
    alone = batch(data, np.array([1]), device="cpu")
    slices_together, series_together = model(together["x"], together["lengths"])
    slices_alone, series_alone = model(alone["x"], alone["lengths"])
    assert torch.allclose(slices_together[1, :3], slices_alone[0], atol=1e-6)
    assert torch.allclose(series_together[1], series_alone[0], atol=1e-6)


def test_the_loss_is_the_metric_on_the_main_stack():
    table, series, features = small_data()
    table["main"] = True  # every slice predicted, so that the loss and the metric see the same slices
    features = np.random.default_rng(1).normal(size=(len(table), 2)).astype(np.float32)
    data = series_data(table, series, features)
    b = batch(data, np.arange(3), device="cpu")
    rng = np.random.default_rng(2)
    slice_logits, series_logits = torch.tensor(rng.normal(size=b["y_image"].shape)).float(), torch.tensor(rng.normal(size=(3, 9))).float()
    image_series = np.concatenate([np.full(len(r), k) for k, r in enumerate(data.rows)])
    y_image = np.concatenate(data.y_image)
    p_image = np.concatenate([expit(slice_logits[k, :len(r)].numpy()) for k, r in enumerate(data.rows)])
    expected = score(data.y_series, expit(series_logits.numpy()), image_series, y_image, p_image)
    assert abs(series_loss(slice_logits, series_logits, b).item() - expected) < 1e-5


def test_a_reversed_series_reverses_its_labels():
    table, series, features = small_data()
    data = series_data(table, series, features)
    b = batch(data, np.array([0]), flip=np.array([True]), neighbours=False, device="cpu")
    assert np.array_equal(b["x"][0].numpy(), data.features[0][::-1])
    assert np.array_equal(b["y_image"][0].numpy(), data.y_image[0][::-1])


def test_resampled_series_come_back_to_their_own_slices():
    table, series, features = small_data()
    data = series_data(table, series, features)
    b = batch(data, np.arange(3), length=4, device="cpu")
    assert b["x"].shape == (3, 4, 6) and b["lengths"].tolist() == [4, 4, 4] and b["n_slices"].tolist() == [5, 3, 7]
    assert np.allclose(b["x"][0, [0, -1], :2].numpy(), data.features[0][[0, -1]])  # the ends are kept
    ramp = torch.linspace(0, 1, 4).repeat(3, 1)
    back = on_slices(ramp, b)
    assert back.shape == (3, 7)
    assert torch.allclose(back[2], torch.linspace(0, 1, 7)) and torch.allclose(back[1, :3], torch.tensor([0, 0.5, 1]))


def test_negative_q_lets_the_series_without_pe_train_the_slice_head():
    table, series, features = small_data()
    table[IMAGE_LABEL] = 0
    table.loc[table["series"] == "s0", IMAGE_LABEL] = 1  # only s0 has positive slices
    data = series_data(table, series, features)
    b = batch(data, np.arange(3), device="cpu")
    series_logits, low, high = torch.zeros(3, 9), torch.zeros(3, 7), torch.zeros(3, 7)
    high[1:] = 5  # other slice logits in the series without PE
    assert series_loss(low, series_logits, b).item() == series_loss(high, series_logits, b).item()
    assert series_loss(low, series_logits, b, negative_q=0.2).item() < series_loss(high, series_logits, b, negative_q=0.2).item()


def test_slice_features_put_the_backbones_side_by_side(tmp_path):
    index = pd.DataFrame({"series": ["a", "a", "b"], "image": ["1", "2", "3"]})
    index.to_csv(tmp_path / "index.csv", index=False)
    for name, width in (("x", 2), ("y", 1)):
        (tmp_path / name).mkdir()
        np.save(tmp_path / name / "f0.npy", np.arange(3 * width, dtype=np.float16).reshape(3, width) + (100 if name == "y" else 0))
        np.save(tmp_path / name / "logits_f0.npy", np.array([-1, -2, -3], np.float32) * (10 if name == "y" else 1))
    both = SliceFeatures(tmp_path, 0, ["x", "y"], "both")
    assert both.dtype == np.float16 and both[[2, 0]].tolist() == [[4, 5, -3, 102, -30], [0, 1, -1, 100, -10]]
    logits = SliceFeatures(tmp_path, 0, ["x", "y"], "logit")
    assert logits.dtype == np.float32 and logits[[1]].tolist() == [[-2, -20]]
    table = pd.DataFrame({"series": ["b", "a", "a"], "image": ["3", "9", "1"], "main": [True, False, True]})
    assert SliceFeatures(tmp_path, 0, ["x"], "embedding").rows_of(table).tolist() == [2, 0]


def test_a_fixed_denominator_adds_the_batches_up_to_the_metric():
    table, series, features = small_data()
    data = series_data(table, series, features)
    rng = np.random.default_rng(3)
    slice_logits, series_logits = torch.tensor(rng.normal(size=(3, 7))).float(), torch.tensor(rng.normal(size=(3, 9))).float()
    whole = batch(data, np.arange(3), device="cpu")
    weight = (whole["q"] * whole["n_slices"]).mean().item() * IMAGE_WEIGHT + SERIES_WEIGHTS.sum()
    parts = [series_loss(slice_logits[i:i + 1, :n], series_logits[i:i + 1], batch(data, np.array([i]), device="cpu"), weight_per_series=weight)
             for i, n in enumerate(whole["n_slices"].tolist())]
    assert abs(sum(p.item() for p in parts) / 3 - series_loss(slice_logits, series_logits, whole).item()) < 1e-5
