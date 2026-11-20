"""Tests for the series model, its batches and its loss (05_series_models)."""
import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from pe.data.prepare import IMAGE_LABEL, SERIES_LABELS
from pe.metrics import score
from pe.models.series_model import SeriesModel
from pe.series_training import batch, series_data, series_loss, with_neighbours


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
