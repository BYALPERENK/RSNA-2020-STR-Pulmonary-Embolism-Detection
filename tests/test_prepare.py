"""Tests for the slice table helpers and the store writer in src/pe/data/prepare.py."""
import numpy as np
import pandas as pd
import pytest

from pe.data.dicom import HU_MIN, read_hu
from pe.data.prepare import STORE, add_stacks, iterative_folds, pixel_stats, stack_ids, write_slice
from test_dicom import write_dicom


def test_a_new_kind_of_slice_starts_a_new_stack():
    keys = ["thin"] * 3 + ["thick"] * 2
    assert stack_ids(keys, [0, 1.25, 2.5, -100, -95]).tolist() == [0, 0, 0, 1, 1]


def test_a_second_pass_over_the_same_positions_is_a_new_stack():
    assert stack_ids(["a"] * 6, [0, 1, 2, 0, 1, 2]).tolist() == [0, 0, 0, 1, 1, 1]
    assert stack_ids(["a"] * 6, [2, 1, 0, 2, 1, 0]).tolist() == [0, 0, 0, 1, 1, 1]


def test_a_gap_in_the_same_direction_stays_in_the_stack():
    assert stack_ids(["a"] * 4, [0, 1, 20, 21]).tolist() == [0, 0, 0, 0]


def test_slices_at_the_same_position_stay_in_the_stack():
    assert stack_ids(["a"] * 4, [0, 1, 1, 2]).tolist() == [0, 0, 0, 0]


def test_add_stacks_orders_each_stack_and_marks_the_main_one():
    slices = pd.DataFrame({"series": "s", "image": list("abcdef"), "instance": range(1, 7), "tilt": [87, 0, 0, 0, 0, 0],
                           "stack_key": ["other plane"] + ["thin"] * 3 + ["thick"] * 2, "pos": [5, 2.5, 1.25, 0, -100, -95]})
    out = add_stacks(slices).set_index("image")
    assert out["stack"].to_dict() == {"a": 0, "b": 1, "c": 1, "d": 1, "e": 2, "f": 2}
    assert out["main"].to_dict() == {"a": False, "b": True, "c": True, "d": True, "e": False, "f": False}
    assert out.loc[["d", "c", "b"], "slice"].tolist() == [0, 1, 2]
    assert out.loc[["c", "f"], "gap_prev"].tolist() == [1.25, 5]
    assert out["n_stacks"].eq(3).all() and out.loc["b", "n_slices"] == 3


def test_iterative_folds_spread_a_rare_label_evenly():
    y = np.zeros((100, 2), dtype=int)
    y[:, 1] = 1   # every series has one storage group
    y[:10, 0] = 1  # a rare label
    fold = iterative_folds(y, 5, seed=0)
    assert np.bincount(fold).tolist() == [20] * 5
    assert np.bincount(fold[:10], minlength=5).tolist() == [2] * 5


def test_pixel_stats_find_the_body_and_the_lungs():
    hu = np.full((100, 100), -1000, dtype=np.int16)
    hu[20:80, 10:90] = 40          # body
    hu[30:70, 20:45] = -850        # right lung
    hu[30:70, 55:80] = -850        # left lung
    hu[90:95, 0:5] = 100           # a small object apart from the body
    stats = pixel_stats(hu)
    assert (stats["body_top"], stats["body_bottom"], stats["body_left"], stats["body_right"]) == (20, 80, 10, 90)
    assert stats["body_share"] == pytest.approx(60 * 80 / 1e4)
    assert (stats["lung_top"], stats["lung_bottom"], stats["lung_left"], stats["lung_right"]) == (30, 70, 20, 80)
    assert stats["lung_share"] == pytest.approx(2 * 40 * 25 / 1e4)


def test_pixel_stats_keep_the_air_below_a_body_wider_than_the_image_outside():
    hu = np.full((100, 100), -1000, dtype=np.int16)
    hu[20:80, :] = 40              # body touching the left and right edges
    hu[30:70, 20:45] = -850        # right lung
    hu[30:70, 55:80] = -850        # left lung
    stats = pixel_stats(hu)
    assert (stats["body_top"], stats["body_bottom"]) == (20, 80)
    assert (stats["lung_top"], stats["lung_bottom"], stats["lung_left"], stats["lung_right"]) == (30, 70, 20, 80)


def test_pixel_stats_of_an_empty_slice():
    stats = pixel_stats(np.full((64, 64), -1000, dtype=np.int16))
    assert stats["body_share"] == 0 and stats["lung_share"] == 0 and stats["body_top"] == -1 and stats["lung_top"] == -1


def test_write_slice_stores_the_hu_and_keeps_existing_files(tmp_path):
    stored = np.full((512, 512), 24, dtype=np.int16)  # -1000 HU with the intercept of -1024
    stored[100:400, 50:450] = 1064                    # 40 HU
    dicom = write_dicom(tmp_path / "a.dcm", stored)
    out = tmp_path / "series" / "a.png"
    stats = write_slice(dicom, out)
    back = STORE.decode(out.read_bytes()).astype(np.int16) + HU_MIN
    assert np.array_equal(back, np.rint(read_hu(dicom)))
    assert not (tmp_path / "series" / "a.png.tmp").exists()
    assert (stats["body_top"], stats["body_left"]) == (100, 50)

    written = out.stat().st_mtime_ns
    assert write_slice(dicom, out) == stats and out.stat().st_mtime_ns == written
