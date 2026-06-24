"""Tests for htr2hpc.train.data — training data utilities."""
import pathlib
import shutil
from unittest.mock import MagicMock, patch

import pytest

from htr2hpc.train.data import get_best_model, get_prelim_model, split_segmentation


# ---------------------------------------------------------------------------
# split_segmentation
# ---------------------------------------------------------------------------


def _make_xml_files(directory: pathlib.Path, count: int):
    """Create `count` dummy .xml files in `directory`."""
    files = []
    for i in range(count):
        f = directory / f"part_{i:03d}.xml"
        f.write_text("<alto/>")
        files.append(f)
    return files


def test_split_segmentation_creates_train_and_validate(tmp_path):
    _make_xml_files(tmp_path, 10)
    split_segmentation(tmp_path)
    assert (tmp_path / "train.txt").exists()
    assert (tmp_path / "validate.txt").exists()


def test_split_segmentation_every_tenth_goes_to_validate(tmp_path):
    _make_xml_files(tmp_path, 20)
    split_segmentation(tmp_path)
    validate_lines = (tmp_path / "validate.txt").read_text().splitlines()
    train_lines = (tmp_path / "train.txt").read_text().splitlines()
    # every 10th file (index 0, 10) -> 2 validate files
    assert len(validate_lines) == 2
    # remaining 18 files go to train
    assert len(train_lines) == 18


def test_split_segmentation_no_overlap(tmp_path):
    _make_xml_files(tmp_path, 15)
    split_segmentation(tmp_path)
    validate_lines = set((tmp_path / "validate.txt").read_text().splitlines())
    train_lines = set((tmp_path / "train.txt").read_text().splitlines())
    assert validate_lines.isdisjoint(train_lines)


def test_split_segmentation_paths_prefixed_with_parts(tmp_path):
    _make_xml_files(tmp_path, 5)
    split_segmentation(tmp_path)
    train_lines = (tmp_path / "train.txt").read_text().splitlines()
    validate_lines = (tmp_path / "validate.txt").read_text().splitlines()
    for line in train_lines + validate_lines:
        assert line.startswith("parts/")


def test_split_segmentation_single_file_goes_to_validate(tmp_path):
    # With 1 file, index 0 goes to validate; train is empty
    _make_xml_files(tmp_path, 1)
    split_segmentation(tmp_path)
    validate_lines = (tmp_path / "validate.txt").read_text().splitlines()
    train_lines = (tmp_path / "train.txt").read_text().splitlines()
    assert len(validate_lines) == 1
    assert len(train_lines) == 0  # "\n".join([]) == "", "".splitlines() == []


def test_split_segmentation_empty_dir_creates_empty_files(tmp_path):
    split_segmentation(tmp_path)
    assert (tmp_path / "train.txt").read_text() == ""
    assert (tmp_path / "validate.txt").read_text() == ""


# ---------------------------------------------------------------------------
# get_prelim_model
# ---------------------------------------------------------------------------


def test_get_prelim_model_creates_copy(tmp_path):
    original = tmp_path / "mymodel_best.mlmodel"
    original.write_bytes(b"model data")
    prelim = get_prelim_model(original)
    assert prelim.exists()
    assert prelim.read_bytes() == b"model data"


def test_get_prelim_model_suffix_is_prelim(tmp_path):
    original = tmp_path / "mymodel_best.mlmodel"
    original.write_bytes(b"x")
    prelim = get_prelim_model(original)
    assert prelim.name.endswith("_prelim.mlmodel")


def test_get_prelim_model_does_not_modify_original(tmp_path):
    original = tmp_path / "mymodel_best.mlmodel"
    original.write_bytes(b"original content")
    get_prelim_model(original)
    assert original.read_bytes() == b"original content"


def test_get_prelim_model_returns_path_in_same_dir(tmp_path):
    original = tmp_path / "run_0001_best.mlmodel"
    original.write_bytes(b"x")
    prelim = get_prelim_model(original)
    assert prelim.parent == tmp_path


def test_get_prelim_model_name_strips_last_segment(tmp_path):
    # "run_0001_best.mlmodel" -> strips "_best" -> "run_0001_prelim.mlmodel"
    original = tmp_path / "run_0001_best.mlmodel"
    original.write_bytes(b"x")
    prelim = get_prelim_model(original)
    assert prelim.name == "run_0001_prelim.mlmodel"


# ---------------------------------------------------------------------------
# get_best_model
# ---------------------------------------------------------------------------


def _make_mlmodel(directory: pathlib.Path, name: str, accuracy: float):
    """Create a dummy .mlmodel file and patch get_model_accuracy for it."""
    f = directory / name
    f.write_bytes(b"fake model")
    return f


class TestGetBestModel:
    """Tests for get_best_model using mocked get_model_accuracy."""

    def test_returns_best_mlmodel_file_when_present(self, tmp_path):
        best = _make_mlmodel(tmp_path, "model_best.mlmodel", 0.95)
        other = _make_mlmodel(tmp_path, "model_0.mlmodel", 0.80)

        with patch("htr2hpc.train.data.get_model_accuracy", return_value=0.95):
            result = get_best_model(tmp_path)

        assert result == best

    def test_returns_none_when_no_models(self, tmp_path):
        result = get_best_model(tmp_path)
        assert result is None

    def test_best_file_not_returned_if_below_original_accuracy(self, tmp_path):
        best = _make_mlmodel(tmp_path, "model_best.mlmodel", 0.70)
        original = _make_mlmodel(tmp_path, "original.mlmodel", 0.90)

        def fake_accuracy(path):
            return 0.70 if "best" in path.name else 0.90

        with patch("htr2hpc.train.data.get_model_accuracy", side_effect=fake_accuracy):
            result = get_best_model(tmp_path, original_model=original)

        assert result is None

    def test_fallback_to_accuracy_scan_when_no_best_file(self, tmp_path):
        m1 = _make_mlmodel(tmp_path, "model_0.mlmodel", 0.70)
        m2 = _make_mlmodel(tmp_path, "model_1.mlmodel", 0.85)
        m3 = _make_mlmodel(tmp_path, "model_2.mlmodel", 0.80)

        accuracies = {m1: 0.70, m2: 0.85, m3: 0.80}

        with patch(
            "htr2hpc.train.data.get_model_accuracy",
            side_effect=lambda p: accuracies[p],
        ):
            result = get_best_model(tmp_path)

        assert result == m2

    def test_fallback_scan_respects_original_model_threshold(self, tmp_path):
        original = _make_mlmodel(tmp_path, "original.mlmodel", 0.90)
        m1 = _make_mlmodel(tmp_path, "model_0.mlmodel", 0.85)

        accuracies = {original: 0.90, m1: 0.85}

        with patch(
            "htr2hpc.train.data.get_model_accuracy",
            side_effect=lambda p: accuracies[p],
        ):
            result = get_best_model(tmp_path, original_model=original)

        assert result is None

    def test_fallback_scan_returns_model_beating_original(self, tmp_path):
        original = _make_mlmodel(tmp_path, "original.mlmodel", 0.80)
        m1 = _make_mlmodel(tmp_path, "model_0.mlmodel", 0.92)

        accuracies = {original: 0.80, m1: 0.92}

        with patch(
            "htr2hpc.train.data.get_model_accuracy",
            side_effect=lambda p: accuracies[p],
        ):
            result = get_best_model(tmp_path, original_model=original)

        assert result == m1

    def test_best_file_returned_even_without_original(self, tmp_path):
        best = _make_mlmodel(tmp_path, "run_best.mlmodel", 0.88)

        with patch("htr2hpc.train.data.get_model_accuracy", return_value=0.88):
            result = get_best_model(tmp_path)

        assert result == best
