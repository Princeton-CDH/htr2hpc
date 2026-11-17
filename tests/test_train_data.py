"""Tests for htr2hpc.train.data — training data utilities."""

import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock, call, create_autospec, patch

import pytest
from kraken.containers import BaselineLine, Region, Segmentation
from kraken.serialization import serialize

from htr2hpc.api_client import eScriptoriumAPIClient
from htr2hpc.train.data import (
    TrainingDataCounts,
    compile_data,
    get_best_model,
    get_document_parts,
    get_model_file,
    get_prelim_model,
    get_segmentation_data,
    get_training_data,
    get_transcription_lines,
    serialize_segmentation,
    split_segmentation,
    upload_best_model,
    upload_models,
)


def make_api(**attrs):
    """Fake eScriptorium API client. Built from the real class, so calling a
    method that doesn't exist (or with the wrong arguments) fails the test.
    Instance attributes set in __init__ aren't part of the spec; pass them in."""
    api = create_autospec(eScriptoriumAPIClient, instance=True)
    for name, value in attrs.items():
        setattr(api, name, value)
    return api


# ---------------------------------------------------------------------------
# serialize_segmentation — kraken 6.x tags generate TAGREFS in ALTO output
# ---------------------------------------------------------------------------


def test_kraken6_tags_generate_tagrefs_in_alto():
    region = Region(
        id="r1",
        boundary=[[0, 0], [0, 100], [100, 100], [100, 0]],
        tags={"type": [{"type": "paragraph"}]},
    )
    seg = Segmentation(
        type="baselines",
        imagename="test.jpg",
        text_direction="horizontal-lr",
        script_detection=False,
        regions={"paragraph": [region]},
        lines=[],
    )
    assert "<OtherTag" in serialize(seg, image_size=(100, 100))


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
        _make_mlmodel(tmp_path, "model_0.mlmodel", 0.80)

        with patch("htr2hpc.train.data.get_model_accuracy", return_value=0.95):
            result = get_best_model(tmp_path)

        assert result == best

    def test_returns_none_when_no_models(self, tmp_path):
        result = get_best_model(tmp_path)
        assert result is None

    def test_best_file_not_returned_if_below_original_accuracy(self, tmp_path):
        _make_mlmodel(tmp_path, "model_best.mlmodel", 0.70)
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


# ---------------------------------------------------------------------------
# get_transcription_lines
# ---------------------------------------------------------------------------


def _make_text_line(line_id, content):
    line = MagicMock()
    line.line = line_id
    line.content = content
    return line


def _make_transcription_page(lines, next_url=None):
    page = MagicMock()
    page.results = lines
    page.next = next_url
    return page


def test_get_transcription_lines_follows_pages():
    api = make_api()
    page2 = _make_transcription_page([_make_text_line(3, "page2line")])
    page1 = _make_transcription_page(
        [_make_text_line(1, "a"), _make_text_line(2, "b")],
        next_url="http://example.com/api/lines/?page=2",
    )
    page1.next_page.return_value = page2
    api.document_part_transcription_list.return_value = page1

    result = get_transcription_lines(api, document_id=10, part_id=5, transcription_id=3)

    # text of every page, looked up by line id
    assert result == {1: "a", 2: "b", 3: "page2line"}


# ---------------------------------------------------------------------------
# get_document_parts
# ---------------------------------------------------------------------------


def _make_parts_page(pks, next_url=None):
    page = MagicMock()
    page.results = [MagicMock(pk=pk) for pk in pks]
    page.next = next_url
    return page


def test_get_document_parts_follows_pages():
    api = make_api()
    page2 = _make_parts_page([104, 105])
    page1 = _make_parts_page(
        [101, 102, 103], next_url="http://example.com/api/parts/?page=2"
    )
    page1.next_page.return_value = page2
    api.document_parts_list.return_value = page1

    result = get_document_parts(api, document_id=42)

    assert result == [101, 102, 103, 104, 105]


# ---------------------------------------------------------------------------
# get_model_file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("job", ["Recognize", "Segment"])
def test_get_model_file_downloads_model_of_requested_type(job, tmp_path):
    api = make_api()
    api.model_details.return_value = SimpleNamespace(
        job=job, file="/models/foo.mlmodel"
    )

    result = get_model_file(api, model_id=7, training_type=job, output_dir=tmp_path)

    api.download_file.assert_called_once_with("/models/foo.mlmodel", tmp_path)
    assert result == api.download_file.return_value


def test_get_model_file_raises_on_job_mismatch(tmp_path):
    api = make_api()
    api.model_details.return_value = SimpleNamespace(
        job="Segment", file="/models/seg.mlmodel"
    )

    with pytest.raises(ValueError, match="Recognize requested"):
        get_model_file(api, model_id=8, training_type="Recognize", output_dir=tmp_path)

    # a model of the wrong type is not downloaded
    api.download_file.assert_not_called()


def test_get_model_file_returns_none_when_no_file(tmp_path):
    # a newly created model record in eScriptorium has no file yet
    api = make_api()
    api.model_details.return_value = SimpleNamespace(job="Recognize", file=None)

    result = get_model_file(
        api, model_id=5, training_type="Recognize", output_dir=tmp_path
    )

    assert result is None
    api.download_file.assert_not_called()


# ---------------------------------------------------------------------------
# get_segmentation_data
#
# The eScriptorium API is the boundary here, so we fake the objects it returns
# (plain namespaces, so a misspelled attribute fails loudly) and check how they
# are translated into kraken objects.
# ---------------------------------------------------------------------------

IMAGE_URI = "/media/documents/42/page1.jpg"


def make_document_details():
    """Document details with one custom line type and one custom block type."""
    return SimpleNamespace(
        pk=42,
        valid_line_types=[
            SimpleNamespace(pk=1, name="Default"),
            SimpleNamespace(pk=2, name="Heading"),
        ],
        valid_block_types=[SimpleNamespace(pk=10, name="Main")],
    )


def make_line(pk, region=None, typology=1):
    return SimpleNamespace(
        pk=pk,
        external_id=f"line_{pk}",
        baseline=[[0, 5], [9, 5]],
        mask=[[0, 0], [9, 0], [9, 9], [0, 9]],
        region=region,
        typology=typology,
    )


def make_part(regions=(), lines=()):
    return SimpleNamespace(
        regions=list(regions),
        lines=list(lines),
        image=SimpleNamespace(uri=IMAGE_URI, size=(100, 100)),
    )


def make_region(pk, typology=10):
    return SimpleNamespace(
        pk=pk,
        external_id=f"region_{pk}",
        box=[[0, 0], [9, 0], [9, 9], [0, 9]],
        typology=typology,
    )


@pytest.fixture
def api(tmp_path):
    """Fake eScriptorium API client that "downloads" the page image."""
    api = make_api(base_url="https://escriptorium.example.com")
    api.download_file.return_value = tmp_path / "documents-42-page1.jpg"
    return api


def test_get_segmentation_data_links_lines_to_typed_regions(api, tmp_path):
    api.document_part_details.return_value = make_part(
        # one region of a known type, one of a type the document doesn't define
        regions=[make_region(100, typology=10), make_region(101, typology=99)],
        lines=[
            make_line(1, region=100, typology=2),
            # orphan line: no region
            make_line(2, region=None, typology=1),
            # unknown line type
            make_line(3, region=101, typology=99),
        ],
    )

    seg, part = get_segmentation_data(api, make_document_details(), 5, tmp_path)

    # regions are grouped by the name of their block type, falling back to
    # "default" when the type is not one of the document's public types
    assert set(seg.regions) == {"Main", "default"}
    assert seg.regions["Main"][0].id == "region_100"
    # lines refer to their region by external id, since the API gives a pk
    assert [line.regions for line in seg.lines] == [
        ["region_100"],
        None,
        ["region_101"],
    ]
    assert [line.tags for line in seg.lines] == [
        {"type": [{"type": "Heading"}]},
        {"type": [{"type": "Default"}]},
        {"type": [{"type": "default"}]},
    ]
    # geometry is taken over unchanged: the region box is the region boundary,
    # the line mask is the line boundary
    assert seg.regions["Main"][0].boundary == make_region(100).box
    assert seg.lines[0].baseline == make_line(1).baseline
    assert seg.lines[0].boundary == make_line(1).mask
    # part details are passed back too, for the image size needed to serialize
    assert part.image.size == (100, 100)


def test_get_segmentation_data_downloads_page_image(api, tmp_path):
    api.document_part_details.return_value = make_part()

    seg, _ = get_segmentation_data(api, make_document_details(), 5, tmp_path)

    # image is saved in the training dir, with the /media/ prefix removed and
    # the path flattened into a single file name
    api.download_file.assert_called_once_with(
        f"https://escriptorium.example.com{IMAGE_URI}",
        tmp_path,
        "documents-42-page1.jpg",
    )
    assert seg.imagename == tmp_path / "documents-42-page1.jpg"


def test_get_segmentation_data_has_no_text_without_transcription(api, tmp_path):
    api.document_part_details.return_value = make_part(lines=[make_line(1)])

    seg, _ = get_segmentation_data(api, make_document_details(), 5, tmp_path)

    assert seg.lines[0].text is None
    # no need to ask the API for transcription lines
    api.document_part_transcription_list.assert_not_called()


def test_get_segmentation_data_adds_transcription_text(api, tmp_path):
    api.document_part_details.return_value = make_part(
        lines=[make_line(1), make_line(2)]
    )
    # transcription only exists for the first line
    api.document_part_transcription_list.return_value = _make_transcription_page(
        [_make_text_line(1, "first line")]
    )

    seg, _ = get_segmentation_data(
        api, make_document_details(), 5, tmp_path, transcription_id=3
    )

    # text is matched to lines by line pk
    assert [line.text for line in seg.lines] == ["first line", None]
    api.document_part_transcription_list.assert_called_once_with(42, 5, 3)


def test_get_segmentation_data_skips_unusable_part(api, tmp_path):
    # the api client returns None when a part can't be parsed
    api.document_part_details.return_value = None

    assert get_segmentation_data(api, make_document_details(), 5, tmp_path) == (
        None,
        None,
    )
    # and we shouldn't try to download an image for it
    api.download_file.assert_not_called()


# ---------------------------------------------------------------------------
# serialize_segmentation
# ---------------------------------------------------------------------------


def test_serialize_segmentation_writes_alto_next_to_image(tmp_path):
    image = tmp_path / "documents-42-page1.jpg"
    seg = Segmentation(
        type="baselines",
        imagename=image,
        text_direction="horizontal-lr",
        script_detection=False,
        regions={
            "Main": [
                Region(
                    id="region_1",
                    boundary=[[0, 0], [0, 99], [99, 99], [99, 0]],
                    tags={"type": [{"type": "Main"}]},
                )
            ]
        },
        lines=[
            BaselineLine(
                id="line_1",
                baseline=[[5, 50], [90, 50]],
                boundary=[[5, 40], [90, 40], [90, 60], [5, 60]],
                regions=["region_1"],
                tags={"type": [{"type": "Default"}]},
            )
        ],
    )

    serialize_segmentation(seg, SimpleNamespace(image=SimpleNamespace(size=(100, 100))))

    xml = (tmp_path / "documents-42-page1.xml").read_text()
    # the image is referenced by file name only, not by the absolute path
    # of the machine that generated it, so the data can be moved to the HPC
    assert "documents-42-page1.jpg" in xml
    assert str(tmp_path) not in xml
    # line types are written as tags that the kraken training can read back
    assert "TAGREFS" in xml


# ---------------------------------------------------------------------------
# compile_data
# ---------------------------------------------------------------------------


def test_compile_data_builds_arrow_file_with_fixed_split(mocker, tmp_path):
    mock_build = mocker.patch("htr2hpc.train.data.build_binary_dataset")
    segmentations = [MagicMock(), MagicMock()]

    output = compile_data(segmentations, tmp_path)

    assert output == tmp_path / "train.arrow"
    mock_build.assert_called_once_with(
        files=segmentations,
        format_type=None,
        output_file=str(tmp_path / "train.arrow"),
        # train/validation split is fixed here rather than left to chance,
        # so that every task training on this data sees the same split
        random_split=(0.9, 0.1, 0),
    )


# ---------------------------------------------------------------------------
# get_training_data
# ---------------------------------------------------------------------------


def make_segmentation(tmp_path, name, n_lines=1, n_regions=1):
    """Segmentation with the given number of lines and regions, with a
    matching (segmentation, part) pair as returned by get_segmentation_data."""
    regions = [
        Region(
            id=f"{name}_r{i}",
            boundary=[[0, 0], [0, 99], [99, 99], [99, 0]],
            tags={"type": [{"type": "Main"}]},
        )
        for i in range(n_regions)
    ]
    lines = [
        BaselineLine(
            id=f"{name}_l{i}",
            baseline=[[5, 50], [90, 50]],
            boundary=[[5, 40], [90, 40], [90, 60], [5, 60]],
            tags={"type": [{"type": "Default"}]},
        )
        for i in range(n_lines)
    ]
    seg = Segmentation(
        type="baselines",
        imagename=tmp_path / f"{name}.jpg",
        text_direction="horizontal-lr",
        script_detection=False,
        regions={"Main": regions} if regions else {},
        lines=lines,
    )
    return seg, SimpleNamespace(image=SimpleNamespace(size=(100, 100)))


@pytest.fixture
def training_api():
    api = make_api()
    api.document_details.return_value = make_document_details()
    return api


def test_get_training_data_segmentation_writes_xml_and_split(
    mocker, training_api, tmp_path
):
    # no transcription: training data is ALTO XML for segmentation training.
    # Each part has a single region (of one type), so the region count is the
    # same whether regions or region types are counted; see PR notes.
    parts = {
        1: make_segmentation(tmp_path, "part1", n_lines=2, n_regions=1),
        2: make_segmentation(tmp_path, "part2", n_lines=3, n_regions=1),
    }
    mocker.patch(
        "htr2hpc.train.data.get_segmentation_data",
        side_effect=lambda api, doc, part_id, out_dir, tid: parts[part_id],
    )

    counts = get_training_data(training_api, tmp_path, 42, part_ids=[1, 2])

    assert counts == TrainingDataCounts(parts=2, lines=5, regions=2)
    assert (tmp_path / "part1.xml").exists()
    assert (tmp_path / "part2.xml").exists()
    # the train/validation split was defined for both files
    split = (tmp_path / "train.txt").read_text() + (
        tmp_path / "validate.txt"
    ).read_text()
    assert "parts/part1.xml" in split
    assert "parts/part2.xml" in split


def test_get_training_data_recognition_compiles_dataset(mocker, training_api, tmp_path):
    # with a transcription, training data is compiled to a binary dataset
    # instead of being serialized as XML
    mock_compile = mocker.patch("htr2hpc.train.data.compile_data")
    seg, part = make_segmentation(tmp_path, "part1", n_lines=4, n_regions=1)
    mock_get_seg = mocker.patch(
        "htr2hpc.train.data.get_segmentation_data", return_value=(seg, part)
    )

    counts = get_training_data(
        training_api, tmp_path, 42, part_ids=[1], transcription_id=7
    )

    assert counts == TrainingDataCounts(parts=1, lines=4, regions=1)
    # transcription id is passed along so lines get their text
    assert mock_get_seg.call_args == call(
        training_api, training_api.document_details.return_value, 1, tmp_path, 7
    )
    mock_compile.assert_called_once_with([seg], tmp_path)
    assert not list(tmp_path.glob("*.xml"))


def test_get_training_data_leaves_out_unusable_parts(mocker, training_api, tmp_path):
    good = make_segmentation(tmp_path, "good", n_lines=2, n_regions=1)
    mocker.patch(
        "htr2hpc.train.data.get_segmentation_data",
        side_effect=[(None, None), good],
    )

    counts = get_training_data(training_api, tmp_path, 42, part_ids=[1, 2])

    # the broken part is not counted, since counts are used
    # to size the HPC job, and nothing is serialized for it
    assert counts == TrainingDataCounts(parts=1, lines=2, regions=1)
    assert [f.name for f in tmp_path.glob("*.xml")] == ["good.xml"]


def test_get_training_data_defaults_to_all_document_parts(
    mocker, training_api, tmp_path
):
    training_api.document_parts_list.return_value = _make_parts_page([11, 12])
    mock_get_seg = mocker.patch(
        "htr2hpc.train.data.get_segmentation_data",
        # each part gets its own segmentation, since serializing modifies it
        side_effect=lambda api, doc, part_id, out_dir, tid: make_segmentation(
            tmp_path, f"part{part_id}"
        ),
    )

    counts = get_training_data(training_api, tmp_path, 42)

    assert [c.args[2] for c in mock_get_seg.call_args_list] == [11, 12]
    assert counts.parts == 2


# ---------------------------------------------------------------------------
# upload_models
# ---------------------------------------------------------------------------


def test_upload_models_uploads_in_epoch_order(tmp_path):
    # kraken names models by epoch (name_0, name_1, ...); a plain alphabetical
    # sort would put epoch 10 before epoch 2
    for epoch in (10, 2, 1):
        (tmp_path / f"model_{epoch}.mlmodel").write_bytes(b"x")
    api = make_api()

    count = upload_models(api, tmp_path, "Segment", show_progress=False)

    assert count == 3
    assert [c.args[0].name for c in api.model_create.call_args_list] == [
        "model_1.mlmodel",
        "model_2.mlmodel",
        "model_10.mlmodel",
    ]
    assert all(c.kwargs == {"job": "Segment"} for c in api.model_create.call_args_list)


@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="known limitation (see NOTE in upload_models): files are sorted by "
    "epoch number, which a *_best.mlmodel file does not have",
)
def test_upload_models_with_best_model_file(tmp_path):
    (tmp_path / "model_0.mlmodel").write_bytes(b"x")
    (tmp_path / "model_best.mlmodel").write_bytes(b"x")

    upload_models(make_api(), tmp_path, "Recognize", show_progress=False)


def test_upload_models_counts_only_successful_uploads(tmp_path):
    for epoch in (1, 2):
        (tmp_path / f"model_{epoch}.mlmodel").write_bytes(b"x")
    api = make_api()
    # the api client returns a falsy value for an upload that didn't work
    api.model_create.side_effect = [MagicMock(), None]

    assert upload_models(api, tmp_path, "Segment", show_progress=False) == 1


# ---------------------------------------------------------------------------
# upload_best_model
# ---------------------------------------------------------------------------


def test_upload_best_model_does_nothing_without_a_best_model(mocker, tmp_path):
    mocker.patch("htr2hpc.train.data.get_best_model", return_value=None)
    api = make_api()

    assert upload_best_model(api, tmp_path, "Recognize") is None
    # nothing is created or overwritten in eScriptorium
    api.model_create.assert_not_called()
    api.model_update.assert_not_called()


def test_upload_best_model_creates_new_model_named_without_best_suffix(
    mocker, tmp_path
):
    mock_best = mocker.patch("htr2hpc.train.data.get_best_model")
    best = tmp_path / "mymodel_best.mlmodel"
    mock_best.return_value = best
    api = make_api()

    assert upload_best_model(api, tmp_path, "Recognize") == best

    api.model_create.assert_called_once_with(
        model_name="mymodel", model_file=best, job="Recognize"
    )
    api.model_update.assert_not_called()


def test_upload_best_model_updates_existing_model_when_id_given(mocker, tmp_path):
    mock_best = mocker.patch("htr2hpc.train.data.get_best_model")
    best = tmp_path / "mymodel_best.mlmodel"
    mock_best.return_value = best
    api = make_api()

    assert upload_best_model(api, tmp_path, "Segment", model_id=12) == best

    api.model_update.assert_called_once_with(12, model_file=best, job="Segment")
    api.model_create.assert_not_called()


@pytest.fixture
def patch_accuracy(mocker):
    """Fake the accuracy kraken stores in a model file.
    Call with a dict of model path -> accuracy."""

    def patch_accuracy(accuracies):
        return mocker.patch(
            "htr2hpc.train.data.get_model_accuracy",
            side_effect=lambda path: accuracies[path],
        )

    return patch_accuracy


def test_upload_best_model_skips_model_worse_than_original(tmp_path, patch_accuracy):
    original = tmp_path / "original.mlmodel"
    original.write_bytes(b"x")
    best = tmp_path / "run_best.mlmodel"
    best.write_bytes(b"x")
    patch_accuracy({original: 0.90, best: 0.80})
    api = make_api()

    result = upload_best_model(api, tmp_path, "Segment", original_model=original)

    # training did not improve on the original, so nothing is uploaded
    assert result is None
    api.model_create.assert_not_called()
    api.model_update.assert_not_called()


def test_upload_best_model_returns_none_when_upload_fails(mocker, tmp_path):
    mock_best = mocker.patch("htr2hpc.train.data.get_best_model")
    mock_best.return_value = tmp_path / "mymodel_best.mlmodel"
    api = make_api()
    api.model_create.return_value = None

    assert upload_best_model(api, tmp_path, "Recognize") is None
