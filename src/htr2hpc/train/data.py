import datetime
import logging
import pathlib
import subprocess
from collections import defaultdict

from kraken.containers import BaselineLine, Region, Segmentation

# skip import, syntax error in current kraken
# from kraken.lib.arrow_dataset import build_binary_dataset
from kraken.serialization import serialize


logger = logging.getLogger(__name__)


# get a document part from eS api and convert into kraken objects


def get_segmentation_data(
    api, document_details, part_id, image_dir
) -> tuple[Segmentation, tuple]:
    """Get a single document part from the eScriptorium API and generate
    a kraken segmentation object.

    Returns a tuple of the segmentation object and the part details from the API,
    which includes image size needed for serialization.
    """

    # document details includes id (pk) and valid line and block types
    document_id = document_details.pk
    # convert list of line types to a lookup from id to name
    line_types = {ltype.pk: ltype.name for ltype in document_details.valid_line_types}
    # same for block types (used for regions)
    block_types = {btype.pk: btype.name for btype in document_details.valid_block_types}
    part = api.document_part_details(document_id, part_id)

    # adapted from escriptorium.app.core.tasks.make_segmentation_training_data
    # (additional logic in make_recognition_segmentation )

    # TODO: for recognition task, we need to get transcription lines
    # from api.document_part_transcription_list
    # each one includes a line id, transcription id, and content
    # ... need to match up based on line pk

    # NOTE: eS celery task training prep only includes regions
    # for segmentation, not recognition

    # gather regions in a dictionary keyed on type name for
    # the segmentation object (name -> list of regions)
    # and also a lookup by id, for assocating lines with regions
    regions = defaultdict(list)
    region_pk_to_id = {}
    for region in part.regions:
        # map pk to external id for lines to use
        region_pk_to_id[region.pk] = region.external_id
        # get region type and create a kraken region object
        region_type = block_types.get(region.typology, "default")
        regions[region_type].append(
            Region(
                id=region.external_id, boundary=region.box, tags={"type": region_type}
            )
        )

    # gather base lines
    baselines = [
        BaselineLine(
            id=line.external_id,
            baseline=line.baseline,
            boundary=line.mask,
            # eScriptorium api returns a single region pk
            # kraken takes a list of string ids
            regions=[region_pk_to_id[line.region]],
            # NOTE: eS celery task training prep only includes text
            # when generating training data for recognition, not segmentation
            # this mirrors the behavior from eS code for export:
            # mark as default if type is not in the public list
            # db includes more types but they are not marked as public
            tags={"type": line_types.get(line.typology, "default")},
        )
        for line in part.lines
    ]
    logger.info(f"Document {document_id} part {part_id}: {len(baselines)} baselines")

    logger.info(
        f"Document {document_id} part {part_id}:  {len(part.regions)} regions, {len(regions.keys())} block types"
    )

    image_uri = f"{api.base_url}{part.image.uri}"
    # download the file and save in the image dir; name based on url without media prefix
    image_file = api.download_file(
        image_uri, image_dir, part.image.uri.replace("/media/", "").replace("/", "-")
    )

    seg = Segmentation(
        # eS task code has text-direction hardcoded as horizontal-lr
        text_direction="horizontal-lr",
        # imagename should be a path to a local image file
        imagename=image_file,
        type="baselines",
        lines=baselines,
        regions=regions,
        script_detection=False,
    )
    return (
        seg,
        part,
    )


def serialize_segmentation(segmentation: Segmentation, part):
    """Serialize a segmentation object as ALTO XML for use as training data.
    Requires kraken :class:`~kraken.containers.Segmentation` and part
    details returned by eScriptorum API.
    """
    # TODO: consider moving this into above method as optional behavior

    # output xml with a base name corresponding to the image file
    xml_path = pathlib.Path(segmentation.imagename).with_suffix(".xml")
    # make image path a local / relative path
    segmentation.imagename = pathlib.Path(segmentation.imagename).name
    logger.debug(f"Serializing segmentation as {xml_path}")
    xml_path.open("w").write(serialize(segmentation, image_size=part.image.size))


def compile_data(segmentations, output_dir):
    """Compile a list of kraken segmentation objects into a binary file for
    recognition training."""
    output_file = output_dir / "dataset.arrow"
    build_binary_dataset(
        files=segmentations, format_type=None, output_file=str(output_file)
    )
    return output_file


def get_model(api, model_id, training_type, output_dir):
    """Download a model file from the eScriptorium and save it to the specified
    directory. Raises a ValueError if the model is not the specified
    training type. Returns a :class:`pathlib.Path` to the downloaded file."""

    model_info = api.model_details(model_id)
    if model_info.job != training_type:
        raise ValueError(
            f"Model {model_id} is a {model_info.job} model, but {training_type} requested"
        )
    return api.download_file(model_info.file, output_dir)


def get_training_data(api, output_dir, document_id, part_ids=None):
    # if part ids are not specified, get all parts
    if part_ids is None:
        doc_parts = api.document_parts_list(document_id)
        part_ids = [part.pk for part in doc_parts.results]

    # document details includes line and block types
    document_details = api.document_details(document_id)

    # get segmentation data for each part of the document that is requested
    segmentation_data = [
        get_segmentation_data(api, document_details, part_id, output_dir)
        for part_id in part_ids
    ]
    # if we're generating alto-xml (i.e., segmentation training data),
    # serialize each of the parts we downloaded
    [serialize_segmentation(seg, part) for (seg, part) in segmentation_data]

    # NOTE: binary compiled data is only supported train and not segtrain
    # compiled_data = compile_data(segmentations, output_dir)


def get_best_model(model_dir: pathlib.Path) -> pathlib.Path | None:
    best = list(model_dir.glob("*_best.mlmodel"))
    return best[0] if best else None


# use api.update_model with model id and pathlib.Path to model file
# to update existing model record with new file
