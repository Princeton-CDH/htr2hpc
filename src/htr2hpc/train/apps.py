import logging
import pathlib
from collections import defaultdict

import parsl
from parsl.app.app import python_app, bash_app
from kraken.containers import BaselineLine, Region, Segmentation
from kraken.lib.arrow_dataset import build_binary_dataset
from kraken.serialization import serialize

logger = logging.getLogger(__name__)


# get a document part from eS api and convert into kraken objects


@python_app(executors=["local"])
def get_segmentation_data(api, document_id, part_id, image_dir) -> Segmentation:
    # get a single document part from eScripotrium API and return as a
    # kraken segmentation

    # TODO: can we cache these? types used across all parts
    # get list of line types and convert to a lookup from id to name
    line_types = {ltype.pk: ltype.name for ltype in api.list_types("line").results}
    # same for block types (used for regions)
    block_types = {btype.pk: btype.name for btype in api.list_types("block").results}
    part = api.document_part_details(document_id, part_id)
    # print(part)
    # image uri: part.image.uri
    # regions : part.regions

    # adapted from escriptorium.app.core.tasks.make_segmentation_training_data

    # gather base lines
    baselines = [
        BaselineLine(
            id=line.external_id,
            baseline=line.baseline,
            boundary=line.mask,
            # NOTE: eS celery task training prep only includes text
            # when generating training data for recognition, not segmentation
            #
            text=line.text if hasattr(line, "text") else None,
            # this mirrors the behavior from eS code for export:
            # mark as default if type is not in the public list
            # db includes more types but they are not marked as public
            tags={"type": line_types.get(line.typology, "default")},
        )
        for line in part.lines
    ]
    logger.info(f"Document {document_id} part {part_id}: {len(baselines)} baselines")

    # NOTE: eS celery task training prep only includes regions
    # for segmentation, not recognition

    # gather regions in a dictionary keyed on type name
    # name -> list of regions
    regions = defaultdict(list)
    for region in part.regions:
        region_type = block_types.get(region.typology, "default")
        regions[region_type].append(
            Region(
                id=region.external_id, boundary=region.box, tags={"type": region_type}
            )
        )
    logger.info(
        f"Document {document_id} part {part_id}:  {len(part.regions)} regions, {len(regions.keys())} block types"
    )

    image_uri = f"{api.base_url}{part.image.uri}"
    # download the file and save in the image dir; name based on url without media prefix
    image_file = api.download_file(
        image_uri, image_dir, part.image.uri.replace("/media/", "").replace("/", "-")
    )

    return Segmentation(
        # eS code has text-direction hardcoded as horizontal-lr
        text_direction="horizontal-lr",
        # imagename should be a path to a local image file; can we use parsl File?
        imagename=image_file,
        type="baselines",
        lines=baselines,
        regions=regions,
        script_detection=False,
    )


@python_app
def compile_data(segmentations, output_dir):
    output_file = output_dir / "dataset.arrow"
    build_binary_dataset(
        files=segmentations, format_type=None, output_file=str(output_file)
    )
    return output_file


# should this be a parsl python app? run in parallel with segmentation data?
def get_model(api, model_id, training_type, output_dir):
    model_info = api.model_details(model_id)
    if model_info.job != training_type:
        raise ValueError(
            f"Model {model_id} is a {model_info.job} model, but {training_type} requested"
        )
    return api.download_file(model_info.file, output_dir)


# def prep_training_data(es_base_url, es_api_token, document_id, part_ids=None):
def prep_training_data(api, base_dir, document_id, part_ids=None):
    # if part ids are not specified, get all parts
    if part_ids is None:
        doc_parts = api.document_parts_list(document_id)
        part_ids = [part.pk for part in doc_parts.results]

    output_dir = base_dir / f"doc{document_id}"
    output_dir.mkdir()
    # TODO: rename parts? pages? (now contains images & alto xml)
    image_dir = output_dir / "images"
    image_dir.mkdir()

    # kick off parsl python app to get document parts as kraken segmentations
    segmentation_data = [
        get_segmentation_data(api, document_id, part_id, image_dir)
        for part_id in part_ids
    ]
    # get all the results
    segmentations = [s.result() for s in segmentation_data]

    # test serializing as alto to compare export, compilation
    for seg in segmentations:
        # output xml next to the image file
        xml_path = pathlib.Path(seg.imagename).with_suffix(".xml")
        # make image path a local / relative path
        seg.imagename = pathlib.Path(seg.imagename).name
        xml_path.open("w").write(serialize(seg))

    # for segtrain, return the path that contains the xml files
    return image_dir

    # FIXME: binary compiled data only seems to work for train and not segtrain
    # compiled_data = compile_data(segmentations, output_dir).result()


@bash_app(executors=["hpc"])
def segtrain(
    inputs=[],
    outputs=[],
    stderr=parsl.AUTO_LOGNAME,
    stdout=parsl.AUTO_LOGNAME,
):
    # first input should be directory for input data
    input_data_dir = inputs[0]
    # second input is model to use as starting point
    input_model = inputs[1]
    # third input is model output
    output_model = inputs[2]
    # create a log directory adjacent to the model
    log_dir = output_model.parent / "kraken_logs"
    log_dir.mkdir()
    # third input is worker count
    workers = inputs[3]
    return (
        f"ketos segtrain --epochs 200 --resize new -i {input_model}"
        + f" -o {output_model} --workers {workers} -d cuda:0 "
        + f"-f xml {input_data_dir}/*.xml "
        + f"--log-dir {log_dir}"
    )
    # TODO: return model as output file


# use api.update_model with model id and pathlib.Path to model file
# to update existing model record with new file
