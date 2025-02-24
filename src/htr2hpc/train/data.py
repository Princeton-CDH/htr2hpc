import logging
import pathlib
import shutil
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass


from kraken.containers import BaselineLine, Region, Segmentation
from kraken.lib.arrow_dataset import build_binary_dataset
from tqdm import tqdm

# skip import, syntax error in current kraken
# from kraken.lib.arrow_dataset import build_binary_dataset
from kraken.serialization import serialize

from htr2hpc.api_client import get_model_accuracy

logger = logging.getLogger(__name__)


# get a document part from eS api and convert into kraken objects


def get_transcription_lines(api, document_id, part_id, transcription_id):
    # The API could have multiple pages of transcription lines;
    # loop until all pages of results are consumed
    text_lines = {}
    # get the first page of results
    transcription_lines = api.document_part_transcription_list(
        document_id, part_id, transcription_id
    )
    while True:
        # gather lines of text from the current page
        for text_line in transcription_lines.results:
            # Each transcription line includes a line id,
            # transcription id, and text content.
            # Add to dict so we can lookup content by line id
            text_lines[text_line.line] = text_line.content
        # if there is another page of results, get them
        if transcription_lines.next:
            transcription_lines = transcription_lines.next_page()
        # otherwise, we've hit the end; stop looping
        else:
            break

    return text_lines


def get_segmentation_data(
    api, document_details, part_id, image_dir, transcription_id=None
) -> tuple[Segmentation, tuple]:
    """Get a single document part from the eScriptorium API and generate
    a kraken segmentation object.

    Returns a tuple of the segmentation object and the part details from the API,
    which includes image size needed for serialization. Includes transcription
    text when a `transcription_id` is specified.
    """

    # document details includes id (pk) and valid line and block types
    document_id = document_details.pk
    # convert list of line types to a lookup from id to name
    line_types = {ltype.pk: ltype.name for ltype in document_details.valid_line_types}
    # same for block types (used for regions)
    block_types = {btype.pk: btype.name for btype in document_details.valid_block_types}
    part = api.document_part_details(document_id, part_id)

    # adapted from escriptorium.app.core.tasks.make_segmentation_training_data
    # and  make_recognition_segmentation
    # NOTE: regions are not strictly needed for recognition training,
    # but does not seem to hurt to include them

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

    # recognition training requires transcription text content
    # if a transcription id is specified, retrieve transcription content
    if transcription_id:
        text_lines = get_transcription_lines(
            api, document_id, part_id, transcription_id
        )
    else:
        text_lines = {}

    baselines = [
        BaselineLine(
            id=line.external_id,
            baseline=line.baseline,
            boundary=line.mask,
            # eScriptorium api returns a single region pk;
            # kraken takes a list of string ids
            # orphan lines have no region
            regions=[region_pk_to_id[line.region]] if line.region else None,
            # mark as default if type is not in the public list
            # db includes more types but they are not marked as public
            tags={"type": line_types.get(line.typology, "default")},
            # get text transcription content for this line, if available
            # (only possible when transcription id is specified)
            text=text_lines.get(line.pk),
        )
        for line in part.lines
    ]

    logger.debug(f"Document {document_id} part {part_id}: {len(baselines)} baselines")
    logger.debug(
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
    
    
def split_segmentation(training_data_dir):
    """Takes as input directory containing ALTO XML files and creates
    a train.txt and validate.txt file which define the train/validation
    split. This allows consistency across the multiple train tasks.
    """
    files_xml = list(training_data_dir.glob("*.xml"))
    files_validate = [f"parts/{f.name}" for f in files_xml[::10]]
    files_train = [f"parts/{f.name}" for f in files_xml if f"parts/{f.name}" not in files_validate]
    
    logger.info(f"Files in train set:\n {files_train}")
    logger.info(f"Files in validation set:\n {files_validate}")
    
    train_path = training_data_dir / "train.txt"
    validate_path = training_data_dir / "validate.txt"
    train_path.open("w").write("\n".join(files_train))
    validate_path.open("w").write("\n".join(files_validate))


def compile_data(segmentations, output_dir):
    """Compile a list of kraken segmentation objects into a binary file for
    recognition training."""
    # NOTE: get code errors in kraken if the image path is not valid.
    # Image path on created segments should be relative to current
    # working directory. Must resolve so the kraken binary compile
    # function can load image files by path.
    output_file = output_dir / "train.arrow"
    build_binary_dataset(
        files=segmentations,
        format_type=None,  # None = kraken Segmentation objects
        output_file=str(output_file),
        random_split=(0.9, 0.1, 0), # predefine train/validation split for consistency across mult train tasks
    )
    return output_file


def get_model_file(api, model_id, training_type, output_dir):
    """Download a model file from the eScriptorium and save it to the specified
    directory. Raises a ValueError if the model is not the specified
    training type. Returns a :class:`pathlib.Path` to the downloaded file."""
    model_info = api.model_details(model_id)
    if model_info.job != training_type:
        raise ValueError(
            f"Model {model_id} is a {model_info.job} model, but {training_type} requested"
        )
    if model_info.file is None:
        # when eScriptorium creates a new model record, it has no file
        # and the file url is null
        # return None for no file
        return None

    return api.download_file(model_info.file, output_dir)


def get_document_parts(api, document_id):
    part_ids = []
    # get first page of results
    document_parts = api.document_parts_list(document_id)
    while True:
        # retrieve part ids from the current page and check for more
        part_ids.extend([part.pk for part in document_parts.results])
        # if there is another page of results, get it
        if document_parts.next:
            document_parts = document_parts.next_page()
        # otherwise, stop looping
        else:
            break
    return part_ids


@dataclass
class TrainingDataCounts:
    parts: int = 0
    lines: int = 0
    regions: int = 0


def get_training_data(
    api, output_dir, document_id, part_ids=None, transcription_id=None
) -> TrainingDataCounts:
    # if part ids are not specified, get all parts
    if part_ids is None:
        part_ids = get_document_parts(api, document_id)

    # document details includes line and block types
    document_details = api.document_details(document_id)

    # get segmentation data for each part of the document that is requested
    segmentation_data = [
        get_segmentation_data(
            api, document_details, part_id, output_dir, transcription_id
        )
        for part_id in part_ids
    ]
    # get counts of data for reporting and scaling slurm request
    counts = TrainingDataCounts(parts=len(segmentation_data))
    # segmentation data is a list of tuples of segment, part
    for seg, _ in segmentation_data:
        counts.lines += len(seg.lines)
        counts.regions += len(seg.regions)

    # if transcription id is specified, compile as binary dataset
    # for recognition training
    if transcription_id:
        segmentations = [seg for seg, _ in segmentation_data]
        compile_data(segmentations, output_dir)

    # if no transcription id is specified, then serialize as
    # alto-xml for segmentation training
    else:
        # serialize each of the parts that were downloaded
        [serialize_segmentation(seg, part) for (seg, part) in segmentation_data]
        # define train/validation split
        split_segmentation(output_dir)

    # return the total counts for various pieces of training data
    return counts
    

def get_prelim_model(input_model: pathlib.Path):
    """Copies the input model to a file with suffix `_prelim.mlmodel`, 
    then returns the path to that newly created file.
    """
    prelim_model = input_model.parent / ('_'.join(input_model.name.split('_')[:-1]) + '_prelim.mlmodel')
    shutil.copy(input_model, prelim_model)
    return prelim_model


def get_best_model(
    model_dir: pathlib.Path, original_model: pathlib.Path = None
) -> pathlib.Path | None:
    """Find the best model in the specified `model_dir` directory.
    By default, looks for a file named `*_best.mlmodel`. If no best model
    is found by filename, looks for best model based on accuracy score
    in kraken metadata. When `original_model` is specified, accuracy
    must be better than the original to be considered 'best'.
    """
    best_accuracy = 0
    # when original model is specified, initialize
    # best accuracy value from that model
    if original_model:
        best_accuracy = get_model_accuracy(original_model)
        print(
            f"Must be better than original model {original_model.name} accuracy {best_accuracy:0.3f}"
        )
    # kraken should normally identify the best model for us
    best = list(model_dir.glob("*_best.mlmodel"))
    # if one was found, return it
    if best:
        accuracy = get_model_accuracy(best[0])
        if accuracy > best_accuracy:
            print(f"Using kraken identified best model {best[0].name}")
            return best[0]
        else:
            print("Training did not improve on original model")

    # if not, try to find one based on accuracy metadata
    else:
        if original_model:
            best = original_model
        print(f"Looking for best model by accuracy")
        for model in model_dir.glob("*.mlmodel"):
            accuracy = get_model_accuracy(model)
            print(f"model: {model.name} accuracy: {accuracy:0.3f}")
            # if accuracy is better than our current best, this model is new best
            if accuracy > best_accuracy:
                best = model
                best_accuracy = accuracy

        # if we found a model better than the original, return it
        if best and best != original_model:
            return best
        if best == original_model:
            print("Training did not improve on original model")


def upload_models(
    api, model_dir: pathlib.Path, model_type: str, show_progress=True
) -> int:
    """Upload all model files in the specified model directory to eScriptorum
    with the specified job type (Segment/Recognize). Returns a count of the
    number of models created."""
    uploaded = 0

    # segtrain creates models based on modelname with _0, _1, _2 ... _49
    # sort numerically on the latter portion of the name
    # NOTE: this older logic breaks with new -q early option that creates a _best model
    modelfiles = sorted(
        model_dir.glob("*.mlmodel"), key=lambda path: int(path.stem.split("_")[-1])
    )
    for model_file in tqdm(
        modelfiles,
        desc=f"Uploading {model_type} models",
        disable=not show_progress,
    ):
        # NOTE: should have error handling here;
        # what kinds of exceptions/errors might occur?
        created = api.model_create(model_file, job=model_type)
        if created:
            uploaded += 1

    return uploaded


def upload_best_model(
    api,
    model_dir: pathlib.Path,
    model_type: str,
    model_id: int = None,
    original_model: pathlib.Path = None,
) -> Optional[pathlib.Path]:
    """Upload the best model in the specified model directory to eScriptorium
    with the specified job type (Segment/Recognize).  If a model id is specified,
    updates that model; otherwise creates a new model. Returns :class:`pathlib.Path` object
    for best model if found and successfully uploaded; otherwise returns None."""
    best_model = get_best_model(model_dir, original_model=original_model)
    if not best_model:
        return None
    # common parameters used for both create and update
    params = {
        "model_file": best_model,
        "job": model_type,
    }
    # if model id is specified, update existing model
    if model_id:
        model = api.model_update(model_id, **params)
    else:
        model = api.model_create(
            # strip off _best from file for model name in eScriptorium
            model_name=best_model.stem.replace("_best", ""),
            **params,
        )
    if model:
        return best_model

    # TODO: return something different here if api call failed?
