import datetime
import logging
import pathlib
import subprocess
from collections import defaultdict

from kraken.containers import BaselineLine, Region, Segmentation

# skip import, syntax error in current kraken
# from kraken.lib.arrow_dataset import build_binary_dataset
from kraken.serialization import serialize
from simple_slurm import Slurm

logger = logging.getLogger(__name__)


# get a document part from eS api and convert into kraken objects


def get_segmentation_data(api, document_id, part_id, image_dir) -> Segmentation:
    # get a single document part from eScripotrium API and return as a
    # kraken segmentation

    # TODO: can we cache these? types used across all parts
    # NOTE: document details includes valid block types and valid line types,
    # maybe we should get that and pass it in to this method
    # get list of line types and convert to a lookup from id to name
    line_types = {ltype.pk: ltype.name for ltype in api.list_types("line").results}
    # same for block types (used for regions)
    block_types = {btype.pk: btype.name for btype in api.list_types("block").results}
    part = api.document_part_details(document_id, part_id)

    # adapted from escriptorium.app.core.tasks.make_segmentation_training_data
    # (additional logic in make_recognition_segmentation )

    # TODO: for recognition task, we need to get transcription lines
    # from api.document_part_transcription_list
    # each one includes a line id, transcription id, and content
    # ... need to match up based on line pk

    # gather base lines
    baselines = [
        BaselineLine(
            id=line.external_id,
            baseline=line.baseline,
            boundary=line.mask,
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


def seralize_segmentation(segmentation: Segmentation):
    # output xml with a base name corresponding to the image file
    xml_path = pathlib.Path(segmentation.imagename).with_suffix(".xml")
    # make image path a local / relative path
    segmentation.imagename = pathlib.Path(segmentation.imagename).name
    logger.debug(f"Serializing segmentation as {xml_path}")
    xml_path.open("w").write(serialize(segmentation))


def compile_data(segmentations, output_dir):
    output_file = output_dir / "dataset.arrow"
    build_binary_dataset(
        files=segmentations, format_type=None, output_file=str(output_file)
    )
    return output_file


# TODO: figure out how to run this as a parsl python app,
# in parallel with downloading training data
def get_model(api, model_id, training_type, output_dir):
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

    # kick off parsl python app to get document parts as kraken segmentations
    segmentation_data = [
        get_segmentation_data(api, document_id, part_id, output_dir)
        for part_id in part_ids
    ]
    # if we're generating alto-xml (i.e., segmentation training data),
    # serialize each of the parts we downloaded
    [seralize_segmentation(seg) for seg in segmentation_data]

    # NOTE: binary compiled data is only supported train and not segtrain
    # compiled_data = compile_data(segmentations, output_dir)


def segtrain(
    input_data_dir: pathlib.Path,
    input_model: pathlib.Path,
    output_model: pathlib.Path,
    num_workers: int = 8,
    # optional param to specify name based on document? include date?
):
    segtrain_slurm = Slurm(
        nodes=1,
        ntasks=1,
        cpus_per_task=num_workers,
        mem_per_cpu="4G",
        gres=["gpu:1"],
        job_name="segtrain",
        output=f"segtrain_{Slurm.JOB_ARRAY_MASTER_ID}.out",
        time=datetime.timedelta(minutes=20),
        # time=datetime.timedelta(hours=2),
    )
    # do we want to use CUDA Multi-Process Service (MPS) ?
    # della documentation says to specify with --gpu-mps,
    # but simple slurm doesn't recognize this as a valid directive.
    # Work around that by adding it as a command (if indeed we want this)
    # segtrain_slurm.add_cmd("#SBATCH --gpu-mps")

    # add commands for setup steps
    segtrain_slurm.add_cmd("module purge")
    segtrain_slurm.add_cmd("module load anaconda3/2024.6")
    segtrain_slurm.add_cmd("conda activate htr2hpc")
    logger.debug(f"sbatch file\n: {segtrain_slurm}")
    # sbatch returns the job id for the created job
    segtrain_cmd = (
        f"ketos segtrain --epochs 200 --resize new -i {input_model}"
        + f" -o {output_model} --workers {num_workers} -d cuda:0 "
        + f"-f xml {input_data_dir}/*.xml "
        # + "--precision 16"  # automatic mixed precision for nvidia gpu
    )

    logger.debug(f"segtrain command: {segtrain_cmd}")
    return segtrain_slurm.sbatch(segtrain_cmd)
    # TODO: calling function needs to check for best model
    # or no model improvement


def get_best_model(model_dir: pathlib.Path) -> pathlib.Path | None:
    best = list(model_dir.glob("*_best.mlmodel"))
    return best[0] if best else None


def slurm_job_queue_status(job_id: int) -> str:
    # use squeue to get the full-word status of a single job
    result = subprocess.run(
        ["squeue", f"--jobs={job_id}", "--only-job-state", "--format=%T", "--noheader"],
        capture_output=True,
        text=True,
    )
    # raise subprocess.CalledProcessError if return code indicates an error
    result.check_returncode()
    # return task status without any whitespace
    # squeue doesn't report on the task when it is completed and no longer in the queue,
    # so empty string means the job is complete
    return result.stdout.strip()


def slurm_job_status(job_id: int) -> set:
    result = subprocess.run(
        ["sacct", f"--jobs={job_id}", "--format=state%15", "--noheader"],
        capture_output=True,
        text=True,
    )
    # raise subprocess.CalledProcessError if return code indicates an error
    result.check_returncode()
    # sacct returns a table with status for each portion of the job;
    # return all unique status codes for now
    return set(result.stdout.split())


# use api.update_model with model id and pathlib.Path to model file
# to update existing model record with new file
