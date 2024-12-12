import logging

from celery import shared_task
from django.apps import apps
from django.contrib.auth import get_user_model
from django.contrib.sites.models import Site
from django.conf import settings
from intspan import intspan

# imports from escriptorium
from apps.users.consumers import send_event

logger = logging.getLogger(__name__)

# override escriptorium training tasks to run on HPC

User = get_user_model()


@shared_task(default_retry_delay=60 * 60)
def segtrain(
    model_pk=None,
    part_pks=[],
    document_pk=None,
    task_group_pk=None,
    user_pk=None,
    **kwargs,
):
    print("### running override segtrain task")

    # NOTE: when called from the web ui, the SegTrainForm includes
    # a field for model_name but that value is not passed to the celery task
    # HOWEVER: when a model name is specified, the form process method
    # creates a new, empty model db record with that name; when override
    # is not requested, form logic creates a copy of the model.
    # So we may need to revise the script with an option to update
    # the specified model.

    # we require both of these; are they really optional?
    if not all([user_pk, document_pk]):
        # can't proceed without both of these
        logger.error(
            f"segtrain called without document_pk or user_pk; document_pk={document_pk} user_pk={user_pk}"
        )
        return

    try:
        user = User.objects.get(pk=user_pk)
    except User.DoesNotExist:
        # error / bail out
        logger.error(f"segtrain called with invalid user_pk {user_pk}")
        return

    # notify user that training is starting
    send_event(
        "document",
        document_pk,
        "training:start",
        {
            "id": model_pk,
        },
    )

    # TODO: mark the model as training
    # OcrModel = apps.get_model("core", "OcrModel")
    # would be nice if the script could handle, but that field is listed
    # as read only in the api

    # generate the command to run

    # create a name for an output directory based on mode and document id
    # TODO: make this relative to scratch & username?
    outdir = f"segtrain_doc{document_pk}"

    site = Site.objects.get(pk=settings.SITE_ID)
    site_url = site.domain
    if not site_url.startswith("http"):
        site_url = f"https://{site_url}"

    arg_options = [
        f"--document {document_pk}",  # document id is always required
        f"--model-name segtrain_doc{document_pk}",  # TODO: get from model
    ]

    # part and model are optional
    if part_pks:
        # parse and serialize with intspan since that's what we use on the other side
        arg_options.append(f"--parts {intspan(part_pks)}")
    if model_pk:
        arg_options.append(f"--model {model_pk}")
    opts = " ".join(arg_options)

    cmd = f"htr2hpc-train segmentation {site_url} {outdir} {opts}"

    # for now just output the command
    logger.info(cmd)


@shared_task(default_retry_delay=60 * 60)
def train(
    transcription_pk=None,
    model_pk=None,
    task_group_pk=None,
    part_pks=None,
    user_pk=None,
    **kwargs,
):
    # we require all of these; are they really optional?
    if not all([user_pk, transcription_pk, part_pks]):
        # can't proceed without these
        logger.error(
            "train called without transcription_pk, part_pks, or user_pk "
            + f"transcription_pk={transcription_pk} part_pks={part_pks} user_pk={user_pk}"
        )
        return

    # create a name for an output directory based on mode and document id
    # TODO: make this relative to scratch & username?
    outdir = f"train_transcription{transcription_pk}"

    # get document from transcription
    Transcription = apps.get_model("core", "Transcription")
    # OcrModel = apps.get_model("core", "OcrModel")
    transcription = Transcription.objects.get(pk=transcription_pk)
    document = transcription.document

    site = Site.objects.get(pk=settings.SITE_ID)
    site_url = site.domain
    if not site_url.startswith("http"):
        site_url = f"https://{site_url}"

    arg_options = [
        # mode-specific options must come before all others
        f"--transcription {transcription_pk}",
        site_url,
        str(outdir),
        f"--document {document.pk}",  # document id is always required
        f"--model-name train_transcription{transcription_pk}",  # TODO: get from model
        # parse and serialize part ids with intspan
        f"--parts {intspan(part_pks)}",
    ]

    if model_pk:
        arg_options.append(f"--model {model_pk}")
    opts = " ".join(arg_options)

    cmd = f"htr2hpc-train transcription {opts}"

    # for now just output the command
    logger.info(cmd)
