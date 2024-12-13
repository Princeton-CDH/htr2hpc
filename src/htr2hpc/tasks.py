import logging

from celery import shared_task
from django.apps import apps
from django.contrib.auth import get_user_model
from django.contrib.sites.models import Site
from django.conf import settings
from django.utils.translation import gettext as _
from intspan import intspan
from fabric import Connection
from invoke.exceptions import UnexpectedExit

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
    task_group_pk=None,  # do train/segtrain update task status anywhere?
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

    # get the requested model from the db
    OcrModel = apps.get_model("core", "OcrModel")
    model = OcrModel.objects.get(pk=model_pk)
    # mark the model as being in training
    # would be nice if the script could handle, but that field is listed
    # as read only in the api
    model.training = True
    model.save()
    # send event to indicate training is starting
    send_event(
        "document",
        document_pk,
        "training:start",
        {
            "id": model_pk,
        },
    )

    # assume we're using LDAP accounts only so usernames match here and on hpc
    username = user.username
    api_token = user.auth_token.key

    # create a name for an output directory based on mode and document id
    working_dir = f"/scratch/gpfs/{username}/htr2hpc"
    # TODO: should we add a timestamp to ensure uniqueness?
    # script will fail if there is an existing directory
    outdir = f"segtrain_doc{document_pk}"

    # generate the command to run
    site = Site.objects.get(pk=settings.SITE_ID)
    site_url = site.domain
    if not site_url.startswith("http"):
        site_url = f"https://{site_url}"

    arg_options = [
        f"--document {document_pk}",  # document id is always required
        f"--model-name {model.name}",
        "--no-progress",  # disable progressbar
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

    # hostname and ssh key path set in django config
    logger.debug(
        f"Connecting to {settings.HPC_HOSTNAME} as {username} with keyfile {settings.HPC_SSH_KEYFILE}"
    )

    user.notify(
        "Starting remote training; slurm portion can be monitored on mydella",
        links=["https://mydella.princeton.edu/pun/sys/dashboard/activejobs"],
    )
    # note: may need to use tmux to keep from disconnecting
    try:
        with Connection(
            host=settings.HPC_HOSTNAME,
            user=username,
            connect_kwargs={"key_filename": settings.HPC_SSH_KEYFILE},
        ) as conn:
            with conn.cd(working_dir):
                result = conn.run(
                    f"module load anaconda3/2024.6 && conda run -n htr2hpc {cmd}",
                    env={"ESCRIPTORIUM_API_TOKEN": api_token},
                )
                print(result)
        # TODO: maybe script can write job id to a  dot file in the output dir
        # so the celery task can check the status?
        # or do we even need that level of detail (pending/running/complete)
    except UnexpectedExit as err:
        print(err)
        # send training error event
        send_event(
            "document",
            document_pk,
            "training:error",
            {
                "id": model.pk,
            },
        )
        user.notify(
            _("Something went wrong running the training."),
            id="training-error",
            level="danger",
        )
        # escriptorium task deletes the model if there is an error
        # is it always safe to do that?
        # model.delete()
        return

    # could the celery task exit? or does it need to monitor the
    # remote script?

    # when/if training completes:
    # - mark model as no longer being trained
    model.training = False
    model.save()
    # - notify the user that training completed
    user.notify(_("Training finished!"), id="training-success", level="success")
    # send training complete event
    send_event(
        "document",
        document_pk,
        "training:done",
        {
            "id": model.pk,
        },
    )


# todo: maybe make the fab command a method that can be run for testing
# from shell/cli ?


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
