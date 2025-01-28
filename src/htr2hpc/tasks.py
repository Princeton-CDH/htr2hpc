import logging
from datetime import datetime

from celery import shared_task
from django.apps import apps
from django.contrib.auth import get_user_model
from django.contrib.sites.models import Site
from django.conf import settings
from django.utils.translation import gettext as _
from intspan import intspan
from fabric import Connection
from invoke.exceptions import UnexpectedExit
from paramiko.ssh_exception import AuthenticationException

# imports from escriptorium
from apps.users.consumers import send_event

logger = logging.getLogger(__name__)

# override escriptorium training tasks to run on HPC

User = get_user_model()


def directory_timestamp():
    # use timestamps to ensure working directories for training data
    # are unique; use human-readable date with time including microseconds
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def start_remote_training(user, working_dir, train_cmd, document_pk, model_pk):
    # common logic for segtrain and train to kick off remote training script

    # assume we're using LDAP accounts only so usernames match here and on hpc
    username = user.username
    api_token = user.auth_token.key

    # hostname and ssh key path set in django config
    logger.debug(
        f"Connecting to {settings.HPC_HOSTNAME} as {username} with keyfile {settings.HPC_SSH_KEYFILE}"
    )

    user.notify(
        "Starting remote training; slurm portion can be monitored on mydella",
        links=[
            {
                "text": "Della Active Jobs",
                "src": "https://mydella.princeton.edu/pun/sys/dashboard/activejobs",
            }
        ],
    )
    # note: may need to use tmux to keep from disconnecting
    try:
        with Connection(
            host=settings.HPC_HOSTNAME,
            user=username,
            connect_timeout=10,
            connect_kwargs={"key_filename": settings.HPC_SSH_KEYFILE},
        ) as conn:
            with conn.cd(working_dir):
                result = conn.run(
                    f"module load anaconda3/2024.6 && conda run -n htr2hpc {train_cmd}",
                    env={"ESCRIPTORIUM_API_TOKEN": api_token},
                )
                logger.info(
                    f"remote training script completed; exit code: {result.exited}"
                )
                # normal exit code is zero
                return result.exited == 0

                # script output is stored in
                # result.stdout/result.stderr
    except AuthenticationException as err:
        logger.error(f"Authentication exception to remote connction: {err}")
        # send training error event
        send_event(
            "document",
            document_pk,
            "training:error",
            {
                "id": model_pk,
            },
        )
        user.notify(
            _(
                "Authentication failed; check that your account is set up properly on della"
            ),
            id="training-error",
            level="danger",
        )
        return False

    except UnexpectedExit as err:
        logger.error(f"Unexpected exit from remote connection: {err}")
        # send training error event
        send_event(
            "document",
            document_pk,
            "training:error",
            {
                "id": model_pk,
            },
        )
        user.notify(
            _("Something went wrong running the training."),
            id="training-error",
            level="danger",
        )
        return False
    except Exception as err:
        logger.error(f"Error: {err} ({err.__class__}")
        # send training error event
        send_event(
            "document",
            document_pk,
            "training:error",
            {
                "id": model_pk,
            },
        )
        user.notify(
            _("Something went wrong running the training."),
            id="training-error",
            level="danger",
        )
        return False


@shared_task(default_retry_delay=60 * 60)
def segtrain(
    model_pk=None,
    part_pks=[],
    document_pk=None,
    task_group_pk=None,  # do train/segtrain update task status anywhere?
    user_pk=None,
    **kwargs,
):
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
    # use task creation time to determine if model was created just prior to training
    TaskGroup = apps.get_model("reporting", "TaskGroup")
    task_group = TaskGroup.objects.get(pk=task_group_pk)

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

    # create a name for an output directory based on mode and document id
    working_dir = f"/scratch/gpfs/{user.username}/htr2hpc"
    # includes a timestamp to ensure uniqueness, since
    # script will fail if there is an existing directory
    outdir = f"segtrain_doc{document_pk}_{directory_timestamp()}"

    # generate the command to run
    site = Site.objects.get(pk=settings.SITE_ID)
    site_url = site.domain
    if not site_url.startswith("http"):
        site_url = f"https://{site_url}"

    arg_options = [
        f"--document {document_pk}",  # document id is always required
        "--no-progress",  # disable progressbar
    ]

    # part ids are optional
    if part_pks:
        # parse and serialize with intspan since that's what we use on the other side
        arg_options.append(f"--parts {intspan(part_pks)}")
    # model is technically optional for this task but it should
    # always be passed in by escriptorium calling code
    if model_pk:
        # eScriptorium behavior is to create a new model that will be
        # updated after training, so if we have a model we always want --update
        arg_options.append(f"--model {model_pk} --update")
    opts = " ".join(arg_options)

    cmd = f"htr2hpc-train segmentation {site_url} {outdir} {opts}"
    # log the command to be run
    logger.info(f"remote training command: {cmd}")

    success = start_remote_training(user, working_dir, cmd, document_pk, model.pk)

    # get a fresh copy of the model from the database,
    # since if htr2hpc-train script succeeded it should have been updated via api
    model = OcrModel.objects.get(pk=model_pk)

    if success:
        # - notify the user that training completed sucessfully
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
    else:
        # if training did not suceeed:

        # escriptorium task deletes the model if there is an error;
        # we want to do that, but check if the model was created after
        # this task started so we don't delete a pre-existing model
        # when overwrite was requested
        if model.file is None or task_group.created_at < model.version_created_at:
            model.delete()
            return

    # mark model as no longer being trained
    model.training = False
    model.save()


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

    try:
        user = User.objects.get(pk=user_pk)
    except User.DoesNotExist:
        # error / bail out
        logger.error(f"train called with invalid user_pk {user_pk}")
        return

    # create a name for an output directory based on mode and document id
    working_dir = f"/scratch/gpfs/{user.username}/htr2hpc"
    # create a name for an output directory based on mode and transcripiton id
    # include a timestamp to ensure uniqueness, since
    # script will fail if there is an existing directory
    outdir = f"train_transcription{transcription_pk}_{directory_timestamp()}"

    # get document from transcription
    Transcription = apps.get_model("core", "Transcription")
    # get the requested model from the db
    OcrModel = apps.get_model("core", "OcrModel")
    model = OcrModel.objects.get(pk=model_pk)
    transcription = Transcription.objects.get(pk=transcription_pk)
    document = transcription.document
    # use task creation time to determine if model record is new
    TaskGroup = apps.get_model("reporting", "TaskGroup")
    task_group = TaskGroup.objects.get(pk=task_group_pk)

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
        # model name should not be used with model id; model id should always be present
        # parse and serialize part ids with intspan
        f"--parts {intspan(part_pks)}",
        "--no-progress",  # disable progressbar
    ]

    # model is technically optional for this task but it should
    # always be passed in by escriptorium calling code
    if model_pk:
        # eScriptorium behavior is to create a new model that will be
        # updated after training, so if we have a model we always want --update
        arg_options.append(f"--model {model_pk} --update")

    opts = " ".join(arg_options)

    cmd = f"htr2hpc-train transcription {opts}"

    # log the command to be run
    logger.info(f"remote training command: {cmd}")

    success = start_remote_training(user, working_dir, cmd, document.pk, model.pk)

    # get a fresh copy of the model from the database,
    # since if htr2hpc-train script succeeded it should have been updated via api
    model = OcrModel.objects.get(pk=model_pk)

    if not success:
        # escriptorium task deletes the model if there is an error;
        # we want to do that, but check if the model was created after
        # this task started so we don't delete a pre-existing model
        # when overwrite was requested
        if model.file is None or task_group.created_at < model.version_created_at:
            model.delete()
            return

    # when/if training completes:
    # - mark model as no longer being trained
    model.training = False
    model.save()
    # - notify the user that training completed
    user.notify(_("Training finished!"), id="training-success", level="success")
    # send training complete event
    send_event(
        "document",
        document.pk,
        "training:done",
        {
            "id": model.pk,
        },
    )
