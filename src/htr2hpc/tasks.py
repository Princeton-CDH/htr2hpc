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


def start_remote_training(
    user, working_dir, train_cmd, document_pk, model_pk, task_report
):
    # common logic for segtrain and train to kick off remote training script

    # assume we're using LDAP accounts only so usernames match here and on hpc
    username = user.username
    api_token = user.auth_token.key

    # hostname and ssh key path set in django config
    logger.debug(
        f"Connecting to {settings.HPC_HOSTNAME} as {username} with keyfile {settings.HPC_SSH_KEYFILE}"
    )

    # add training command to task report
    task_report.append(f"remote training command:\n  {train_cmd}\n")

    # note: may need to use tmux to keep from disconnecting
    try:
        with Connection(
            host=settings.HPC_HOSTNAME,
            user=username,
            connect_timeout=10,
            connect_kwargs={"key_filename": settings.HPC_SSH_KEYFILE},
        ) as conn:
            # only notify once we successfully connect
            user.notify(
                "Starting remote training; slurm portion can be monitored on mydella",
                links=[
                    {
                        "text": "Della Active Jobs",
                        "src": "https://mydella.princeton.edu/pun/sys/dashboard/activejobs",
                    }
                ],
            )

            with conn.cd(working_dir):
                result = conn.run(
                    f"module load anaconda3/2024.6 && conda run -n htr2hpc {train_cmd}",
                    env={"ESCRIPTORIUM_API_TOKEN": api_token},
                    warn=True,  # don't throw unexpected error on exit != 0
                )
                logger.info(
                    f"remote training script completed; exit code: {result.exited}"
                )
                # refresh task report to get any messages added via api
                task_report.refresh_from_db()

                # script output is stored in result.stdout/result.stderr
                # add output to task report
                task_report.append(
                    f"\n\nremote script output:\n\n{result.stdout}\n\n{result.stderr}\n\n"
                )
                if "Slurm job was cancelled" in result.stdout:
                    task_report.cancel("(slurm cancellation)")
                    # notify the user of the error
                    user.notify(
                        "Training was cancelled via slurm",
                        id="training-warning",
                        level="warning",
                    )

                # normal exit code is zero;
                # if non-zero then training didn't succeed in some way
                return result.exited == 0

    except (AuthenticationException, UnexpectedExit) as err:
        if isinstance(err, AuthenticationException):
            logger.error(f"Authentication exception to remote connection: {err}")
            error_message = "Authentication failed; check that your account is set up properly on della"
        else:
            logger.error(f"Unexpected exit from remote connection: {err}")
            error_message = "Something went wrong running the training."

        # notify the user of the error
        user.notify(
            error_message,
            id="training-error",
            level="danger",
        )
        # also store in the task report
        # but first refresh task report to get any messages added via api
        task_report.refresh_from_db()
        task_report.error(error_message)

        # send training error event
        send_event(
            "document",
            document_pk,
            "training:error",
            {
                "id": model_pk,
            },
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
    task_report = task_group.taskreport_set.first()

    # if the model is older than the task group, then we infer that
    # overwrite was requested on the form (update an existing model)
    model_overwrite = model.version_created_at < task_group.created_at
    if model_overwrite:
        logger.debug(
            f"Inferring model overwrite requested based on model/task creation dates (model:{model.version_created_at} task:{task_group.created_at})"
        )

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
        f"--task-report {task_report.pk}",  # task reporting
    ]

    # part ids are optional
    if part_pks:
        # parse and serialize with intspan since that's what we use on the other side
        arg_options.append(f"--parts {intspan(part_pks)}")
    # model is technically optional for this task but it should
    # always be passed in by escriptorium calling code
    if model_pk:
        arg_options.append(f"--model {model_pk}")
        # eScriptorium behavior is to create a new model that will be
        # updated after training, so if we have a model we always want to update
        # the model; but when overwriting an existing model, only update if improved
        if model_overwrite:
            arg_options.append("--update-if-improved")
        else:
            arg_options.append("--update")

    opts = " ".join(arg_options)

    cmd = f"htr2hpc-train segmentation {site_url} {outdir} {opts}"
    # log the command to be run
    logger.info(f"remote training command: {cmd}")

    success = start_remote_training(
        user, working_dir, cmd, document_pk, model.pk, task_report
    )

    # refresh model data from the database,
    # since if htr2hpc-train script succeeded it should have been updated via api
    model.refresh_from_db()

    if success:
        # check for case where training completed but model did not improve.
        # i.e., no new model was uploaded or cloned model is still parent file
        if model.file is None or (
            model.parent is not None and model.file == model.parent.file
        ):
            user.notify(
                "Training completed but did not result in an improved model",
                id="training-warning",
                level="warning",
            )
            # assuming equivalent to did not converge in escriptorium code
            send_event(
                "document",
                document_pk,
                "training:error",
                {
                    "id": model.pk,
                },
            )

            # delete the original model unless overwrite was requested
            if not model_overwrite:
                model.delete()

        else:
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
        # we want to do that, but don't delete a pre-existing model
        # when overwrite was requested
        if not model_overwrite:
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
    task_report = task_group.taskreport_set.first()

    # if the model is older than the task group, then we infer that
    # overwrite was requested on the form (update an existing model)
    model_overwrite = model.version_created_at < task_group.created_at
    if model_overwrite:
        logger.debug(
            f"Inferring model overwrite requested based on model/task creation dates (model:{model.version_created_at} task:{task_group.created_at})"
        )

    # mark the model as being in training
    # would be nice if the script could handle, but that field is listed
    # as read only in the api
    model.training = True
    model.save()
    # send event to indicate training is starting
    send_event(
        "document",
        document.pk,
        "training:start",
        {
            "id": model_pk,
        },
    )

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
        f"--task-report {task_report.pk}",  # task reporting
    ]

    # model is technically optional for this task but it should
    # always be passed in by escriptorium calling code
    if model_pk:
        arg_options.append(f"--model {model_pk}")
        # eScriptorium behavior is to create a new model that will be
        # updated after training, so if we have a model we always want to update
        # the model; but when overwriting an existing model, only update if improved
        if model_overwrite:
            arg_options.append("--update-if-improved")
        else:
            arg_options.append("--update")

    opts = " ".join(arg_options)

    cmd = f"htr2hpc-train transcription {opts}"

    # log the command to be run
    logger.info(f"remote training command: {cmd}")

    success = start_remote_training(
        user, working_dir, cmd, document.pk, model.pk, task_report
    )

    # refresh model data from the database,
    # since if htr2hpc-train script succeeded it should have been updated via api
    model.refresh_from_db()
    if success:
        # NOTE: duplicated code from segtrain

        # check for case where training completed but model did not improve.
        # i.e., no new model was uploaded or cloned model is still parent file
        if model.file is None or (
            model.parent is not None and model.file == model.parent.file
        ):
            user.notify(
                "Training completed but did not result in an improved model",
                id="training-warning",
                level="warning",
            )
            # assuming equivalent to did not converge in escriptorium code
            send_event(
                "document",
                document.pk,
                "training:error",
                {
                    "id": model.pk,
                },
            )

            # delete the original model unless overwrite was requested
            if not model_overwrite:
                model.delete()

        else:
            # otherwise, notify the user that training completed sucessfully
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

    else:
        # escriptorium task deletes the model if there is an error;
        # we want to do that, unless overwrite of an existing model was requested
        if not model_overwrite:
            model.delete()
            return

    # when/if training completes:
    # - mark model as no longer being trained
    model.training = False
    model.save()


@shared_task(default_retry_delay=60 * 60, bind=True)
def hpc_user_setup(self, user_pk=None):
    try:
        user = User.objects.get(pk=user_pk)
    except User.DoesNotExist:
        # error / bail out
        logger.error(f"hpc_user_setup called with invalid user_pk {user_pk}")
        return

    # by default, escriptorium reporting code attaches signal handlers
    # that should create a task group and task report for this task id
    TaskReport = apps.get_model("reporting", "TaskReport")
    # don't error if the task report can't be found
    task_report = TaskReport.objects.filter(task_id=self.request.id).first()

    # hostname and ssh key path set in django config
    logger.debug(
        f"Connecting to {settings.HPC_HOSTNAME} as {user.username} with keyfile {settings.HPC_SSH_KEYFILE}"
    )

    # bash setup script is included with this package
    user_setup_script = settings.HTR2HPC_INSTALL_DIR / "train" / "user_setup.sh"
    user.notify(
        "Running user setup script, on first run this may take a while...",
        id="htr2hpc-setup-start",
        level="info",
    )
    try:
        with Connection(
            host=settings.HPC_HOSTNAME,
            user=user.username,
            connect_timeout=10,
            connect_kwargs={"key_filename": settings.HPC_SSH_KEYFILE},
        ) as conn:
            # copy setup script to server
            conn.put(user_setup_script)
            # run the script with options; skip ssh setup (must already be setup
            # for this task to run) and ensure htr2hpc install is up to date

            setup_cmd = (
                f"./{user_setup_script.name}  --skip-ssh-setup --reinstall-htr2hpc"
            )
            # document setup command options in task report
            if task_report:
                task_report.append(f"Running setup script:\n  {setup_cmd}\n\n")

            result = conn.run(setup_cmd)
            # remove the setup script from the server; don't error if not there
            # (if user clicks the button twice it may already be removed)
            conn.run(f"rm -f ./{user_setup_script.name}")

            # add script output to task report
            if task_report:
                # script output is stored in result.stdout/result.stderr
                task_report.append(
                    f"\n\nsetup script output:\n\n{result.stdout}\n\n{result.stderr}\n\n"
                )

            if "Setup complete" in result.stdout:
                user.notify(
                    "Remote setup completed",
                    id="htr2hpc-setup-success",
                    level="success",
                )
            # log script output for debugging
            logger.debug(f"user setup script output:\n{result.stdout}")
    except AuthenticationException as err:
        error_message = f"Authentication exception to remote connection: {err}"
        logger.error(error_message)
        if task_report:
            task_report.append(error_message)
        # notify the user of the error
        user.notify(
            "Authentication failed; check that your account on della is set up for remote access",
            id="setup-error",
            level="danger",
        )
    except UnexpectedExit as err:
        error_message = f"Error running remote setup script: {err}"
        logger.error(error_message)
        if task_report:
            task_report.append(error_message)
        logger.error(error_message)
        user.notify(
            "Something went wrong running remote user setup",
            id="setup-error",
            level="danger",
        )
