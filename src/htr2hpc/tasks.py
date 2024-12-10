import logging

from celery import shared_task
from django.contrib.auth import get_user_model
from django.contrib.sites.shortcuts import get_current_site
from intspan import intspan

logger = logging.getLogger(__name__)

# override escriptorium training tasks to run on HPC

User = get_user_model()


def override_tasks():
    from escriptorium import celery

    celery.app.tasks.unregister("core.tasks.segtrain")


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
            "segtrain called with out document_pk or user_pk (document_pk={document_pk} user_pk={user_pk}"
        )
        return

    try:
        user = User.objects.get(pk=user_pk)
    except User.DoesNotExist:
        # error / bail out
        logger.error(f"segtrain called with invalid user_pk {user_pk}")
        return

    # do we need to mark the model as training?
    # should that be done here or in the script via api?

    # generate the command to run

    # create a name for an output directory based on mode and document id
    outdir = f"segtrain_doc{document_pk}"

    site = get_current_site()
    site_url = site.domain
    if not site_url.startswith("http"):
        site_url = f"https://{site_url}"

    arg_options = [
        f"--document {document_pk}",  # document id is always required
        f"--model-name segtrain_doc{document_pk}",  # our script requires a model name
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
