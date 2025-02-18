from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from htr2hpc.tasks import hpc_user_setup


@login_required
@require_http_methods(["POST"])
def remote_user_setup(request):
    # it seems that the taskreport must be associated with a document,
    # so skip the task reporting logic here and just use notifications

    # queue the celery setup task
    hpc_user_setup.delay(user_pk=request.user.pk)
    # redirect back to the profile page
    redirect = HttpResponseRedirect(reverse("profile-api-key"))
    redirect.status_code = 303  # See other
    return redirect
