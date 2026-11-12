from django.urls import include, path
from escriptorium.urls import urlpatterns

urlpatterns += [
    path("accounts/", include("pucas.cas_urls")),
    path("", include("htr2hpc.htr2hpc_urls")),
]
