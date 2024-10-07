from django.urls import include, path

from escriptorium.urls import urlpatterns

urlpatterns += [
    (path("accounts/", include("pucas.cas_urls"))),
]
