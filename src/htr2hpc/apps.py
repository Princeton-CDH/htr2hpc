from django.apps import AppConfig


class Htr2HpcConfig(AppConfig):
    name = "htr2hpc"

    def ready(self):
        import htr2hpc.users  # noqa: F401 — registers admin overrides
