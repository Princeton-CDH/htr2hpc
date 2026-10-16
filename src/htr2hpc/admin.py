from django.contrib import admin  # pragma: no cover
from django.contrib.auth import get_user_model  # pragma: no cover
from pucas.admin import CasUserAdmin  # pragma: no cover
from users.admin import MyUserAdmin  # pragma: no cover

from htr2hpc.models import HpcJobStats, SiteContent  # pragma: no cover


class Htr2HpcUserAdmin(CasUserAdmin, MyUserAdmin):  # pragma: no cover
    """Extends eScriptorium's UserAdmin with pucas CAS user management.

    Not covered by tests because this module depends on eScriptorium's MyUserAdmin
    which is not available in the test environment. If functionality is added
    here, update test settings to support it.
    """


# replace eScriptorium's User registration with our extended admin
admin.site.unregister(get_user_model())  # pragma: no cover
admin.site.register(get_user_model(), Htr2HpcUserAdmin)  # pragma: no cover


@admin.register(HpcJobStats)  # pragma: no cover
class HpcJobStatsAdmin(admin.ModelAdmin):  # pragma: no cover
    list_display = (
        "user",
        "job_type",
        "slurm_job_id",
        "max_memory_mb",
        "avg_memory_mb",
        "elapsed_seconds",
        "gpu_utilization",
        "scratch_storage_mb",
        "created_at",
    )
    list_filter = ("job_type", "user")
    search_fields = ("user__username", "slurm_job_id")
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)


@admin.register(SiteContent)  # pragma: no cover
class SiteContentAdmin(admin.ModelAdmin):  # pragma: no cover
    list_display = ("key", "label", "updated_at")
    search_fields = ("key", "label", "body")
    readonly_fields = ("updated_at",)
    ordering = ("key",)
