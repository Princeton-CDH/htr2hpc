from typing import ClassVar

from django import forms
from django.contrib import admin, messages
from django.contrib.auth import get_user_model
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path
from django.utils.translation import ngettext
from pucas.ldap import LDAPSearch, LDAPSearchException, user_info_from_ldap
from users.admin import MyUserAdmin


def init_new_user(user, user_info):
    """pucas EXTRA_USER_INIT hook: make new CAS accounts inactive by default."""
    # admins must explicitly activate accounts before users can log in
    user.is_active = False


class CasUserInitForm(forms.Form):
    """Form to initialize one or more CAS user accounts by netid."""

    netids = forms.CharField(
        label="NetIDs",
        help_text="Enter one or more Princeton NetIDs, separated by spaces or newlines.",
        widget=forms.Textarea(attrs={"rows": 4}),
    )

    def clean_netids(self):
        return self.cleaned_data["netids"].split()


class Htr2HpcUserAdmin(MyUserAdmin):
    """Extends eScriptorium's UserAdmin with CAS user management."""

    actions: ClassVar = [*MyUserAdmin.actions, "activate"]

    def activate(self, request, queryset):
        """Admin action to activate selected user accounts."""
        updated = queryset.update(is_active=True)
        self.message_user(
            request,
            ngettext(
                "%d user activated.",
                "%d users activated.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    activate.short_description = "Activate selected users"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "cas-init/",
                self.admin_site.admin_view(self.cas_user_init),
                name="users_user_cas_init",
            ),
        ]
        return custom_urls + urls

    def cas_user_init(self, request):
        """View to initialize CAS user accounts by netid."""
        User = get_user_model()

        if request.method == "POST":
            form = CasUserInitForm(request.POST)
            if form.is_valid():
                netids = form.cleaned_data["netids"]
                created_users = []
                errors = []

                ldap = LDAPSearch()
                for netid in netids:
                    try:
                        # verify netid exists in LDAP before creating a DB record
                        ldap.find_user(netid)
                        user, created = User.objects.get_or_create(username=netid)
                        # only init new users; skip existing to avoid resetting is_active
                        if created:
                            user_info_from_ldap(user)
                        created_users.append((netid, created))
                    except LDAPSearchException:  # noqa: PERF203
                        errors.append(netid)

                if created_users:
                    created = [n for n, c in created_users if c]
                    existing = [n for n, c in created_users if not c]
                    if created:
                        self.message_user(
                            request,
                            "Created accounts: {}".format(", ".join(created)),
                            messages.SUCCESS,
                        )
                    if existing:
                        self.message_user(
                            request,
                            "Already exists: {}".format(", ".join(existing)),
                            messages.INFO,
                        )
                if errors:
                    self.message_user(
                        request,
                        "NetIDs not found in LDAP: {}".format(", ".join(errors)),
                        messages.ERROR,
                    )

                return redirect("..")
        else:
            form = CasUserInitForm()

        context = dict(
            self.admin_site.each_context(request),
            form=form,
            opts=self.model._meta,
            title="Initialize CAS Users",
        )
        return TemplateResponse(
            request,
            "admin/users/user/cas_user_init.html",
            context,
        )

    def changelist_view(self, request, extra_context=None):
        if extra_context is None:
            extra_context = {}
        extra_context["cas_init_url"] = "cas-init/"
        return super().changelist_view(request, extra_context=extra_context)


# replace eScriptorium's User registration with our extended admin
admin.site.unregister(get_user_model())
admin.site.register(get_user_model(), Htr2HpcUserAdmin)
