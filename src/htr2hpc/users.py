from django import forms
from django.contrib import admin, messages
from django.contrib.auth import get_user_model
from django.core.validators import RegexValidator
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path

from pucas.ldap import LDAPSearchException, init_cas_user
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
        if request.method == "POST":
            form = CasUserInitForm(request.POST)
            if form.is_valid():
                netids = form.cleaned_data["netids"]
                created_list = []
                existing_list = []
                errors = []

                for netid in netids:
                    try:
                        _, created = init_cas_user(netid)
                        if created:
                            created_list.append(netid)
                        else:
                            existing_list.append(netid)
                    except LDAPSearchException:
                        errors.append(netid)

                if created_list:
                    self.message_user(
                        request,
                        "Created accounts: %s" % ", ".join(created_list),
                        messages.SUCCESS,
                    )
                if existing_list:
                    self.message_user(
                        request,
                        "Already exists: %s" % ", ".join(existing_list),
                        messages.INFO,
                    )
                if errors:
                    self.message_user(
                        request,
                        "NetIDs not found in LDAP: %s" % ", ".join(errors),
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
