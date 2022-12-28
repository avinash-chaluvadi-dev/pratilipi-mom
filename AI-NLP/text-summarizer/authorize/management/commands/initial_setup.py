from django.core.management.base import BaseCommand

from authorize.models import UserRoles


def setup_roles():
    UserRoles.objects.get_or_create(
        role_type="sme",
        upload=True,
        mom=True,
        share_mom=True,
        dashboard=True,
        home_screen="upload",
    )
    UserRoles.objects.get_or_create(
        role_type="business_user",
        dashboard=True,
        qc=True,
        home_screen="dashboard",
    )
    UserRoles.objects.get_or_create(
        role_type="qc_admin",
        qc=True,
        configuration=True,
        home_screen="configuration",
    )
    UserRoles.objects.get_or_create(
        role_type="admin",
        upload=True,
        mom=True,
        share_mom=True,
        dashboard=True,
        qc=True,
        configuration=True,
        home_screen="dashboard",
    )


class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        print("Starting roles setup..")
        setup_roles()
        print("Roles setup success!")
