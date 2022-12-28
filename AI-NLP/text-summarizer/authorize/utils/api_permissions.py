from rest_framework.permissions import SAFE_METHODS, BasePermission

from authorize.models import UserRoles


class FileUploadAPIPermission(BasePermission):
    def has_permission(self, request, view):
        return (
            request.user.is_authenticated
            and UserRoles.objects.filter(pk=request.user.role.id, upload=True).exists()
        )


class FeedbackLoopAPIPermission(BasePermission):
    def has_permission(self, request, view):
        return (
            request.user.is_authenticated
            and UserRoles.objects.filter(pk=request.user.role.id, mom=True).exists()
        )


class MoMAPIPermission(BasePermission):
    def has_permission(self, request, view):
        return (
            request.user.is_authenticated
            and UserRoles.objects.filter(
                pk=request.user.role.id, share_mom=True
            ).exists()
        )


class InsightDashboardAPIPermission(BasePermission):
    def has_permission(self, request, view):
        return (
            request.user.is_authenticated
            and UserRoles.objects.filter(
                pk=request.user.role.id, dashboard=True
            ).exists()
        )


class ConfigScreenAPIPermission(BasePermission):
    def has_permission(self, request, view):
        return request.user.is_authenticated and (
            UserRoles.objects.filter(
                pk=request.user.role.id, configuration=True
            ).exists()
            or request.method in SAFE_METHODS
        )


class QCDashboardAPIPermission(BasePermission):
    def has_permission(self, request, view):
        return (
            request.user.is_authenticated
            and UserRoles.objects.filter(pk=request.user.role.id, qc=True).exists()
        )
