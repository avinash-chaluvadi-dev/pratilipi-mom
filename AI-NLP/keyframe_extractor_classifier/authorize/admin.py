# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application views***
    @Description    : The django :: core :: AbstractUser vs AbstractBaseUser
                        There are two modern ways to create a custom user model in Django: AbstractUser and AbstractBaseUser. 
                        In both cases we can subclass them to extend existing functionality however AbstractBaseUser requires much, much more work. 
                        Seriously, don't mess with it unless we're really know what we're doing. 
                        And if we did, we wouldn't be require for HIVE prithilipi tool, because we are yet to bring out LDAP interation.Author ~AH40222.

                        FYI, hence we'll use AbstractUser which actually subclasses AbstractBaseUser but provides more default configuration.

    @Date           : 07-10-2021
    @Last Modified  : 12-10-2021

"""
"""Register models to django admin."""

from django.contrib import admin
from django.contrib.auth.models import Group

from authorize.models import CustomUser, SessionProhibition, SessionTransaction


class UserAdmin(admin.ModelAdmin):
    """Customize user/admin view on djano admin."""

    search_fields = ("email", "username")
    list_display = ("email", "is_active")
    list_filter = ("is_active", "is_active")
    ordering = ("email",)

    fieldsets = (
        (None, {"fields": ("email", "password")}),
        ("Permissions", {"fields": ("is_admin", "is_staff")}),
        (
            "Primary personal information",
            {"fields": ("username",)},
        ),
        ("is_active", {"fields": ("is_active",)}),
    )


# authorize : customuser model registred : Pratilipi tool.
admin.site.register(CustomUser, UserAdmin)
admin.site.register(SessionTransaction)
admin.site.register(SessionProhibition)
