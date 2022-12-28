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
import jwt
from django.conf import settings
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.db import migrations, models


class UserManager(BaseUserManager):
    def _create_user(self, email, password, is_staff, is_superuser, **extra_fields):
        """
        Creates and saves a User with the given email and password.
        """
        if not email:
            raise ValueError("The given email must be set")
        email = self.normalize_email(email)
        CustomUser = self.model(
            email=email,
            is_staff=is_staff,
            is_active=True,
            is_superuser=is_superuser,
            **extra_fields
        )
        CustomUser.set_password(password)
        CustomUser.save(using=self._db)
        return CustomUser

    def create_user(self, email, password=None):
        """
        Creates and saves a User with the given email and password.
        """
        if not email:
            raise ValueError("Users must have an email address")

        CustomUser = self.model(
            email=self.normalize_email(email),
        )
        # print("create_superuser::=================>", password)
        CustomUser.set_password(password)
        CustomUser.save(using=self._db)
        return CustomUser

    def create_superuser(self, email, password):
        """
        Creates and saves a superuser with the given email and password.
        """
        # print("create_superuser::=================>", password)
        CustomUser = self.create_user(
            email,
            password=password,
        )
        CustomUser.staff = True
        CustomUser.admin = True
        CustomUser.save(using=self._db)
        return CustomUser


class UserRoles(models.Model):
    role_type = models.CharField(max_length=250, unique=True)
    role_description = models.CharField(max_length=250, blank=True, null=True)
    upload = models.BooleanField(default=False)
    mom = models.BooleanField(default=False)
    share_mom = models.BooleanField(default=False)
    dashboard = models.BooleanField(default=False)
    configuration = models.BooleanField(default=False)
    qc = models.BooleanField(default=False)
    home_screen = models.CharField(max_length=50)

    class Meta:
        db_table = "api_roles"
        ordering = ["-id"]


class CustomUser(AbstractBaseUser):
    """
    The customized abstract base user cusotomuser class
    this class mainly deails with Pratilipi tool user registration purpose,session management purpose, token action
    and other concurence prevention mechanisms.
    """

    email = models.EmailField(
        verbose_name="email address",
        max_length=100,
        unique=True,
    )
    USERNAME_FIELD = "email"
    team = models.ForeignKey(
        "rest_api.Team",
        related_name="users",
        on_delete=models.CASCADE,
    )
    name = models.CharField(max_length=25, blank=True, null=True)
    authorize_timestamp = models.DateTimeField(auto_now=False, auto_now_add=True)
    modified = models.DateField(null=True)
    comments = models.CharField(max_length=250, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    role = models.ForeignKey(
        UserRoles,
        related_name="users",
        on_delete=models.CASCADE,
    )

    class Meta:
        db_table = "api_customuser"
        ordering = ["-id"]

    # add additional fields in here
    def __str__(self):
        return self.email.__str__()

    objects = UserManager()
    REQUIRED_FIELDS = []

    @property
    def token(self):
        return self._generate_jwt_token()

    def _generate_jwt_token(self):
        token = jwt.encode(
            {"email": self.email},
            settings.SECRET_KEY,
            algorithm="HS256",
        )
        return token

    @property
    def is_sme(self):
        return self.role.role_type == "sme"
