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
    role_type = models.CharField(max_length=250, blank=True, null=True)
    login = models.CharField(max_length=2, blank=True, null=True)
    file_upload = models.CharField(max_length=2, blank=True, null=True)
    human_feedback_loop = models.CharField(max_length=2, blank=True, null=True)
    share_mom = models.CharField(max_length=2, blank=True, null=True)
    insights_dashboard = models.CharField(max_length=2, blank=True, null=True)
    config_screen = models.CharField(max_length=2, blank=True, null=True)
    qc_dashboard = models.CharField(max_length=2, blank=True, null=True)

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
        if self.email != None:
            return self.email.__str__()
        return

    objects = UserManager()
    REQUIRED_FIELDS = []

    # add additional fields in here
    def __str__(self):
        if self.email != None:
            return self.email.__str__()
        return

    def has_perm(self, perm, obj=None):
        "Does the user have a specific permission?"
        # Simplest possible answer: Yes, always
        return True

    def has_module_perms(self, app_label):
        "Does the user have permissions to view the app `app_label`?"
        # Simplest possible answer: Yes, always
        return True

    @property
    def is_staff(self):
        "Is the user a member of staff?"
        return self.staff

    @property
    def is_admin(self):
        "Is the user a admin member?"
        return self.admin


class SessionTransaction(models.Model):
    """
    this class mainly deails with Pratilipi tool user transection mechanism.
    by refering this model we can try to prevent the user concurency and other session network distrabences.

    #sestransectionid customuserid domainid email sestoken actionitem transectionlock authorize_timestamp comments modified
    """

    sestransectionid = models.AutoField(primary_key=True, null=False, unique=True)
    customuserid = models.ForeignKey(
        CustomUser,
        related_name="ses_tran_customUser",
        on_delete=models.CASCADE,
    )
    domainid = models.CharField(max_length=5000, blank=True, null=True)
    email = models.EmailField(
        verbose_name="email address",
        max_length=100,
        blank=True,
        null=True,
    )
    actionitem = models.CharField(max_length=50, blank=True, null=True)
    fileid = models.IntegerField(null=True)
    teamid = models.IntegerField(null=True)
    transectionlock = models.BooleanField(default=True)
    authorize_timestamp = models.DateTimeField(auto_now=False, auto_now_add=True)
    comments = models.CharField(max_length=250, blank=True, null=True)
    modified = models.DateField(null=True)

    class Meta:
        db_table = "api_session_transaction"
        ordering = ["-sestransectionid"]

    # add additional fields in here
    def __int__(self):
        if self.sestransectionid != None:
            return self.sestransectionid.__int__()
        return


class SessionProhibition(models.Model):
    """
    this class mainly deails with Pratilipi tool user transection mechanism to prohibit the user access, logout, session out, etc.
    by refering this model we can try to prevent user if he logouted and other session network distrabences.

    #sesprobhitionid customuserid domainid email sestoken authorize_timestamp comments modified
    """

    sesprobhitionid = models.AutoField(primary_key=True, null=False, unique=True)
    customuserid = models.ForeignKey(
        CustomUser,
        related_name="ses_proh_customUser",
        on_delete=models.CASCADE,
    )
    domainid = models.CharField(max_length=5000, blank=True, null=True)
    email = models.EmailField(
        verbose_name="email address",
        max_length=100,
        blank=True,
        null=True,
    )
    authorize_timestamp = models.DateTimeField(auto_now=False, auto_now_add=True)
    comments = models.CharField(max_length=250, blank=True, null=True)
    modified = models.DateField(null=True)
    is_blocked_status = models.BooleanField(default=True)

    class Meta:
        db_table = "api_session_prohibition"
        ordering = ["-sesprobhitionid"]

    # add additional fields in here
    def __int__(self):
        if self.sesprobhitionid != None:
            return self.sesprobhitionid.__int__()
        return
