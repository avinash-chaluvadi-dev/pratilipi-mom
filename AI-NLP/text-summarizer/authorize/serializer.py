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
from django.contrib.auth import authenticate, get_user_model
from django.db import models
from rest_framework import serializers
from rest_framework.permissions import IsAuthenticated

from authorize.models import CustomUser, UserManager

from .models import UserRoles


# Authorize New User serializer
class AuthorizeNewUserRegisterSerializer(serializers.ModelSerializer):
    objects = UserManager()

    class Meta:
        model = CustomUser
        fields = "__all__"

        extra_kwargs = {
            "password": {"write_only": True, "min_length": 8},
        }

    def create(self, validated_data):
        return CustomUser.objects.create(**validated_data)


class RoleSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserRoles
        fields = "__all__"


class LoginSerializer(serializers.Serializer):
    email = serializers.CharField(max_length=255)
    name = serializers.CharField(read_only=True)
    password = serializers.CharField(max_length=128, write_only=True)
    token = serializers.CharField(max_length=255, read_only=True)
    role = serializers.DictField(read_only=True)

    def validate(self, data):
        email = data.get("email", None)
        password = data.get("password", None)

        if email is None:
            raise serializers.ValidationError("An email address is required to log in.")

        if password is None:
            raise serializers.ValidationError("A password is required to log in.")

        user = authenticate(username=email, password=password)

        if user is None:
            raise serializers.ValidationError(
                "A user with this email and password was not found."
            )

        if not user.is_active:
            raise serializers.ValidationError("This user has been deactivated.")
        data["token"] = user.token
        data["name"] = user.name
        data["role"] = RoleSerializer(user.role).data
        return data
