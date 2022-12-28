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
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import make_password
from django.db import models
from rest_framework import serializers
from rest_framework.permissions import IsAuthenticated

from authorize.models import (CustomUser, SessionProhibition,
                              SessionTransaction, UserManager)


# Authorize New User serializer
class AuthorizeNewUserRegisterSerializer(serializers.ModelSerializer):
    objects = UserManager()

    class Meta:
        model = CustomUser
        """
            username
            password
            email 
            domainid
            teamid
            staff
            admin
            Authorize_timestamp
            modified_timestamp
            comments
            status
            role
        """
        fields = (
            "username",
            "password",
            "email",
            "domainid",
            "teamid",
            "staff",
            "admin",
            "status",
            "role",
        )
        extra_kwargs = {
            "password": {"write_only": True, "min_length": 8},
        }


# Authorize User SessionTransaction serializer
class AuthorizeSessionTransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = SessionTransaction
        # sestransectionid customuserid domainid email actionitem transectionlock Authorize_timestamp comments modified
        fields = (
            "customuserid",
            "domainid",
            "email",
            "actionitem",
            "teamid",
            "fileid",
            "transectionlock",
            "comments",
            "modified",
        )

        def create(self, validated_data):
            # print("=============================ses-trsn:", validated_data)
            SessionTransaction = SessionTransaction.objects.create(
                customuserid=validated_data["customuserid"],
                domainid=validated_data["domainid"],
                email=validated_data["email"],
                actionitem=validated_data["actionitem"],
                transectionlock=validated_data["transectionlock"],
                comments=validated_data["comments"],
                modified=validated_data["modified"],
                fileid=validated_data["fileid"],
                teamid=validated_data["teamid"],
            )
            return SessionTransaction.objects.create(**validated_data)


# Authorize User Prohibition serializer
class AuthorizeSessionProhibitionSerializer(serializers.ModelSerializer):
    class Meta:
        model = SessionProhibition
        # sesprobhitionid customuserid domainid email sestoken Authorize_timestamp comments modified
        fields = (
            "customuserid",
            "domainid",
            "email",
            "comments",
            "modified",
            "is_blocked_status",
        )

        def create(self, validated_data):
            # print("=============================ses-trsn:", validated_data)
            SessionProhibition = SessionProhibition.objects.create(
                customuserid=validated_data["customuserid"],
                domainid=validated_data["domainid"],
                email=validated_data["email"],
                comments=validated_data["comments"],
                modified=validated_data["modified"],
                is_blocked_status=validated_data["is_blocked_status"],
            )
            return SessionProhibition.objects.create(**validated_data)
