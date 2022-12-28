# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application views***
    @Description    : 
    @Date           : 07-10-2021
    @Last Modified  : 12-10-2021
"""

import logging as lg
import time

from django.contrib.auth.hashers import check_password
from django.core.exceptions import ValidationError
from django.http import HttpResponse
from django.shortcuts import render
from rest_framework import generics, mixins, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from authorize.models import (CustomUser, SessionProhibition,
                              SessionTransaction, UserRoles)
from authorize.serializer import (AuthorizeNewUserRegisterSerializer,
                                  AuthorizeSessionProhibitionSerializer,
                                  AuthorizeSessionTransactionSerializer)
from boiler_plate.utility import utils

logger = lg.getLogger("file")
import json


# Authorize new user
class AuthorizeNewUser(generics.GenericAPIView):
    serializer_class = AuthorizeNewUserRegisterSerializer

    def post(self, request, *args, **kwargs):
        try:
            result = {"status": status.HTTP_201_CREATED}

            if len(request.data) > 0:
                # print("-"*100)
                # print("data", request.data)
                # print("-"*100)

                if request.data.get("role") == None or (
                    UserRoles.objects.values("role_type")
                    .filter(role_type=request.data.get("role"))
                    .count()
                    == 0
                ):
                    result["data"] = {
                        "error": "Can you please provide the role type details!, Ex('admin','qc-admin','etc.')",
                    }
                    result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                    return HttpResponse(
                        status=result["status"],
                        content=json.dumps(result["data"]),
                    )
                elif (
                    request.data.get("email") == None
                    or len(request.data.get("email")) == 0
                ):
                    result["data"] = {
                        "error": "Can you please provide the email details!",
                    }
                    result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                    return HttpResponse(
                        status=result["status"],
                        content=json.dumps(result["data"]),
                    )
                elif (
                    request.data.get("teamid") == None
                    or len(request.data.get("teamid")) == 0
                ):
                    result["data"] = {
                        "error": "Can you please provide the team id details!",
                    }
                    result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                    return HttpResponse(
                        status=result["status"],
                        content=json.dumps(result["data"]),
                    )
                elif (
                    request.data.get("domainid") == None
                    or len(request.data.get("domainid")) == 0
                ):
                    result["data"] = {
                        "error": "Can you please provide the domain id details!",
                    }
                    result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                    return HttpResponse(
                        status=result["status"],
                        content=json.dumps(result["data"]),
                    )
                else:
                    # print("REQUEST DATA::", request.data['role'], UserRoles.objects.values('id').filter(role_type=request.data.get('role'))[0].get('id'))
                    # align-role:
                    if (
                        UserRoles.objects.values("id")
                        .filter(role_type=request.data.get("role"))[0]
                        .get("id")
                        != None
                    ):
                        request.data["role"] = (
                            UserRoles.objects.values("id")
                            .filter(role_type=request.data.get("role"))[0]
                            .get("id")
                        )

                    serializer = self.get_serializer(data=request.data)
                    # print("data::serializer::",serializer)
                    serializer.is_valid(raise_exception=True)
                    user = serializer.save()
                    user.set_password(user.password)
                    user.save()
                    result["data"] = {
                        "user": AuthorizeNewUserRegisterSerializer(
                            user, context=self.get_serializer_context()
                        ).data,
                        "message": "New user has been created successfully !",
                    }
                    result["status"] = status.HTTP_201_CREATED
            else:
                result["data"] = {
                    "error": "Can you please provide the required details to register new user to access pratilipi tool (Email,Role Id, etc..)!"
                }
                result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                return HttpResponse(
                    status=result["status"], content=json.dumps(result["data"])
                )
                logger.warning(f"Internal server errors :: AuthorizeNewUser")

        except Exception as error:
            result["data"] = {
                "error": f"Something went wrong while processing new user request pratilipi tool ! Exception {error}"
            }
            result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
            logger.error(f"Exception :: AuthorizeNewUser :: str{error}")

        return HttpResponse(status=result["status"], content=json.dumps(result["data"]))


# Authorize login user
class GetAuthorizeuserSessionToken(generics.GenericAPIView):
    serializer_class = AuthorizeSessionTransactionSerializer

    def post(self, request, *args, **kwargs):
        try:
            result = {"status": status.HTTP_201_CREATED}
            if len(request.data) > 0:

                email = request.data.get("email")
                password = request.data.get("password")

                if (
                    request.data.get("email") == None
                    or len(request.data.get("email")) == 0
                ):
                    result["data"] = {
                        "error": "Can you please provide the register email id to access the tool!",
                    }
                    result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                    return HttpResponse(
                        status=result["status"],
                        content=json.dumps(result["data"]),
                    )

                elif (
                    request.data.get("password") == None
                    or len(request.data.get("password")) == 0
                ):
                    result["data"] = {
                        "error": "Can you please provide the password details to access the tool!",
                    }
                    result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                    return HttpResponse(
                        status=result["status"],
                        content=json.dumps(result["data"]),
                    )

                if CustomUser.objects.filter(email=email).exists():
                    dictionary = CustomUser.objects.filter(email=email).values()[0]

                    print("db:data:", dictionary)

                    if CustomUser.objects.filter(
                        email=email
                    ).exists() and check_password(password, dictionary["password"]):

                        id = dictionary["id"]

                        if id != None:
                            user_role_dtls = UserRoles.objects.values(
                                "id", "role_type"
                            ).filter(id=dictionary["role_id"])[0]
                            # print("==========ROLE", user_role_dtls.get('id'),user_role_dtls.get('role_type'))

                        # Tokens::
                        token_encoded_jwt = utils.get_sestokens(
                            dictionary["email"], dictionary["domainid"]
                        )
                        time.sleep(0.5)
                        accessToken = utils.get_sestokens(
                            dictionary["email"], dictionary["domainid"]
                        )
                        runtimedictionary = CustomUser.objects.filter(
                            email=email, id=id
                        ).values()[0]

                        # print("===========>>>", runtimedictionary)

                        """
                            # -------- update teamId and otherId
                            # -------- update teamId and otherId
                            # -------- update teamId and otherId
                        """
                        request.data["teamid"] = dictionary["teamid_id"]
                        request.data["fileid"] = None
                        sesobject = utils.set_sessiontransection_dtls(
                            request.data, runtimedictionary, "login"
                        )
                        sesserializer = self.get_serializer(data=sesobject)
                        sesserializer.is_valid(raise_exception=True)
                        sessionTransaction = sesserializer.save()
                        sessionTransaction.save()

                        result["data"] = {
                            "token": token_encoded_jwt,
                            "access_token": accessToken,
                            "email": email,
                            "id": id,
                            "role_id": user_role_dtls.get("id"),
                            "role_type": user_role_dtls.get("role_type"),
                            "message": "You Have Successfully Logged in to Pratilipi !",
                        }
                        result["status"] = status.HTTP_201_CREATED

                    else:
                        result["data"] = {
                            "message": "Invalid User Information Provided Please Check with Pratilipi System Administrator !",
                            "email": email,
                        }
                        result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                else:
                    result["data"] = {
                        "message": "Invalid User Information Provided Please Check with Pratilipi System Administrator !",
                        "email": email,
                    }
                    result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR

            else:
                result["data"] = {
                    "error": "Can you please provide the required details access/login pratilipi tool (Email,Password)!"
                }
                result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                logger.warning(
                    f"Internal server errors :: GetAuthorizeuserSessionToken"
                )

        except Exception as error:
            import traceback

            print(traceback.format_exc())
            result["data"] = {
                "error": f"Something went wrong while processing user login api @pratilipi tool ! Exception {error}"
            }
            result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
            logger.error(f"Exception :: GetAuthorizeuserSessionToken :: str{error}")

        return HttpResponse(status=result["status"], content=json.dumps(result["data"]))


# Authorize refresh user / generate new token
class GetAuthorizeuserSessionRrefreshToken(generics.GenericAPIView):

    serializer_class = AuthorizeSessionTransactionSerializer

    def post(self, request, *args, **kwargs):

        try:

            result = {"status": status.HTTP_201_CREATED}
            if len(request.data) > 0:
                email = request.data.get("email")
                password = request.data.get("password")

                if (
                    request.data.get("email") == None
                    or len(request.data.get("email")) == 0
                ):
                    result["data"] = {
                        "error": "Can you please provide the register email id to access the tool!",
                    }
                    result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                    return HttpResponse(
                        status=result["status"],
                        content=json.dumps(result["data"]),
                    )

                elif (
                    request.data.get("password") == None
                    or len(request.data.get("password")) == 0
                ):
                    result["data"] = {
                        "error": "Can you please provide the password details to generate the tool refresh token!",
                    }
                    result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                    return HttpResponse(
                        status=result["status"],
                        content=json.dumps(result["data"]),
                    )

                if CustomUser.objects.filter(email=email).exists():
                    dictionary = CustomUser.objects.filter(email=email).values()[0]
                    # print("db:data:", dictionary)

                    if CustomUser.objects.filter(
                        email=email
                    ).exists() and check_password(password, dictionary["password"]):

                        id = dictionary["id"]

                        if id != None:
                            user_role_dtls = UserRoles.objects.values(
                                "id", "role_type"
                            ).filter(id=dictionary["role_id"])[0]
                            # print("==========ROLE", user_role_dtls.get('id'),user_role_dtls.get('role_type'))

                        runtimedictionary = CustomUser.objects.filter(
                            email=email, id=dictionary["id"]
                        ).values()[0]

                        # set- session transection table: transectionlock = 0
                        # 1 : lock active
                        # 0 : lock released
                        SessionTransaction.objects.filter(
                            email=email,
                            customuserid_id=dictionary["id"],
                            domainid=dictionary["domainid"],
                            transectionlock=1,
                        ).update(transectionlock=0)

                        # Refresh-Tokens::
                        token_encoded_jwt = utils.get_sestokens(
                            dictionary["email"], dictionary["domainid"]
                        )
                        time.sleep(0.5)
                        accessToken = utils.get_sestokens(
                            dictionary["email"], dictionary["domainid"]
                        )

                        refreshruntimedictionary = CustomUser.objects.filter(
                            email=email, id=id
                        ).values()[0]

                        request.data["teamid"] = dictionary["teamid_id"]
                        request.data["fileid"] = None
                        sesobject = utils.set_sessiontransection_dtls(
                            request.data, refreshruntimedictionary, "block"
                        )
                        sesserializer = self.get_serializer(data=sesobject)
                        sesserializer.is_valid(raise_exception=True)
                        sessionTransaction = sesserializer.save()
                        sessionTransaction.save()

                        result["data"] = {
                            "token": token_encoded_jwt,
                            "access_token": accessToken,
                            "email": email,
                            "id": id,
                            "role_id": user_role_dtls.get("id"),
                            "role_type": user_role_dtls.get("role_type"),
                            "message": "You have successfully registered with new token. You have permissions enabled to access Pratilipi tool !",
                        }
                        result["status"] = status.HTTP_201_CREATED

                    else:
                        result["data"] = {
                            "email": email,
                            "message": "Invalid User Information Provided Please Check with Pratilipi System Administrator !",
                        }
                        result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                else:
                    result["data"] = {
                        "error": "Can you please provide the required details to get login refresh token for pratilipi tool (Email,Password)!"
                    }
                    result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                    logger.warning(
                        f"Internal server errors :: GetAuthorizeuserSessionRrefreshToken"
                    )
            else:
                result["data"] = {
                    "error": "Can you please provide the required details to get login refresh token for pratilipi tool (Email,Password)!"
                }
                result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                logger.warning(
                    f"Internal server errors :: GetAuthorizeuserSessionRrefreshToken"
                )

        except Exception as error:
            result["data"] = {
                "error": f"Something went wrong while processing user login api @pratilipi tool ! Exception {error}"
            }
            result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
            logger.error(
                f"Exception :: GetAuthorizeuserSessionRrefreshToken :: str{error}"
            )

        return HttpResponse(status=result["status"], content=json.dumps(result["data"]))


# Authorize logout user
class SetAuthorizeSessionProhibition(generics.GenericAPIView):
    serializer_class = AuthorizeSessionProhibitionSerializer

    def post(self, request, *args, **kwargs):
        try:

            result = {"status": status.HTTP_201_CREATED}
            if len(request.data) > 0:
                email = request.data.get("email")
                sestoken = request.data.get("sestoken")

                if (
                    request.data.get("email") == None
                    or len(request.data.get("email")) == 0
                ):
                    result["data"] = {
                        "error": "Can you please provide the register email id to access the tool!",
                    }
                    result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                    return HttpResponse(
                        status=result["status"],
                        content=json.dumps(result["data"]),
                    )

                if CustomUser.objects.filter(email=email).exists():
                    dictionary = CustomUser.objects.filter(email=email).values()[0]
                    # print("db:data:", dictionary)
                    if CustomUser.objects.filter(email=email).exists():
                        runtimedictionary = CustomUser.objects.filter(
                            email=email, id=dictionary["id"]
                        ).values()[0]

                        # set- session transection table: transectionlock = 0
                        # 1 : lock active
                        # 0 : lock released
                        SessionTransaction.objects.filter(
                            email=email,
                            customuserid_id=dictionary["id"],
                            domainid=dictionary["domainid"],
                            transectionlock=1,
                        ).update(transectionlock=0)
                        time.sleep(0.5)
                        sesprohibitionobject = utils.set_session_prohibition_dtls(
                            request.data, runtimedictionary
                        )
                        sesserializer = self.get_serializer(data=sesprohibitionobject)
                        sesserializer.is_valid(raise_exception=True)
                        SessionProhibition = sesserializer.save()
                        SessionProhibition.save()

                        result["data"] = {
                            "message": "You have been logged out!",
                            "email": email,
                        }
                        result["status"] = status.HTTP_201_CREATED

                    else:
                        result["data"] = {
                            "email": email,
                            "error": "Invalid User Information Provided Please Check with Pratilipi System Administrator !",
                        }
                        result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                        logger.warning(
                            f"Internal server errors :: SetAuthorizeSessionProhibition"
                        )
                else:
                    result["data"] = {
                        "email": email,
                        "error": "Invalid User Information Provided Please Check with Pratilipi System Administrator !",
                    }
                    result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                    logger.warning(
                        f"Internal server errors :: SetAuthorizeSessionProhibition"
                    )
            else:
                result["data"] = {
                    "error": "Can you please provide the required details to logout pratilipi tool (Email,Token, etc.)!"
                }
                result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                logger.warning(
                    f"Internal server errors :: SetAuthorizeSessionProhibition"
                )

        except Exception as error:
            result["data"] = {
                "error": f"Something went wrong while processing user login api @pratilipi tool ! Exception {error}"
            }
            result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
            logger.error(f"Exception :: SetAuthorizeSessionProhibition :: str{error}")

        return HttpResponse(status=result["status"], content=json.dumps(result["data"]))
