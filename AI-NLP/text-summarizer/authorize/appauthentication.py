import logging as lg
import time

import jwt
from django.conf import settings
from django.contrib.auth import get_user_model
from rest_framework import authentication, exceptions

logger = lg.getLogger("file")


class AppAuthentication(authentication.BaseAuthentication):

    authentication_header_prefix = "Token"

    def authenticate(self, request):

        request.user = None
        """
            The auth_header element via request header should have 2 key elementes
            1: Token
            2: JWT authenticate token
        """
        auth_header = authentication.get_authorization_header(request).split()
        auth_header_prefix = self.authentication_header_prefix.lower()

        if not auth_header:
            return None

        if len(auth_header) == 1 or len(auth_header) > 2:
            return None

        prefix = auth_header[0].decode("utf-8")
        token = auth_header[1].decode("utf-8")

        if prefix.lower() != auth_header_prefix:
            return None

        return self._authenticate_credentials(request, token)

    def _authenticate_credentials(self, request, token):

        user = get_user_model()
        """
        - Try to authenticate the given credentials.
        - If authentication is successful, return the user and token.
        - If not, throw an error.
        """
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])

        except jwt.DecodeError:
            msg = "Invalid User authentication. Could not decode token."
            raise exceptions.AuthenticationFailed(msg)

        try:
            user = user.objects.get(email=payload["email"])

        except user.DoesNotExist:
            msg = "No user matching this token was found."
            raise exceptions.AuthenticationFailed(msg)

        if not user.is_active:
            msg = "This user has been deactivated."
            raise exceptions.AuthenticationFailed(msg)

        return (user, token)
