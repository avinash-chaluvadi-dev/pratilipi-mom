from rest_framework import generics, status
from rest_framework.response import Response


class Ping(generics.GenericAPIView):
    def get(self, request, *args, **kwargs):
        return Response(status=status.HTTP_200_OK, data={"status": "Alive"})
