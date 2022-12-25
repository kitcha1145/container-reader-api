import json
import cv2
import rest_framework.status

from django.http.request import QueryDict
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from aiApp.serializers import UserSerializer, AiApiSerializer, ContainerSerializer
from aiApp.models import UserManagement, ImageInferProto
from rest_framework import serializers
import base64
from PIL import Image
import io
import numpy as np

from aiApi.wsgi import ml_model


class ImageInfer(APIView):
    # permission_classes = (IsAuthenticated,)

    def image_check(self, data):
        ret = False
        response = {}
        if data.get("image") is None:
            response = {
                "image": [
                    "This field is required."
                ],
            }
        else:
            ret = True
        return ret, response

    def user_update(self, request):
        try:
            qs = UserManagement.objects.get(name=request.user)
            serializer_g = UserSerializer(qs, many=False)

            data = {
                'name': str(request.user),
                'credit': serializer_g.data['credit'] + 1,
                'credit_limit': serializer_g.data['credit_limit']
            }
            Qd = QueryDict('', mutable=True)
            qs = UserManagement.objects.get(name=request.user)
            serializer_g = UserSerializer(qs, many=False)
            data['credit'] = serializer_g.data['credit'] + 1
            Qd.update(data)
            Qd.update(request.data)
            serializer = UserSerializer(qs, data=Qd)
            if serializer.is_valid():
                serializer.save()
                # return Response(serializer.data)
            else:
                print(serializer.errors)

        except Exception as err:
            data = {
                'name': str(request.user),
                'credit': 1,
                'credit_limit': 10
            }
            Qd = QueryDict('', mutable=True)
            Qd.update(data)
            Qd.update(request.data)
            serializer = UserSerializer(data=Qd)
            if serializer.is_valid():
                serializer.save()
            #     return Response(serializer.data)
            # return Response(serializer.errors)
            else:
                print(serializer.errors)

    def get(self, request, *args, **kwargs):
        if kwargs.get('id') is not None:
            try:
                qs = ImageInferProto.objects.get(id=kwargs.get('id'))
                serializer = AiApiSerializer(qs, many=False)
                return Response(serializer.data, status=rest_framework.status.HTTP_200_OK)
            except Exception as err:
                print(err)
                return Response(data={'error': str(err)}, status=rest_framework.status.HTTP_204_NO_CONTENT)

        else:
            qs = ImageInferProto.objects.all()
            serializer = AiApiSerializer(qs, many=True)
            return Response(serializer.data, status=rest_framework.status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        # print(request.data['image'].name)
        # print(request.data['mode'])
        # print(request.data['country'])
        # ml_model[]
        if kwargs.get('model') is not None:
            if ml_model.get(kwargs.get('model')) is not None or True:
                if kwargs.get('model') == 'container_reader':

                    ret, response = self.image_check(request.data)
                    if not ret:
                        return Response(response, status=rest_framework.status.HTTP_400_BAD_REQUEST)
                    data = {
                        'name': request.data['image'].name,
                        'results': "{}"
                    }

                    Qd = QueryDict('', mutable=True)
                    Qd.update(data)
                    Qd.update(request.data)
                    serializerA = ContainerSerializer(data=Qd)
                    if serializerA.is_valid():
                        # self.user_update(request)
                        img = request.data['image'].file.read()
                        img = Image.open(io.BytesIO(img))
                        npimg = np.array(img)
                        npimg = cv2.cvtColor(npimg, cv2.COLOR_BGRA2RGB)
                        if request.data.get('debug') is not None:
                            results = ml_model[kwargs.get('model')].predict(npimg,
                                                                            debug=True)
                            if isinstance(results, bytes):
                                return HttpResponse(results, content_type='image/jpeg')
                            else:
                                return Response(results, status=rest_framework.status.HTTP_201_CREATED)
                        else:
                            results = ml_model[kwargs.get('model')].predict(npimg)
                            data = {
                                'name': request.data['image'].name,
                                'image': request.data['image'],
                                'results': json.dumps(results)
                            }
                            return Response(results, status=rest_framework.status.HTTP_201_CREATED)
                    return Response(serializerA.errors, status=rest_framework.status.HTTP_400_BAD_REQUEST)
        return Response(status=rest_framework.status.HTTP_400_BAD_REQUEST)


class UserView(APIView):

    permission_classes = (IsAuthenticated, )

    def get(self, request, *args, **kwargs):
        try:
            try:
                qs = UserManagement.objects.get(name=request.user)
                serializer = UserSerializer(qs, many=False)
                return Response(serializer.data, status=rest_framework.status.HTTP_200_OK)
            except Exception as err:
                print(err)
                return Response(data={'error': str(err)}, status=rest_framework.status.HTTP_204_NO_CONTENT)
        except Exception as err:
            return Response(status=rest_framework.status.HTTP_400_BAD_REQUEST)

    # def post(self, request, *args, **kwargs):
    #     try:
    #         qs = UserManagement.objects.get(name=request.user)
    #         serializer_g = StudentSerializer(qs, many=False)
    #
    #         data = {
    #             'name': str(request.user),
    #             'credit': serializer_g.data['credit']+1,
    #             'credit_limit': serializer_g.data['credit_limit']
    #         }
    #         Qd = QueryDict('', mutable=True)
    #         Qd.update(data)
    #         Qd.update(request.data)
    #         serializer = StudentSerializer(qs, data=Qd)
    #         if serializer.is_valid():
    #             serializer.save()
    #             return Response(serializer.data)
    #         return Response(serializer.errors)
    #
    #     except Exception as err:
    #         data = {
    #             'name': str(request.user),
    #             'credit': 0,
    #             'credit_limit': 10
    #         }
    #         Qd = QueryDict('', mutable=True)
    #         Qd.update(data)
    #         Qd.update(request.data)
    #         serializer = StudentSerializer(data=Qd)
    #         if serializer.is_valid():
    #             serializer.save()
    #             return Response(serializer.data)
    #         return Response(serializer.errors)

