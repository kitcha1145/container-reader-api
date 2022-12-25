from rest_framework import serializers
from aiApp.models import UserManagement, ImageInferProto, Yolov5Proto, AlprProto


class UserSerializer(serializers.ModelSerializer):

    class Meta:
        model = UserManagement
        fields = ('name', 'credit', 'created_on', 'lastest_call', 'credit_limit')


class AiApiSerializer(serializers.ModelSerializer):

    class Meta:
        model = ImageInferProto
        fields = ('image', 'name', )


class YoloV5Serializer(serializers.ModelSerializer):

    class Meta:
        model = Yolov5Proto
        fields = ('image', 'name', 'results')


class AlprSerializer(serializers.ModelSerializer):

    class Meta:
        model = AlprProto
        fields = ('image', 'mode', 'country', 'name', 'results')

class ContainerSerializer(serializers.ModelSerializer):
    class Meta:
        model = AlprProto
        fields = ('image', 'mode', 'results')