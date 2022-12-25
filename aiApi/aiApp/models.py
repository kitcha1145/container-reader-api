from django.db import models
import datetime

from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

# from .serializers import StudentSerializer


# Create your models here.
class Yolov5Proto(models.Model):
    name = models.CharField(max_length=100)
    image = models.BinaryField()
    results = models.CharField(max_length=10000)

    def __str__(self):
        return f'{self.name}'


class AlprProto(models.Model):
    name = models.CharField(max_length=100)
    mode = models.IntegerField()
    country = models.CharField(max_length=10)
    image = models.BinaryField()
    results = models.CharField(max_length=10000)

    def __str__(self):
        return f'{self.name}'


class ImageInferProto(models.Model):
    # image = models.FileField()
    name = models.CharField(max_length=100)
    id = models.IntegerField(primary_key=True, unique=True, blank=False, auto_created=True)
    image = models.BinaryField()

    def __str__(self):
        return f'{self.id}. {self.name}'


class UserManagement(models.Model):
    name = models.CharField(max_length=100)
    credit = models.IntegerField()
    credit_limit = models.IntegerField()

    # age = models.IntegerField()
    # start_time = models.DateTimeField(null=True, blank=True)
    created_on = models.DateTimeField(default=datetime.datetime.now)
    lastest_call = models.DateTimeField(auto_now=True)
    # description = models.TextField()
    # date_enrolled = models.DateTimeField(auto_now=True)
    # tracks = models.CharField(max_length=100)

    def __str__(self):
        return self.name


# from django.http.request import QueryDict
# for user in User.objects.all():
#     try:
#         qs = UserManagement.objects.get(name=user)
#         print(qs)
#     except Exception as err:
#         data = {
#             'name': user,
#             'count': 0,
#             'age': -1
#         }
#         Qd = QueryDict('', mutable=True)
#         Qd.update(data)
#         serializer = StudentSerializer(data=Qd)
#         if serializer.is_valid():
#             serializer.save()
    # print(user)
    # Token.objects.get_or_create
#     print(Token.objects.get_or_create(user=user))