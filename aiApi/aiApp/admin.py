from django.contrib import admin
from aiApp.models import UserManagement, ImageInferProto, Yolov5Proto

# Register your models here.
admin.site.register(ImageInferProto)
admin.site.register(UserManagement)
admin.site.register(Yolov5Proto)