"""
WSGI config for aiApi project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
from apps.ml.detection import container_reader

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aiApi.settings')

application = get_wsgi_application()

import traceback

ml_model = {}
try:
    ml_model["container_reader"] = container_reader()


except Exception as e:
    traceback.print_exc()
    # print("Exception while loading the algorithms to the registry,", str(e))
