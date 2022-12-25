#!/bin/sh

#python3.8 manage.py makemigrations --merge
#python3.8 manage.py migrate --noinput
#python3.8 manage.py collectstatic --noinput
python3.8 manage.py runserver $HOST:$API_PORT