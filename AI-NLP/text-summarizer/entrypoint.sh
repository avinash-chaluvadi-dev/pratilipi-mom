#!/bin/sh
SETTINGS="sense_maker.settings.base"
echo $SETTINGS
if [ "$1" = "uat" ]; then
    SETTINGS="sense_maker.settings.uat";
fi;
export DJANGO_SETTINGS_MODULE=$SETTINGS
python manage.py migrate --settings=$SETTINGS --no-input
gunicorn sense_maker.wsgi:application --bind 0.0.0.0:8080 $GUNICORN_OPTNS >> /logs/$APP_NAME.$host.log 2>&1