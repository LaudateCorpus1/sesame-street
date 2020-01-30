#!/bin/sh
mkdir .log 2> /dev/null
DEBUG=0 authbind gunicorn -b 0.0.0.0:8000 backend:app --access-logfile .log/access.log --error-logfile .log/general.log
