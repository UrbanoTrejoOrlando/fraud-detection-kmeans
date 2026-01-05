#!/usr/bin/env bash
# build.sh

# Salir en caso de error
set -o errexit

# Instalar dependencias
pip install -r requirements.txt

# Colectar archivos estáticos
python manage.py collectstatic --noinput

# Aplicar migraciones
python manage.py migrate