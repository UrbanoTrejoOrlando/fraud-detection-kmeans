# gunicorn.conf.py
import multiprocessing

# Configuración de Gunicorn optimizada para Render
bind = "0.0.0.0:10000"
workers = 2  # Reducido para plan Free
threads = 2  # Reducido
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Aumentado a 120 segundos
keepalive = 5
spew = False
daemon = False
raw_env = [
    "DJANGO_SETTINGS_MODULE=fraud_detection.settings",
]
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None
errorlog = "-"
loglevel = "info"
accesslog = "-"
proc_name = "fraud_detection"