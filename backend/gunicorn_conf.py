# promptos_backend/gunicorn_conf.py
bind = "0.0.0.0:8000"
workers = 4
threads = 2
worker_class = "uvicorn.workers.UvicornWorker"
