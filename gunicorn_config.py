# Gunicorn configuration for production deployment
import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 10000)}"
backlog = 2048

# Worker processes
workers = 1  # Single worker for ML models to avoid memory issues
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 0
max_requests_jitter = 0
preload_app = True
timeout = 120
keepalive = 5

# Restart workers after this many requests, to help control memory usage
max_requests = 1000
max_requests_jitter = 100

# Logging
loglevel = "info"
accesslog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
errorlog = "-"

# Process naming
proc_name = "resume-extractor-api"

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None
