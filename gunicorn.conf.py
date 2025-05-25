# gunicorn.conf.py
workers = 2  # Adjust based on CPU cores (2 is safe for 1-2 GB RAM)
threads = 4  # Allows concurrent requests
timeout = 120  # Increase from 30s to 120s to handle long imports
max_requests = 1000  # Recycle workers to prevent memory leaks
max_requests_jitter = 100  # Randomize recycling
loglevel = 'info'  # Log level for debugging