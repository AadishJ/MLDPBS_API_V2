# Gunicorn configuration file
timeout = 300  # Increase timeout to 5 minutes (from default 30 seconds)
workers = 1  # Reduce number of workers to save memory
threads = 2  # Use threading instead of multiple processes
worker_class = "gthread"  # Thread-based workers
max_requests = 10  # Restart workers after handling 10 requests to free memory
max_requests_jitter = 5  # Add jitter to prevent all workers restarting at once
