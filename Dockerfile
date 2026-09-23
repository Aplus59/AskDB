# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# Fail fast and log immediately: a container that buffers its output tells you
# nothing while it is starting up.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so editing source does not reinstall the world.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[api]"

# The databases are mounted at runtime rather than copied in. Mini-Dev alone
# is about 1.5 GB, which has no business inside an image that is otherwise a
# few hundred megabytes, and the data changes on a different schedule from
# the code.
ENV ASKDB_DATA_DIR=/data

# Nothing here needs to write to the filesystem.
RUN useradd --create-home --uid 10001 askdb
USER askdb

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "uvicorn", "askdb.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
