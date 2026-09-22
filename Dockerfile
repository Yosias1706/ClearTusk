# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CLEARTUSK_RUNTIME_DIR=/var/lib/cleartusk

# libsndfile is required by soundfile; ffmpeg lets pydub read compressed uploads.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY cleartusk ./cleartusk
RUN pip install --upgrade pip && pip install ".[postgres,server]"

COPY alembic.ini wsgi.py ./
COPY alembic ./alembic

RUN useradd --system --create-home cleartusk \
 && mkdir -p /var/lib/cleartusk \
 && chown -R cleartusk:cleartusk /app /var/lib/cleartusk
USER cleartusk

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status == 200 else 1)"

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "180", "wsgi:app"]
